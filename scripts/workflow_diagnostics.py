"""
workflow_diagnostics.py

Per-user HTCondor workflow diagnostics across three dimensions:
  1. Job duration / short-job overhead
  2. CPU efficiency (relative to lsstsvc1 baseline)
  3. Memory efficiency and overrequest (relative to lsstsvc1 baseline)

Impact is ranked by equivalent node-hours, combining overhead core-hours and
memory waste TB-hours on a common scale derived from the actual hardware mix:
  - Pool: 110 nodes x 4 GB/core x 120 cores + 35 nodes x 6 GB/core x 120 cores
  - Weighted memory per node: 537.9 GB
  - 1 overhead core-hour  = 1/120          = 0.0083 node-hours
  - 1 TB-hour memory waste = 1/(537.9/1024) = 1.904  node-hours
  - equiv_node_hours = overhead_ch / 120 + mem_waste_tbh / 0.525

Requires environment variables:
  OPENSEARCH_HOST, OPENSEARCH_USER, OPENSEARCH_PASS

Usage:
  python workflow_diagnostics.py [--days N] [--output FILE]
"""

import os
import sys
import yaml
import argparse
from opensearchpy import OpenSearch
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDEX            = "htcondor-history-v1"
LOCAL_OVERHEAD_S = 60       # empirical HTCondor overhead per job for local submissions

# Hardware pool for node-hour conversions
# 110 nodes x 4 GB/core x 120 cores + 35 nodes x 6 GB/core x 120 cores
POOL_NODES_4GB   = 110
POOL_NODES_6GB   = 35
CORES_PER_NODE   = 120
MEM_GB_PER_NODE  = (POOL_NODES_4GB * 4 * CORES_PER_NODE +
                    POOL_NODES_6GB  * 6 * CORES_PER_NODE) / \
                   (POOL_NODES_4GB + POOL_NODES_6GB)        # 537.9 GB
MEM_TB_PER_NODE  = MEM_GB_PER_NODE / 1024                   # 0.5253 TB

# lsstsvc1 production baseline (update periodically as workflows are optimised)
BASELINE_CPU_EFF_P50    = 0.83
BASELINE_MEM_EFF_P50    = 0.48
BASELINE_MEM_OVERREQ_X  = 2.23

# Flagging thresholds
SHORT_JOB_MIN_M         = 5.0   # minutes
SHORT_JOB_FLAG_PCT      = 20.0  # % of jobs below SHORT_JOB_MIN_M to flag user
MIN_JOBS_TO_FLAG        = 50    # ignore users with fewer jobs than this
CPU_FLAG_VS_BASELINE    = 0.50  # flag if p50 cpu eff < 50% of baseline
MEM_OVERREQ_FLAG_X      = 5.0   # flag if mean overrequest > 5x (absolute)

# Per-run detail threshold: users above this node-hour impact get run breakdowns
NODE_HOUR_DETAIL_THRESH = 50    # node-hours

# Known special-case accounts to exclude from CPU flagging
KNOWN_SPECIAL = {"jkurla", "daues"}

# ---------------------------------------------------------------------------
# Node-hour helpers
# ---------------------------------------------------------------------------

def overhead_to_node_hours(overhead_ch):
    """Convert overhead core-hours to equivalent node-hours."""
    return overhead_ch / CORES_PER_NODE

def mem_waste_to_node_hours(waste_tbh):
    """Convert memory waste TB-hours to equivalent node-hours."""
    return waste_tbh / MEM_TB_PER_NODE

def combined_node_hours(overhead_ch, waste_tbh):
    """Combined impact in equivalent node-hours."""
    return overhead_to_node_hours(overhead_ch) + mem_waste_to_node_hours(waste_tbh)

def run_node_hours(est_ovhd_h, waste_tbh):
    """Per-run combined node-hours."""
    return combined_node_hours(est_ovhd_h, waste_tbh)

# ---------------------------------------------------------------------------
# Painless scripts (with field-existence guards)
# ---------------------------------------------------------------------------

CPU_EFF_SCRIPT = """
    if (!doc.containsKey('RemoteUserCpu')      || doc['RemoteUserCpu'].empty)      return null;
    if (!doc.containsKey('RemoteSysCpu')       || doc['RemoteSysCpu'].empty)       return null;
    if (!doc.containsKey('RemoteWallClockTime') || doc['RemoteWallClockTime'].empty) return null;
    if (!doc.containsKey('RequestCpus')        || doc['RequestCpus'].empty)        return null;
    double cpu  = doc['RemoteUserCpu'].value + doc['RemoteSysCpu'].value;
    double wall = doc['RemoteWallClockTime'].value * doc['RequestCpus'].value;
    return wall > 0 ? cpu / wall : null;
"""

MEM_EFF_SCRIPT = """
    if (!doc.containsKey('MemoryUsage')   || doc['MemoryUsage'].empty)   return null;
    if (!doc.containsKey('RequestMemory') || doc['RequestMemory'].empty) return null;
    double used = doc['MemoryUsage'].value;
    double req  = doc['RequestMemory'].value;
    return req > 0 ? used / req : null;
"""

MEM_WASTE_SCRIPT = """
    if (!doc.containsKey('MemoryUsage')       || doc['MemoryUsage'].empty)       return 0.0;
    if (!doc.containsKey('RequestMemory')     || doc['RequestMemory'].empty)     return 0.0;
    if (!doc.containsKey('RemoteWallClockTime')|| doc['RemoteWallClockTime'].empty) return 0.0;
    double used = doc['MemoryUsage'].value;
    double req  = doc['RequestMemory'].value;
    double wall = doc['RemoteWallClockTime'].value;
    return req > used ? (req - used) * wall / 3600.0 : 0.0;
"""

WAIT_SCRIPT = """
    doc['JobStartDate'].value.toInstant().epochSecond
    - doc['QDate'].value.toInstant().epochSecond
"""

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def make_client():
    with open("/sdf/home/j/jchiang/.lsst/secrets") as fobj:
        auth = list(yaml.safe_load(fobj).items())[0]

    host = "usdf-opensearch.slac.stanford.edu"
    port = 443
    client = OpenSearch(hosts=[{'host': host, "port": port}],
                        http_compress=True, http_auth=auth,
                        use_ssl=True, verify_certs=True,
                        timeout=120,
                        ssl_assert_hostname=False, ssl_show_warn=False)
    return client

# ---------------------------------------------------------------------------
# Search helper
# ---------------------------------------------------------------------------

def run_search(client, aggs, base_filter, extra_filter=None, timeout=120):
    filters = base_filter + (extra_filter or [])
    return client.search(
        index=INDEX,
        size=0,
        params={"timeout": timeout},
        body={
            "query": {"bool": {"must": filters}},
            "aggs": aggs,
        }
    )

# ---------------------------------------------------------------------------
# Per-user aggregations
# ---------------------------------------------------------------------------

def build_user_aggs():
    return {
        "by_user": {
            "terms": {"field": "Owner", "size": 200},
            "aggs": {
                # --- identity ---
                "job_count":   {"value_count": {"field": "Owner"}},
                "run_count":   {"cardinality": {"field": "JobBatchId"}},

                # --- duration / short-job ---
                "mean_rwct":   {"avg":  {"field": "RemoteWallClockTime"}},
                "p50_rwct":    {"percentiles": {"field": "RemoteWallClockTime",
                                                "percents": [50]}},
                "total_rwct":  {"sum":  {"field": "RemoteWallClockTime"}},
                "short_count": {"filter": {"range": {
                                    "RemoteWallClockTime": {"lt": SHORT_JOB_MIN_M * 60}}}},
                "vshort_count":{"filter": {"range": {
                                    "RemoteWallClockTime": {"lt": 120}}}},
                "mean_wait":   {"avg":  {"script": WAIT_SCRIPT}},
                "p90_wait":    {"percentiles": {"script": WAIT_SCRIPT,
                                               "percents": [90]}},

                # --- CPU efficiency ---
                "mean_cpu_eff":{"avg":  {"script": CPU_EFF_SCRIPT}},
                "p50_cpu_eff": {"percentiles": {"script": CPU_EFF_SCRIPT,
                                               "percents": [25, 50, 75]}},
                "low_cpu_count": {"filter": {"bool": {"must": [
                    {"exists": {"field": "RemoteUserCpu"}},
                    {"exists": {"field": "RemoteSysCpu"}},
                    {"exists": {"field": "RemoteWallClockTime"}},
                    {"exists": {"field": "RequestCpus"}},
                    {"script": {"script":
                        "(doc['RemoteUserCpu'].value + doc['RemoteSysCpu'].value) "
                        "/ (doc['RemoteWallClockTime'].value * doc['RequestCpus'].value) < 0.30"
                    }}
                ]}}},
                "multicore_count": {"filter": {"bool": {"must": [
                    {"exists": {"field": "RequestCpus"}},
                    {"range": {"RequestCpus": {"gt": 1}}}
                ]}}},

                # --- memory efficiency ---
                "mean_mem_eff":       {"avg": {"script": MEM_EFF_SCRIPT}},
                "p50_mem_eff":        {"percentiles": {"script": MEM_EFF_SCRIPT,
                                                       "percents": [25, 50, 75]}},
                "mean_mem_used_mb":   {"avg": {"field": "MemoryUsage"}},
                "mean_mem_req_mb":    {"avg": {"field": "RequestMemory"}},
                "total_mem_waste_mbh":{"sum": {"script": MEM_WASTE_SCRIPT}},
                "low_mem_eff_count":  {"filter": {"bool": {"must": [
                    {"exists": {"field": "MemoryUsage"}},
                    {"exists": {"field": "RequestMemory"}},
                    {"script": {"script":
                        "doc['RequestMemory'].value > 0 && "
                        "doc['MemoryUsage'].value / doc['RequestMemory'].value < 0.20"
                    }}
                ]}}},
            }
        }
    }

# ---------------------------------------------------------------------------
# Flatten user aggregation response
# ---------------------------------------------------------------------------

def flatten_users(resp, overhead_s=LOCAL_OVERHEAD_S):
    rows = []
    for b in resp["aggregations"]["by_user"]["buckets"]:
        job_count     = b["job_count"]["value"]
        short_count   = b["short_count"]["doc_count"]
        vshort_count  = b["vshort_count"]["doc_count"]
        multicore_cnt = b["multicore_count"]["doc_count"]
        low_cpu_cnt   = b["low_cpu_count"]["doc_count"]
        low_mem_cnt   = b["low_mem_eff_count"]["doc_count"]
        total_rwct    = b["total_rwct"]["value"] or 0
        mem_waste_mbh = b["total_mem_waste_mbh"]["value"] or 0
        p50_cpu       = b["p50_cpu_eff"]["values"]
        p50_mem       = b["p50_mem_eff"]["values"]
        mean_mem_used = b["mean_mem_used_mb"]["value"] or 0
        mean_mem_req  = b["mean_mem_req_mb"]["value"] or 0
        overhead_h    = job_count * (short_count / job_count if job_count else 0) \
                        * overhead_s / 3600
        waste_tbh     = mem_waste_mbh / (1024 * 1024)

        rows.append({
            "owner":              b["key"],
            "job_count":          job_count,
            "run_count":          b["run_count"]["value"],

            # duration
            "mean_rwct_m":        (b["mean_rwct"]["value"] or 0) / 60,
            "p50_rwct_m":         (b["p50_rwct"]["values"]["50.0"] or 0) / 60,
            "total_rwct_h":       total_rwct / 3600,
            "short_pct":          100 * short_count  / job_count if job_count else 0,
            "vshort_pct":         100 * vshort_count / job_count if job_count else 0,
            "est_overhead_h":     overhead_h,
            "mean_wait_s":        b["mean_wait"]["value"],
            "p90_wait_s":         b["p90_wait"]["values"]["90.0"],

            # CPU
            "mean_cpu_eff":       b["mean_cpu_eff"]["value"],
            "p50_cpu_eff":        p50_cpu["50.0"],
            "p25_cpu_eff":        p50_cpu["25.0"],
            "low_cpu_pct":        100 * low_cpu_cnt   / job_count if job_count else 0,
            "multicore_pct":      100 * multicore_cnt / job_count if job_count else 0,

            # memory
            "mean_mem_eff":       b["mean_mem_eff"]["value"],
            "p50_mem_eff":        p50_mem["50.0"],
            "p25_mem_eff":        p50_mem["25.0"],
            "mean_mem_used_mb":   mean_mem_used,
            "mean_mem_req_mb":    mean_mem_req,
            "mem_overreq_x":      mean_mem_req / mean_mem_used if mean_mem_used > 0 else None,
            "mem_waste_tbh":      waste_tbh,
            "low_mem_eff_pct":    100 * low_mem_cnt / job_count if job_count else 0,
        })

    df = pd.DataFrame(rows)

    # derived columns
    df["cpu_vs_baseline"]    = df["p50_cpu_eff"]   / BASELINE_CPU_EFF_P50
    df["mem_vs_baseline"]    = df["p50_mem_eff"]   / BASELINE_MEM_EFF_P50
    df["overreq_vs_baseline"]= df["mem_overreq_x"] / BASELINE_MEM_OVERREQ_X
    df["mem_overreq_x"]      = df["mem_overreq_x"].fillna(0)

    # node-hour impact columns
    df["overhead_node_h"] = df["est_overhead_h"].apply(overhead_to_node_hours)
    df["mem_waste_node_h"]= df["mem_waste_tbh"].apply(mem_waste_to_node_hours)
    df["total_node_h"]    = df["overhead_node_h"] + df["mem_waste_node_h"]

    # flag columns
    df["flag_short"] = (
        (df["short_pct"] > SHORT_JOB_FLAG_PCT) &
        (df["job_count"] > MIN_JOBS_TO_FLAG)
    )
    df["flag_cpu"] = (
        (df["cpu_vs_baseline"] < CPU_FLAG_VS_BASELINE) &
        (df["multicore_pct"] < 50) &
        (df["job_count"] > MIN_JOBS_TO_FLAG) &
        (~df["owner"].isin(KNOWN_SPECIAL))
    )
    df["flag_mem"] = (
        (df["mem_overreq_x"] > MEM_OVERREQ_FLAG_X) &
        (df["job_count"] > MIN_JOBS_TO_FLAG)
    )
    df["flag_count"] = df[["flag_short", "flag_cpu", "flag_mem"]].sum(axis=1)

    return df

# ---------------------------------------------------------------------------
# Per-run detail aggregation (for flagged users)
# ---------------------------------------------------------------------------

def build_run_aggs():
    return {
        "by_run": {
            "terms": {"field": "JobBatchId", "size": 500},
            "aggs": {
                "job_count":    {"value_count": {"field": "JobBatchId"}},
                "run_start":    {"min": {"field": "JobStartDate"}},
                "run_end":      {"max": {"field": "CompletionDate"}},
                "mean_rwct":    {"avg": {"field": "RemoteWallClockTime"}},
                "total_rwct":   {"sum": {"field": "RemoteWallClockTime"}},
                "short_count":  {"filter": {"range": {
                                    "RemoteWallClockTime": {"lt": SHORT_JOB_MIN_M * 60}}}},
                "mean_cpu_eff": {"avg": {"script": CPU_EFF_SCRIPT}},
                "mean_mem_req": {"avg": {"field": "RequestMemory"}},
                "mean_mem_used":{"avg": {"field": "MemoryUsage"}},
                "mem_waste_mbh":{"sum": {"script": MEM_WASTE_SCRIPT}},
                "rwct_hist": {
                    "range": {
                        "field": "RemoteWallClockTime",
                        "ranges": [
                            {"to": 60},
                            {"from": 60,   "to": 120},
                            {"from": 120,  "to": 300},
                            {"from": 300,  "to": 600},
                            {"from": 600,  "to": 1200},
                            {"from": 1200, "to": 3600},
                            {"from": 3600},
                        ]
                    }
                },
                "mem_hist": {
                    "range": {
                        "field": "MemoryUsage",
                        "ranges": [
                            {"to": 500},
                            {"from": 500,  "to": 1000},
                            {"from": 1000, "to": 2000},
                            {"from": 2000, "to": 4000},
                            {"from": 4000, "to": 8000},
                            {"from": 8000},
                        ]
                    }
                },
            }
        }
    }

RWCT_LABELS = ["<1m", "1-2m", "2-5m", "5-10m", "10-20m", "20-60m", ">60m"]
MEM_LABELS  = ["<500MB", "500M-1G", "1-2G", "2-4G", "4-8G", ">8G"]

def flatten_runs(resp, overhead_s=LOCAL_OVERHEAD_S):
    buckets = resp["aggregations"]["by_run"]["buckets"]
    if not buckets:
        return pd.DataFrame()

    rows = []
    for b in buckets:
        job_count   = b["job_count"]["value"]
        run_start   = b["run_start"]["value"] / 1000
        run_end     = b["run_end"]["value"] / 1000
        run_wall    = run_end - run_start
        mean_rwct   = b["mean_rwct"]["value"] or 0
        total_rwct  = b["total_rwct"]["value"] or 0
        short_count = b["short_count"]["doc_count"]
        mem_req     = b["mean_mem_req"]["value"] or 0
        mem_used    = b["mean_mem_used"]["value"] or 0
        waste_tbh   = (b["mem_waste_mbh"]["value"] or 0) / (1024 * 1024)
        ovhd_h      = short_count * overhead_s / 3600

        rwct_profile = " | ".join(
            f"{RWCT_LABELS[i]}:{h['doc_count']}"
            for i, h in enumerate(b["rwct_hist"]["buckets"])
            if h["doc_count"] > 0
        )
        mem_profile = " | ".join(
            f"{MEM_LABELS[i]}:{h['doc_count']}"
            for i, h in enumerate(b["mem_hist"]["buckets"])
            if h["doc_count"] > 0
        )

        rows.append({
            "run_id":        b["key"],
            "job_count":     job_count,
            "run_wall_m":    run_wall / 60,
            "mean_rwct_m":   mean_rwct / 60,
            "short_pct":     100 * short_count / job_count if job_count else 0,
            "est_ovhd_h":    ovhd_h,
            "cpu_eff":       b["mean_cpu_eff"]["value"],
            "mem_req_mb":    mem_req,
            "mem_used_mb":   mem_used,
            "overreq_x":     mem_req / mem_used if mem_used > 0 else None,
            "waste_tbh":     waste_tbh,
            "node_hours":    run_node_hours(ovhd_h, waste_tbh),
            "rwct_profile":  rwct_profile,
            "mem_profile":   mem_profile,
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def fmt(df):
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 120)
    return df

def section(title):
    print(f"\n{'='*len(title)}")
    print(title)
    print('='*len(title))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30,
                        help="Survey window in days (default: 30)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional output file path")
    args = parser.parse_args()

    if args.output:
        sys.stdout = open(args.output, "w")

    client = make_client()

    base_filter = [
        {"range": {"CompletionDate": {"gte": f"now-{args.days}d"}}},
        {"range": {"RemoteWallClockTime": {"gt": 0}}},
    ]

    # -----------------------------------------------------------------------
    # Per-user query
    # -----------------------------------------------------------------------
    print(f"Fetching per-user metrics ({args.days}-day window)...")
    user_resp = run_search(client, build_user_aggs(), base_filter)
    df = fmt(flatten_users(user_resp))

    # -----------------------------------------------------------------------
    # Facility-wide summary
    # -----------------------------------------------------------------------
    total_jobs    = df["job_count"].sum()
    total_short   = (df["job_count"] * df["short_pct"] / 100).sum()
    total_rwct    = df["total_rwct_h"].sum()
    total_ovhd    = df["est_overhead_h"].sum()
    total_waste   = df["mem_waste_tbh"].sum()
    total_node_h  = df["total_node_h"].sum()

    section(f"Facility-wide summary ({args.days}-day window)")
    print(f"  Total jobs:                  {total_jobs:>12,.0f}")
    print(f"  Short jobs (<{SHORT_JOB_MIN_M:.0f} min):        "
          f"{total_short:>12,.0f}  ({100*total_short/total_jobs:.1f}%)")
    print(f"  Total payload wall time:     {total_rwct:>12,.1f}  core-hours")
    print(f"  Est. HTCondor overhead:      {total_ovhd:>12,.1f}  core-hours  "
          f"({100*total_ovhd/total_rwct:.1f}% of payload)")
    print(f"  Est. memory waste:           {total_waste:>12,.1f}  TB-hours")
    print(f"  Combined impact:             {total_node_h:>12,.1f}  equiv. node-hours")
    print(f"\n  Node-hour conversion (pool: {POOL_NODES_4GB}x4GB + {POOL_NODES_6GB}x6GB nodes, "
          f"{CORES_PER_NODE} cores/node, {MEM_GB_PER_NODE:.1f} GB/node avg):")
    print(f"    1 overhead core-hour  = {1/CORES_PER_NODE:.4f} node-hours")
    print(f"    1 TB-hour mem waste   = {1/MEM_TB_PER_NODE:.3f} node-hours")
    print(f"\n  lsstsvc1 baseline (reference):")
    print(f"    CPU efficiency p50:   {BASELINE_CPU_EFF_P50:.2f}")
    print(f"    Memory eff p50:       {BASELINE_MEM_EFF_P50:.2f}")
    print(f"    Memory overrequest:   {BASELINE_MEM_OVERREQ_X:.2f}x")

    # -----------------------------------------------------------------------
    # Unified per-user summary — sorted by total_node_h
    # -----------------------------------------------------------------------
    section("Per-user summary: all three dimensions (sorted by equiv. node-hours)")
    cols = [
        "owner", "job_count", "run_count",
        "p50_rwct_m", "short_pct", "est_overhead_h",
        "p50_cpu_eff", "cpu_vs_baseline",
        "mem_overreq_x", "mem_waste_tbh",
        "overhead_node_h", "mem_waste_node_h", "total_node_h",
        "flag_short", "flag_cpu", "flag_mem", "flag_count",
    ]
    print(df[cols].sort_values("total_node_h", ascending=False).to_string(index=False))

    # -----------------------------------------------------------------------
    # Duration flagged — sorted by node-hours
    # -----------------------------------------------------------------------
    section(f"Flagged: short jobs > {SHORT_JOB_FLAG_PCT:.0f}% and > {MIN_JOBS_TO_FLAG} jobs "
            f"(sorted by node-hours)")
    dur_cols = ["owner", "job_count", "run_count", "p50_rwct_m", "mean_rwct_m",
                "short_pct", "vshort_pct", "est_overhead_h", "overhead_node_h"]
    flagged_dur = df[df["flag_short"]][dur_cols].sort_values("overhead_node_h", ascending=False)
    print(flagged_dur.to_string(index=False))

    # -----------------------------------------------------------------------
    # CPU flagged — sorted by node-hours
    # -----------------------------------------------------------------------
    section(f"Flagged: CPU efficiency p50 < {CPU_FLAG_VS_BASELINE*100:.0f}% of baseline "
            f"({CPU_FLAG_VS_BASELINE * BASELINE_CPU_EFF_P50:.2f}), multicore < 50% "
            f"(sorted by node-hours)")
    cpu_cols = ["owner", "job_count", "run_count", "p50_cpu_eff", "p25_cpu_eff",
                "cpu_vs_baseline", "low_cpu_pct", "multicore_pct", "total_node_h"]
    flagged_cpu = df[df["flag_cpu"]][cpu_cols].sort_values("total_node_h", ascending=False)
    print(flagged_cpu.to_string(index=False))

    # -----------------------------------------------------------------------
    # Memory flagged — sorted by node-hours
    # -----------------------------------------------------------------------
    section(f"Flagged: memory overrequest > {MEM_OVERREQ_FLAG_X:.0f}x and > {MIN_JOBS_TO_FLAG} jobs "
            f"(sorted by node-hours)")
    mem_cols = ["owner", "job_count", "run_count", "mean_mem_req_mb", "mean_mem_used_mb",
                "mem_overreq_x", "overreq_vs_baseline", "p50_mem_eff",
                "mem_waste_tbh", "mem_waste_node_h"]
    flagged_mem = df[df["flag_mem"]][mem_cols].sort_values("mem_waste_node_h", ascending=False)
    print(flagged_mem.to_string(index=False))

    # -----------------------------------------------------------------------
    # Users flagged on multiple dimensions — sorted by node-hours
    # -----------------------------------------------------------------------
    section("Users flagged on 2+ dimensions (sorted by node-hours)")
    multi_cols = ["owner", "job_count", "est_overhead_h", "p50_cpu_eff",
                  "cpu_vs_baseline", "mem_overreq_x", "mem_waste_tbh",
                  "total_node_h", "flag_short", "flag_cpu", "flag_mem", "flag_count"]
    multi_flagged = df[df["flag_count"] >= 2][multi_cols].sort_values(
        ["flag_count", "total_node_h"], ascending=[False, False])
    print(multi_flagged.to_string(index=False))

    # -----------------------------------------------------------------------
    # Per-run detail: users above NODE_HOUR_DETAIL_THRESH
    # All flagged users above threshold get a combined run-level breakdown
    # sorted by node-hours, showing both overhead and memory waste dimensions
    # -----------------------------------------------------------------------
    detail_users = df[
        (df["flag_count"] >= 1) &
        (df["total_node_h"] >= NODE_HOUR_DETAIL_THRESH)
    ].sort_values("total_node_h", ascending=False)["owner"].tolist()

    if detail_users:
        section(f"Per-run detail: flagged users with >= {NODE_HOUR_DETAIL_THRESH} equiv. node-hours")
        print(f"  Users: {detail_users}\n")
        for owner in detail_users:
            resp = run_search(client, build_run_aggs(), base_filter,
                              extra_filter=[{"term": {"Owner": owner}}])
            run_df = flatten_runs(resp)
            if run_df.empty:
                print(f"--- {owner} --- (no JobBatchId data available)\n")
                continue

            run_df = run_df.sort_values("node_hours", ascending=False)
            run_cols = ["run_id", "job_count", "run_wall_m", "mean_rwct_m",
                        "short_pct", "est_ovhd_h", "cpu_eff",
                        "mem_req_mb", "mem_used_mb", "overreq_x",
                        "waste_tbh", "node_hours", "rwct_profile"]
            print(f"--- {owner} ---")
            print(fmt(run_df)[run_cols].to_string(index=False))
            print()

    if args.output:
        sys.stdout.close()

if __name__ == "__main__":
    main()
