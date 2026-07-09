#!/usr/bin/env python3
"""
Query OpenSearch for failed HTCondor jobs within a given JobBatchId.

Failed jobs are defined as:
  - JobStatus 3: Removed
  - JobStatus 5: Held
  - JobStatus 4 + ExitCode > 0: Completed with non-zero exit code
  - JobStatus 4 + ExitBySignal True: Completed but killed by a signal
"""

import argparse
import json
import yaml
from opensearchpy import OpenSearch, RequestsHttpConnection

# ---------------------------------------------------------------------------
# Connection defaults — override via CLI args or environment as needed
# ---------------------------------------------------------------------------
DEFAULT_INDEX = "htcondor-history-v1"


def build_client() -> OpenSearch:

    with open("/sdf/home/j/jchiang/.lsst/secrets") as fobj:
        auth = list(yaml.safe_load(fobj).items())[0]

    host = "usdf-opensearch.slac.stanford.edu"
    port = 443
    return OpenSearch(hosts=[{'host': host, "port": port}],
                      http_compress=True, http_auth=auth,
                      use_ssl=True, verify_certs=True,
                      ssl_assert_hostname=False, ssl_show_warn=False)


def build_query(job_batch_id: str, page_size: int) -> dict:
    """
    Return an OpenSearch query dict that finds all failed jobs for the
    given JobBatchId.

    JobStatus values:
      1 = Idle, 2 = Running, 3 = Removed, 4 = Completed, 5 = Held, 6 = Transferring Output
    """
    return {
        "size": page_size,
        "query": {
            "bool": {
                # Must match the batch
                "filter": [
                    {
                        "term": {
                            "JobBatchId.keyword": job_batch_id
                        }
                    }
                ],
                # At least one failure condition must be true
                "should": [
                    # Removed (condor_rm or system-initiated)
                    {"term": {"JobStatus": 3}},
                    # Held (unrecoverable error requiring human attention)
                    {"term": {"JobStatus": 5}},
                    # Completed but exited with a non-zero return code
                    {
                        "bool": {
                            "must": [
                                {"term": {"JobStatus": 4}},
                                {"range": {"ExitCode": {"gt": 0}}},
                            ]
                        }
                    },
                    # Completed but killed by a signal (crash, OOM, etc.)
                    {
                        "bool": {
                            "must": [
                                {"term": {"JobStatus": 4}},
                                {"term": {"ExitBySignal": True}},
                            ]
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        },
        # Return the most useful diagnostic fields
        "_source": [
            "ClusterId",
            "ProcId",
            "JobBatchId",
            "JobStatus",
            "ExitCode",
            "ExitBySignal",
            "ExitSignal",
            "HoldReason",
            "HoldReasonCode",
            "RemoveReason",
            "Owner",
            "Cmd",
            "QDate",
            "CompletionDate",
        ],
        "sort": [
            {"ClusterId": {"order": "asc"}},
            {"ProcId":    {"order": "asc"}},
        ],
    }


JOB_STATUS_LABELS = {
    1: "Idle",
    2: "Running",
    3: "Removed",
    4: "Completed",
    5: "Held",
    6: "Transferring Output",
}

FAILURE_REASON_LABELS = {
    3: "Removed",
    5: "Held",
}


def classify_failure(source: dict) -> str:
    status = source.get("JobStatus")
    if status == 3:
        return f"Removed — {source.get('RemoveReason', 'no reason recorded')}"
    if status == 5:
        return (
            f"Held (code {source.get('HoldReasonCode', '?')}) "
            f"— {source.get('HoldReason', 'no reason recorded')}"
        )
    if status == 4:
        if source.get("ExitBySignal"):
            return f"Killed by signal {source.get('ExitSignal', '?')}"
        exit_code = source.get("ExitCode", 0)
        return f"Non-zero exit code {exit_code}"
    return "Unknown failure"


def fetch_failed_jobs(client: OpenSearch, index: str, job_batch_id: str,
                      page_size: int = 100) -> list[dict]:
    """Paginate through all failed jobs for the given JobBatchId."""
    query = build_query(job_batch_id, page_size)
    all_hits: list[dict] = []
    offset = 0

    while True:
        query["from"] = offset
        response = client.search(index=index, body=query)
        hits = response["hits"]["hits"]
        if not hits:
            break
        all_hits.extend(hits)
        offset += len(hits)
        total = response["hits"]["total"]["value"]
        if offset >= total:
            break

    return all_hits


def print_results(hits: list[dict], as_json: bool) -> None:
    if as_json:
        records = [h["_source"] for h in hits]
        print(json.dumps(records, indent=2, default=str))
        return

    if not hits:
        print("No failed jobs found.")
        return

    print(f"{'Job ID':<20} {'Status':<22} {'Owner':<15} {'Failure Reason'}")
    print("-" * 100)
    for hit in hits:
        src = hit["_source"]
        cluster = src.get("ClusterId", "?")
        proc = src.get("ProcId", "?")
        job_id = f"{cluster}.{proc}"
        status_label = JOB_STATUS_LABELS.get(src.get("JobStatus"), "Unknown")
        owner = src.get("Owner", "?")
        reason = classify_failure(src)
        print(f"{job_id:<20} {status_label:<22} {owner:<15} {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find failed HTCondor jobs in OpenSearch by JobBatchId."
    )
    parser.add_argument("job_batch_id", help="The JobBatchId to search for")
    parser.add_argument("--index",   default=DEFAULT_INDEX, help="Index or index pattern")
    parser.add_argument("--page-size", default=100, type=int,
                        help="Number of results to fetch per page (default 100)")
    parser.add_argument("--json",    action="store_true",
                        help="Output results as JSON instead of a table")
    parser.set_defaults(verify_certs=True)
    args = parser.parse_args()

    client = build_client()

    hits = fetch_failed_jobs(client, args.index, args.job_batch_id, args.page_size)
    print(f"Found {len(hits)} failed job(s) for JobBatchId='{args.job_batch_id}'\n")
    print_results(hits, as_json=args.json)


if __name__ == "__main__":
    main()
