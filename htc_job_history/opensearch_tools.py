import yaml
from collections import defaultdict
from datetime import datetime, timezone
import pandas as pd
from opensearchpy import OpenSearch


__all__ = ["get_job_batch_ids", "get_os_job_info"]


with open("/sdf/home/j/jchiang/.lsst/secrets") as fobj:
    auth = list(yaml.safe_load(fobj).items())[0]

host = "usdf-opensearch.slac.stanford.edu"
port = 443
OSCLIENT = OpenSearch(hosts=[{'host': host, "port": port}],
                      http_compress=True, http_auth=auth,
                      use_ssl=True, verify_certs=True,
                      ssl_assert_hostname=False, ssl_show_warn=False)


def get_job_batch_ids(batch_name_substr, start_date, end_date,
                      index="htcondor-history-v1"):
    body = {
        "query": {
            "bool": {
                "must": [
                    {"wildcard": {"JobBatchName": f"*{batch_name_substr}*"}}
                ],
                "filter": [
                    {"range": {"JobStartDate": {
                        "gte": start_date,
                        "lte": end_date,
                        "format": "strict_date_optional_time"
                    }}}
                ]
            }
        },
        "aggs": {
            "batches": {
                "terms": {"field": "JobBatchId", "size": 10000},
                "aggs": {
                    "start":      {"min": {"field": "JobStartDate"}},
                    "completion": {"max": {"field": "CompletionDate"}},
                    "name":       {"terms": {"field": "JobBatchName", "size": 1}}
                }
            }
        },
        "size": 0
    }

    response = OSCLIENT.search(body=body, index=index)

    data = defaultdict(list)
    for bucket in response['aggregations']['batches']['buckets']:
        if bucket['completion']['value'] is None:
            continue
        data['JobBatchId'].append(str(bucket['key']))
        start = datetime.fromtimestamp(
            int(bucket['start']['value_as_string']), tz=timezone.utc)
        data['JobStartDate'].append(start)
        completion = datetime.fromtimestamp(
            int(bucket['completion']['value_as_string']), tz=timezone.utc)
        data['CompletionDate'].append(completion)
        data['JobBatchName'].append(bucket['name']['buckets'][0]['key'])
    return pd.DataFrame(data)


def get_os_job_info(job_batch_id, index="htcondor-history-v1", size=10000):
    columns = ["JobBatchId", "ClusterId", "ProcId", "JobStartDate",
               "JobCurrentStartDate", "Iwd",
               "CompletionDate", "JobStatus", "bps_job_name",
               "bps_job_label", "StartdName", "ExitCode", "Err", "QDate",
               "RequestCpus", "RemoteUserCpu", "RemoteSysCpu",
               "RemoteWallClockTime", "ResidentSetSize", "RequestMemory",
               "MemoryProvisioned"]

    body = {
        'query': {
            'bool': {
                'filter': [
                    {'terms': {'JobBatchId': [job_batch_id]}},
                ]
            }
        },
        '_source': columns
    }

    response = OSCLIENT.search(
        size=size,
        body=body,
        index=index,
        scroll="1m",
    )

    data = defaultdict(list)
    scroll_id = response['_scroll_id']
    hits = response['hits']['hits']
    while len(hits) > 0:
        for item in hits:
            row = item['_source']
            if (row["RemoteWallClockTime"] <= 0 or
                "bps_job_label" not in row):
                continue
            for column in columns:
                if column not in row:
                    value = None
                else:
                    value = row[column]
                data[column].append(value)
            start_ts = row.get('JobStartDate', 0)
            start_dt = datetime.fromtimestamp(start_ts) if start_ts else None
            data['start_dt'].append(start_dt)
            end_ts = row.get('CompletionDate', 0)
            end_dt = datetime.fromtimestamp(end_ts) if end_ts else None
            data['end_dt'].append(end_dt)
            # CPU efficiency calculation
            cpu_efficiency = ((row["RemoteUserCpu"] + row["RemoteSysCpu"]) /
                              row["RemoteWallClockTime"])
            data['cpu_efficiency'].append(cpu_efficiency)
            data['memory_request'].append(row["RequestMemory"]/1e3)  # GB
            data['memory_provisioned'].append(row["MemoryProvisioned"]/1e3)  # GB
            data['rss'].append(row["ResidentSetSize"]/1e6)  # GB
        response = OSCLIENT.scroll(
            scroll_id=scroll_id,
            scroll="1m",
        )
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
    df0 = pd.DataFrame(data)
    return df0
