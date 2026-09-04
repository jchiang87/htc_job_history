import sys
from htc_job_history import get_workflows, job_performance


def job_performance_tables(jira_ticket, hours_back=21*24, min_doc_count=5):
    df0 = get_workflows(jira_ticket, hours_back=hours_back)
    df0 = df0.query(f"doc_count >= {min_doc_count}")
    df0 = df0.sort_values("JobStartDate", ignore_index=True)
    tables = {}
    for job_batch_id, job_batch_name in zip(df0['JobBatchId'],
                                            df0['JobBatchName']):
        print(job_batch_id, job_batch_name)
        try:
            df1 = job_performance(job_batch_id)
        except KeyError as eobj:
            print(eobj)
        else:
            print(df1[df1.columns[:-1]].to_string())
        print()
        tables[job_batch_id] = df1
    return tables


if __name__ == '__main__':
    jira_ticket = sys.argv[1]
    tables = job_performance_tables(jira_ticket)
