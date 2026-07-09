from collections import defaultdict
import multiprocessing
import pandas as pd
import htcondor


def gather_info(args):
    schedd_ad, kwargs = args
    schedd = htcondor.Schedd(schedd_ad)
    result = list(schedd.query(**kwargs))
    result.extend(list(schedd.history(**kwargs)))
    data = defaultdict(list)
    if result:
        for classad in result:
            if not classad or "JobBatchId" not in classad:
                continue
            for key in kwargs["projection"]:
                data[key].append(classad[key])
    return pd.DataFrame(data)


def find_job_batch_ids(user):
    # Get a list of schedds from collector
    collector = htcondor.Collector()
    schedds = {_['Name']: _ for _ in
               collector.locateAll(htcondor.DaemonTypes.Schedd)}
    # Keyword arguments to pass to schedd.query and schedd.history
    kwargs = dict(
        projection=[
            "ClusterId",
            "ProcId",
            "JobStatus",
            "Owner",
            "Cmd",
            "JobBatchId",
            "JobBatchName"
        ],
        constraint=f'Owner == \"{user}\"'
    )

    # Loop over schedds and gather info
    args = [(_, kwargs) for _ in schedds.values()]
    with multiprocessing.Pool(processes=len(schedds)) as pool:
        dfs = [_ for _ in pool.map(gather_info, args) if not _.empty]

    if not dfs:
        print(f"No workflows found for {user}.")
        return None
    df0 = pd.concat(dfs).sort_values("JobBatchId")
    print(df0[["JobBatchId", "JobBatchName"]]
          .drop_duplicates(ignore_index=True))
    return df0


if __name__ == '__main__':
    import sys
    user = sys.argv[1]
    df0 = find_job_batch_ids(user)
