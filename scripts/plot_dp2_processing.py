import os
import matplotlib.pyplot as plt
import pandas as pd
from htc_job_history import (get_job_batch_ids, get_os_job_info,
                             plot_time_history)


df0 = get_job_batch_ids("DM-53881", "2026-01-20", "2026-06-02")
print(len(df0))
selection = (df0['JobBatchName'].str.startswith("LSSTCam")
             | df0['JobBatchName'].str.contains("hips"))
df0 = df0[selection]
print(len(df0))

plt.ion()
plt.figure(1, figsize=(10, 4))
plt.clf()

for color, stage in zip(("red", "green", "blue", "orange", "cyan"),
                        (
                            "stage1",
                            "stage2",
                            "stage3",
                            "stage4",
                            "hips",
                        )):
    job_info_file = f"{stage}_job_info.parquet"
    if not os.path.isfile(job_info_file):
        if stage == "hips":
            selection = df0['JobBatchName'].str.contains(stage)
        else:
            selection = (df0['JobBatchName'].str.contains(stage)
                         & ~df0['JobBatchName'].str.contains("hips"))
        df = df0[selection]
        print(stage, len(df))
        dfs = []
        for i, (_, row) in enumerate(df.iterrows()):
            job_batch_id = row['JobBatchId']
            job_batch_name = row['JobBatchName']
            print(i, job_batch_id)
            dfs.append(get_os_job_info(job_batch_id))
        my_df = pd.concat(dfs)
        # Omit jobs that have not finished
        my_df = my_df.query("CompletionDate == CompletionDate")
        my_df.to_parquet(job_info_file)
    else:
        my_df = pd.read_parquet(job_info_file)
    my_df.to_parquet(job_info_file)
    wall_time = plot_time_history(my_df, label=stage, color=color,
                                  weight_column="RequestCpus", alpha=1.0)
    cpu_time = plot_time_history(my_df, label=f"{stage}, cpu weighted",
                                 color=color, weight_column="cpu_efficiency",
                                 alpha=0.5, make_plot=False)
    print(f"{stage} {cpu_time:.1e} {wall_time:.1e} {cpu_time/wall_time:.2f}")

plt.legend(fontsize='x-small')
