from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from astropy.time import Time
import numpy as np
from htc_job_history import get_job_batch_ids, get_os_job_info, plot_time_history

class JobInspector:
    def __init__(self, batch_name_substr, start_date=None, end_date=None,
                 hours_back=None):
        if end_date is None:
            end_date = datetime.now().isoformat()
        if start_date is None:
            dt = timedelta(hours=hours_back)
            start_date = Time(end_date, format="isot").datetime - dt
            start_date = start_date.isoformat()
        self.df0 = get_job_batch_ids(batch_name_substr, start_date, end_date)
        if self.df0.empty:
            raise ValueError("No workflows found")
        self.df0 = self.df0.sort_values("JobStartDate", ignore_index=True)
        print(self.df0.tail(20))
        self.df = {}
        self.job_batch_id = None
        self._task_types = None

    def plot_job_batch(self, job_batch_id, fignum=1, target_task=None,
                       oplot=False, gb_per_core=4.0, figsize=(10, 8)):
        if isinstance(job_batch_id, int):
            job_batch_id = self.df0.iloc[job_batch_id]["JobBatchId"]
            print(f"plotting data for {job_batch_id}")
        else:
            if (job_batch_id not in self.df and
                job_batch_id not in set(self.df0["JobBatchId"])):
                print(f"{job_batch_id} not found in current set")
        if job_batch_id not in self.df:
            self.df[job_batch_id] = get_os_job_info(job_batch_id)
            self._task_types = None
        self.job_batch_id = job_batch_id
        if target_task is not None:
            query = f"bps_job_label == '{target_task}'"
            df = self.df[job_batch_id].query(query)
            label = target_task
            print(target_task, end=": ")
        else:
            df = self.df[job_batch_id]
            label = None
            print("num jobs", end=": ")
        print(len(df))

        if fignum not in plt.get_fignums():
            plt.figure(fignum, figsize=figsize)
        else:
            plt.figure(fignum)
        if not oplot:
            plt.clf()
        plt.subplot(2, 1, 1)
        wall, artist = plot_time_history(df, weight_column="RequestCpus",
                                         alpha=1.0, label=label)
        color = artist.get_color()
        cpu, _ = plot_time_history(df, weight_column="cpu_efficiency",
                                   alpha=1.0, color=color, linestyle="--")
        if target_task is None:
            plt.title(f"cpu_efficiency = {cpu/wall:.2f}")
        plt.ylabel("concurrent processes")
        if target_task is None:
            plt.subplot(2, 1, 2)
            mem_request, _ = plot_time_history(
                df, weight_column="RequestCpus", alpha=1.0, color=color,
                yfactor=np.ceil(df["memory_provisioned"].to_numpy()/gb_per_core),
                label="provisioned")
            rss, _ =  plot_time_history(df, weight_column="RequestCpus", alpha=1.0,
                                        color=color, linestyle="--",
                                        yfactor=df["rss"].to_numpy()/gb_per_core,
                                        label="rss-weighted")
            plt.title(f"memory efficiency = {rss/mem_request:.2f}")
            plt.ylabel("core occupancy")
            plt.legend(fontsize='x-small')
            plt.suptitle(f"JobBatchID: {job_batch_id}")
        plt.xlabel("Time (PT)")
        if target_task is not None:
            plt.legend(fontsize='x-small')
        plt.tight_layout()

    def overlay_tasks(self):
        for task_type in self.task_types():
            self.plot_job_batch(self.job_batch_id, target_task=task_type,
                                oplot=True)

    def task_types(self, job_batch_id=None):
        if self._task_types is not None:
            return self._task_types
        if job_batch_id is None:
            job_batch_id = self.job_batch_id
        df = self.df[job_batch_id].sort_values("JobStartDate")
        tasks = df['bps_job_label'].to_list()
        num_tasks = len(set(tasks).difference({None}))
        task_list = []
        i = 0
        while len(task_list) < num_tasks:
            if tasks[i] is not None and tasks[i] not in task_list:
                task_list.append(tasks[i])
            i += 1
        self._task_types = task_list
        return task_list


if __name__ == '__main__':
    plt.ion()
