from collections import defaultdict
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
from astropy.time import Time
import numpy as np
import pandas as pd
from htc_job_history import (get_job_batch_ids, get_os_job_info,
                             plot_time_history, get_workflows, job_performance)


class JobInspector:
    def __init__(self, batch_name_substr, hours_back=None,
                 start_date=None, end_date=None, min_doc_count=5,
                 bps_job_label=None, verbose=True):
        self.df = {}
        self.job_batch_id = None
        self._task_types = {}
        self._fignum = 1
        self.df0 = get_workflows(batch_name_substr,
                                 hours_back=hours_back,
                                 start_date=start_date,
                                 end_date=end_date,
                                 bps_job_label=bps_job_label)
        if not self.df0.empty:
            self.df0 = self.df0.query(f"doc_count > {min_doc_count}")
            self.df0 = self.df0.sort_values("JobStartDate", ignore_index=True)
            columns = "JobBatchId doc_count JobBatchName".split()
            if verbose:
                print(self.df0[columns].tail(20))
        else:
            if verbose:
                print("No workflows found")

    def get_job_info(self, job_batch_ids, added_attributes=None):
        if not isinstance(job_batch_ids, (tuple, list)):
            job_batch_ids = [job_batch_ids]
        _ids = []
        for job_batch_id in job_batch_ids:
            if isinstance(job_batch_id, int):
                _ids.append(self.df0.iloc[job_batch_id]['JobBatchId'])
            else:
                _ids.append(job_batch_id)
        return pd.concat([get_os_job_info(_, added_attributes=added_attributes)
                          for _ in _ids])

    def plot(self, job_batch_id, fignum=1, target_task=None, oplot=False,
             gb_per_core=4.096, figsize=(10, 8), show_legend=True,
             refresh=False, added_attributes=None, timezone="US/Pacific",
             plot_memory_usage=True):
        if isinstance(job_batch_id, int):
            job_batch_id = self.df0.iloc[job_batch_id]["JobBatchId"]
            print(f"plotting data for {job_batch_id}")
        else:
            if (self.df0.empty or
                (job_batch_id not in self.df and
                 job_batch_id not in set(self.df0["JobBatchId"]))
            ):
                print(f"{job_batch_id} not found in current set")
        if job_batch_id not in self.df or refresh:
            self.df[job_batch_id] = get_os_job_info(
                job_batch_id, added_attributes=added_attributes)
        self.job_batch_id = job_batch_id
        try:
            job_batch_name \
                = (self.df0.query(f"JobBatchId == '{job_batch_id}'")
                   .iloc[0]["JobBatchName"])
        except (IndexError, pd.errors.UndefinedVariableError):
            job_batch_name = ""
        if target_task is not None:
            query = f"bps_job_label == '{target_task}'"
            df = self.df[job_batch_id].query(query)
            label = target_task
#            print(target_task, end=": ")
        else:
            df = self.df[job_batch_id]
            label = None
#            print("num jobs", end=": ")
#        print(len(df))

        if fignum not in plt.get_fignums():
            plt.figure(fignum, figsize=figsize)
        else:
            plt.figure(fignum)
        self._fignum = fignum
        if not oplot:
            plt.clf()
        if plot_memory_usage:
            plt.subplot(2, 1, 1)
        wall, artist = plot_time_history(df, weight_column="RequestCpus",
                                         alpha=1.0, label=label,
                                         timezone=timezone)
        color = artist.get_color()
        cpu, _ = plot_time_history(df, weight_column="cpu_efficiency",
                                   alpha=0.5, color=color, linestyle="--",
                                   timezone=timezone)
        if target_task is None:
            if plot_memory_usage:
                plt.title(f"cpu_efficiency = {cpu/wall:.2f}")
            else:
                plt.title(f"{job_batch_id}: {job_batch_name}")
            plt.xlabel(f"Time ({timezone})")
        plt.ylabel("concurrent processes")
        if target_task is None and plot_memory_usage:
            plt.subplot(2, 1, 2)
            mem_request, _ = plot_time_history(
                df, alpha=1.0, color=color,
                yfactor=np.ceil(df["memory_provisioned"]/gb_per_core),
                label="provisioned",
                timezone=timezone)
            mem_needed, _ =  plot_time_history(
                df, alpha=0.5, color=color, linestyle=":",
                yfactor=np.ceil(df["rss"].to_numpy()/gb_per_core),
                label="needed",
                timezone=timezone)
            rss, _ =  plot_time_history(
                df, alpha=0.5, color=color, linestyle="--",
                yfactor=df["rss"].to_numpy()/gb_per_core,
                label="rss-weighted",
                timezone=timezone)
            plt.title(f"memory efficiency = {mem_needed/mem_request:.2f}")
            plt.ylabel("core occupancy")
            plt.legend(fontsize='x-small')
            plt.suptitle(f"{job_batch_id}: {job_batch_name}")
            plt.xlabel(f"Time ({timezone})")
        if show_legend and target_task is not None:
            plt.legend(fontsize='x-small')
        plt.tight_layout()
        if target_task is None:
            self.overlay_tasks(show_legend=True, timezone=timezone,
                               plot_memory_usage=plot_memory_usage)
            df = job_performance(job_batch_id)
            print(df[df.columns[:-2]].to_string())

    def overlay_tasks(self, show_legend=False, timezone="UTC",
                      plot_memory_usage=True):
        for task_type in self.task_types():
            self.plot(self.job_batch_id, target_task=task_type, oplot=True,
                      show_legend=False, fignum=self._fignum, timezone=timezone,
                      plot_memory_usage=plot_memory_usage)
        if show_legend:
            plt.legend(fontsize=6, ncol=2)

    def task_types(self, job_batch_id=None):
        if job_batch_id in self._task_types:
            return self._task_types[job_batch_id]
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
        self._task_types[job_batch_id] = task_list
        return task_list

    def current_df(self):
        return self.df[self.job_batch_id]


if __name__ == '__main__':
    plt.ion()
