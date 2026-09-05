#!/usr/bin/env python
from collections import defaultdict
import click
import matplotlib.pyplot as plt
import pandas as pd
from htc_job_history import (get_workflows, job_performance, plot_time_history,
                             get_os_job_info, PipelineStageClassifier)


STAGE_CLASSIFIER = PipelineStageClassifier()


class JobPerformanceTables:
    def __init__(self, jira_ticket, hours_back=21*24, min_doc_count=5):
        self.jira_ticket = jira_ticket
        df0 = get_workflows(jira_ticket, hours_back=hours_back)
        df0 = df0.query(f"doc_count >= {min_doc_count}")
        self.df0 = df0.sort_values("JobStartDate", ignore_index=True)
        self._job_info = {}
        self._job_batch_id_stage = {}

    def job_info(self, job_batch_id):
        if job_batch_id not in self._job_info:
            self._job_info[job_batch_id] = get_os_job_info(job_batch_id)
        return self._job_info[job_batch_id]

    def job_batch_id_stage(self, job_batch_id):
        if job_batch_id not in self._job_batch_id_stage:
            df = self.job_info(job_batch_id)
            current_stage = STAGE_CLASSIFIER.classify(set(df['bps_job_label']))
            self._job_batch_id_stage[job_batch_id] = current_stage
        return self._job_batch_id_stage[job_batch_id]

    def print_tables(self, roll_up=True):
        if roll_up:
            df0 = pd.concat([self.job_info(_) for _ in self.df0['JobBatchId']])
            df1 = job_performance(df0)
            print(self.jira_ticket)
            print(df1[df1.columns[:-1]].head(20).to_string())
            print()
            return df1
        dfs = []
        for job_batch_id, job_batch_name in zip(self.df0['JobBatchId'],
                                                self.df0['JobBatchName']):
            try:
                df1 = job_performance(self.job_info[job_batch_id])
            except KeyError:
                continue
            print(job_batch_id, job_batch_name)
            print(df1[df1.columns[:-1]].to_string())
            print()
            df1['JobBatchId'] = job_batch_id
            dfs.append(df1)
        return pd.concat(dfs)

    def plot_processing_history(self, notes="", timezone="US/Pacific"):
        dfs = []
        for job_batch_id in self.df0['JobBatchId']:
            df = get_os_job_info(job_batch_id)
            current_stage = STAGE_CLASSIFIER.classify(set(df['bps_job_label']))
            self._job_batch_id_stage[job_batch_id] = current_stage
            df['stage'] = current_stage
            dfs.append(df)
        df1 = pd.concat(dfs)
        stages = sorted(set(df1['stage']))
        for stage in stages:
            df = df1.query(f"stage=='{stage}'")
            plot_time_history(df, weight_column="RequestCpus",
                              timezone=timezone, label=stage)
        plt.legend(fontsize='x-small')
        plt.xlabel(f"Time ({timezone})")
        plt.ylabel("concurrent processes")
        plt.title(f"{self.jira_ticket}: {notes}")
        return df1

    def print_db_job_stats(self):
        data = defaultdict(list)
        for job_batch_id in sorted(set(self.df0['JobBatchId'])):
            df0 = self.job_info(job_batch_id)
            for job_name in ("buildQuantumGraph", "finalJob"):
                data['job_batch_id'].append(job_batch_id)
                data['stage'].append(self.job_batch_id_stage(job_batch_id))
                data['job_name'].append(job_name)
                df1 = df0.query(f"bps_job_label=='{job_name}'")
                if df1.empty:
                    data['wall_time'].append(0)
                    data['wait_time'].append(0)
                else:
                    data['wall_time'].append(df1.iloc[0]['wall_time'])
                    data['wait_time'].append(df1.iloc[0]['wait_time'])
        df2 = pd.DataFrame(data)
        data = defaultdict(list)
        wait_time_names = {"buildQuantumGraph": "bQG_wait_time",
                           "finalJob": "fJ_wait_time"}
        for stage in sorted(set(df2['stage'])):
            data['stage'].append(stage)
            for job_name in sorted(set(df2['job_name'])):
                df3 = df2.query(f"stage=='{stage}' and job_name=='{job_name}'")
                data[job_name].append(sum(df3['wall_time'])/60.)
                data[wait_time_names[job_name]].append(sum(df3['wait_time'])/60.)
        print("Database times (min)")
        with pd.option_context("display.precision", 0):
            print(pd.DataFrame(data).to_string())
        print()

@click.command()
@click.argument("jira_ticket")
@click.option("--notes", default="", help="Ticket notes")
@click.option("--fignum", default=1, help="Figure number")
def make_tables(jira_ticket, notes, fignum):
    plt.ion()
    fignum = int(fignum)
    if fignum not in plt.get_fignums():
        plt.figure(fignum, figsize=(12, 5))
    else:
        plt.figure(fignum)
    plt.clf()
    tables = JobPerformanceTables(jira_ticket)
    tables.print_tables()
    df1 = tables.plot_processing_history(notes=notes)
    tables.print_db_job_stats()
    return tables


if __name__ == '__main__':
    tables = make_tables()
