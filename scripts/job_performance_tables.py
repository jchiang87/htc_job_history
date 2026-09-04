#!/usr/bin/env python
import sys
import click
import matplotlib.pyplot as plt
import pandas as pd
from htc_job_history import (get_workflows, job_performance, plot_time_history,
                             get_os_job_info, PipelineStageClassifier)


STAGE_CLASSIFIER = PipelineStageClassifier()


def job_performance_tables(jira_ticket, hours_back=21*24, min_doc_count=5,
                           roll_up=True):
    df0 = get_workflows(jira_ticket, hours_back=hours_back)
    df0 = df0.query(f"doc_count >= {min_doc_count}")
    df0 = df0.sort_values("JobStartDate", ignore_index=True)
    if roll_up:
        df1 = job_performance(df0['JobBatchId'].to_list())
        print(jira_ticket)
        print(df1[df1.columns[:-1]].head(20).to_string())
        return df1

    dfs = []
    for job_batch_id, job_batch_name in zip(df0['JobBatchId'],
                                            df0['JobBatchName']):
        try:
            df1 = job_performance(job_batch_id)
        except KeyError:
            continue
        print(job_batch_id, job_batch_name)
        print(df1[df1.columns[:-1]].to_string())
        print()
        df1['JobBatchId'] = job_batch_id
        dfs.append(df1)
    return pd.concat(dfs)


def plot_processing_history(jira_ticket, notes="",
                            hours_back=21*24, min_doc_count=5,
                            timezone="US/Pacific"):
    df0 = get_workflows(jira_ticket, hours_back=hours_back)
    df0 = df0.query(f"doc_count >= {min_doc_count}")
    df0 = df0.sort_values("JobStartDate", ignore_index=True)
    dfs = []
    for job_batch_id in df0['JobBatchId']:
        df = get_os_job_info(job_batch_id)
        df['stage'] = STAGE_CLASSIFIER.classify(set(df['bps_job_label']))
        dfs.append(df)
    df1 = pd.concat(dfs)
    stages = sorted(set(df1['stage']))
    for stage in stages:
        df = df1.query(f"stage=='{stage}'")
        plot_time_history(df, weight_column="RequestCpus", timezone=timezone,
                          label=stage)
    plt.legend(fontsize='x-small')
    plt.xlabel(f"Time ({timezone})")
    plt.ylabel("concurrent processes")
    plt.title(f"{jira_ticket}: {notes}")
    return df1


@click.command()
@click.argument("jira_ticket")
@click.option("--notes", default="", help="Ticket notes")
@click.option("--fignum", default=None, help="Figure number")
def make_tables(jira_ticket, notes, fignum):
    plt.ion()
    if fignum is not None:
        fignum = int(fignum)
        if not plt.fignum_exists(fignum):
            plt.figure(fignum, figsize=(12, 5))
        else:
            plt.figure(fignum)
    plt.clf()
    df0 = job_performance_tables(jira_ticket)
    df1 = plot_processing_history(jira_ticket, notes=notes)

if __name__ == '__main__':
    make_tables()
