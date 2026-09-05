import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from astropy.time import Time
import numpy as np
import pandas as pd
import lsst.daf.butler as daf_butler
from lsst.pipe.base import Pipeline
from .opensearch_tools import get_os_job_info, get_job_batch_ids


__all__ = ["get_workflows", "job_performance", "PipelineStageClassifier"]


class PipelineStageClassifier:
    def __init__(self, pipeline_yaml=None, repo="dp2_prep"):
        if pipeline_yaml is None:
            pipeline_yaml = os.path.join(os.environ["DRP_PIPE_DIR"],
                                         "pipelines", "LSSTCam",
                                         "DRP.yaml")
        pipeline = Pipeline.from_uri(pipeline_yaml)
        butler = daf_butler.Butler(repo)
        pg = pipeline.to_graph(registry=butler.registry)
        stages = sorted(_ for _ in pg.task_subsets.keys()
                        if _.startswith("stage"))
        self.task_subsets = defaultdict(set)
        for stage in stages:
            stage_name = stage[:len("stage1")]
            self.task_subsets[stage_name].update(set(pg.task_subsets[stage]))
        # Include common clusters
        self.task_subsets['stage1'].update({"step1detector", "step1b_visits"})
        self.task_subsets['stage2'].update({"step2d_refitpsf", "step2d_visits"})
        self.task_subsets['stage3'].update({"makeWarpTract", "coadd"})
        self.task_subsets['stage4'].update({"diffim", "step4c_forced_phot",
                                            "step4c_forced_phot_dia"})

    def classify(self, task_list):
        overlaps = []
        for stage, subset in self.task_subsets.items():
            overlaps.append((len(subset.intersection(set(task_list))), stage))
        return sorted(overlaps, key=lambda x: x[0])[-1][-1]


def get_workflows(batch_name_substr, hours_back=None, start_date=None,
                  end_date=None, bps_job_label=None):
    # Find all runs with the desired ticket number in the JobBatchName.
    if hours_back is None and start_date is None:
        print("Considering last 14*24 hours:")
        hours_back = 14*24
    if end_date is None:
        end_date = datetime.now(timezone.utc).isoformat()[:-len("+00:00")]
    if start_date is None:
        dt = timedelta(hours=hours_back)
        start_date = Time(end_date, format="isot").datetime - dt
        start_date = start_date.isoformat()
    return get_job_batch_ids(batch_name_substr, start_date, end_date,
                             bps_job_label=bps_job_label)


def job_performance(job_batch_ids):
    if isinstance(job_batch_ids, pd.DataFrame):
        df0 = job_batch_ids
    elif isinstance(job_batch_ids, str):
        df0 = get_os_job_info(job_batch_ids)
    else:
        df0 = pd.concat([get_os_job_info(_) for _ in job_batch_ids])
    job_types = set(df0['bps_job_label'])
    data = defaultdict(list)
    for job_type in job_types:
        if job_type in ("buildQuantumGraph", "preparePayloadWorkflow",
                        "pipetaskInit", "finalJob"):
            continue
        df = df0.query(f"bps_job_label=='{job_type}' and cpu_time > 0")
        data['job_type'].append(job_type)
        total_wall_time = sum(df['wall_time']*df['RequestCpus'])/3600.
        total_cpu_time = sum(df['cpu_time'])/3600.
        data['total_wall_time (h)'].append(total_wall_time)
        data['total_cpu_time (h)'].append(total_cpu_time)
        data['wall - cpu time'].append(total_wall_time - total_cpu_time)
        data['mean wall/cpu'].append(
            np.mean(df['wall_time']*df['RequestCpus']/df['cpu_time']))
        data['num_jobs'].append(len(df))
        data['mean wait time (min)'].append(
            np.mean((df['JobStartDate'] - df['QDate'])/60.)
        )
        data['mean memory_request'].append(np.mean(df['memory_request']))

        df1 = pd.DataFrame(data).sort_values(
            ['wall - cpu time', 'mean wall/cpu', 'num_jobs'],
            ascending=False, ignore_index=True)

    return df1
