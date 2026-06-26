from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ["plot_time_history"]


def plot_time_history(df0, label=None, linestyle=None, color=None, alpha=0.5,
                      weight_column=None, make_plot=True, yfactor=1.0):
    start_time = df0['JobStartDate'].to_numpy()
    end_time = df0['CompletionDate'].to_numpy()
    start_dt = df0['start_dt'].to_numpy()
    end_dt = df0['end_dt'].to_numpy()
    offset_time = np.concat((start_time, end_time)) - 0.1
    offset_dt = np.concat((start_dt, end_dt)) - np.timedelta64(timedelta(seconds=0.1))
    zeros = np.zeros(len(offset_time))
    if weight_column is None:
        delta = np.ones(len(df0))*yfactor
    else:
        delta = df0[weight_column].to_numpy()*yfactor
    df = pd.DataFrame(dict(times=np.concat((start_time, end_time, offset_time)),
                           deltas=np.concat((delta, -delta, zeros)),
                           datetime=np.concat((start_dt, end_dt, offset_dt))))
    df = df.sort_values('times')
    df['num_jobs'] = np.cumsum(df['deltas'])
    artist = None
    if make_plot:
        artist = plt.plot(df['datetime'], df['num_jobs'], label=label,
                          linestyle=linestyle, color=color, alpha=alpha)[0]
    return np.trapezoid(df["num_jobs"], df["times"])/3600., artist
