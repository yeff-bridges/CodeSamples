import numpy as np
import pandas as pd

def bootstrap(df_x, df_y, rate=0.5, size=None):
    if size is None:
        size = df_x.shape[0]
    df_ind = np.random.choice(df_x.shape[0], int(df_x.shape[0] * rate), replace=False)
    df_sample_ind = np.random.choice(df_ind, size)
    return df_x.loc[df_sample_ind], df_y.loc[df_sample_ind]