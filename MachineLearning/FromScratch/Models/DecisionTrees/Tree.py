import pandas as pd
import numpy as np
from itertools import zip_longest

class Tree:
    def __init__(self, height, imp_func, agg_func, num=50, min_points=1, gain_thresh=0.1, max_points=None):
        self.height = height
        self.imp_func = imp_func
        self.agg_func = agg_func
        self.num = num
        self.min_points = min_points
        self.gain_thresh = gain_thresh
        self.max_points = max_points

    def fit(self, x, y):
        if self.height > 0 and x.shape[0] > self.min_points:
            if self.max_points is None:
                self.max_points = x.shape[0]
            self.leaf = False
            self.gain = -np.inf
            for feat in x.columns:
                if x[feat].dtype == 'O':
                    mask = x[feat]
                    gain = info_gain(y, mask, self.imp_func)
                    if gain > self.gain:
                        self.gain = gain
                        self.feat = feat
                        best_mask = mask
                else:
                    thresholds = np.linspace(x[feat].min(), x[feat].max(), num=self.num)
                    for i, t in enumerate(thresholds):
                        mask = x[feat] > t
                        gain = info_gain(y, mask, self.imp_func)
                        if gain > self.gain:
                            self.gain = gain
                            self.t = t
                            self.feat = feat
                            best_mask = mask

            self.gain_scaled = self.gain * x.shape[0]
            self.gain_scaled_norm = self.gain * x.shape[0] / self.max_points
            if self.gain < self.gain_thresh:
                self.leaf = True
                self.val = self.agg_func(y[y.columns[0]])
                return

            self.children = []
            self.cats = np.unique(best_mask)
            for i in self.cats:
                self.children.append(Tree(
                    height=self.height-1,
                    imp_func=self.imp_func,
                    agg_func=self.agg_func,
                    num=self.num,
                    min_points=self.min_points,
                    gain_thresh=self.gain_thresh,
                    max_points=self.max_points
                ))
                self.children[-1].fit(
                    x=x[best_mask == i],
                    y=y[best_mask == i],
                )

        else:
            self.leaf = True
            self.val = self.agg_func(y[y.columns[0]])

    def predict(self, x):
        out = np.ones((x.shape[0], 1))
        if self.leaf:
            return out * self.val
        else:
            if x[self.feat].dtype != 'O':
                mask = x[self.feat] > self.t
            else:
                mask = x[self.feat]

            for i, c in enumerate(self.cats):
                out[c == mask] = self.children[i].predict(x[c == mask])

            return out

    def get_splits(self):
        if self.leaf:
            return []
        else:
            splits = []
            child_splits = []
            for child in self.children:
                child_splits.append(child.get_splits())

            for i in zip_longest(*child_splits, fillvalue=[]):
                splits.append(sum(i, []))

            return [[{
                'feat': self.feat,
                'gain': self.gain,
                'scaled gain': self.gain_scaled,
                'norm scaled gain': self.gain_scaled_norm
            }]] + splits