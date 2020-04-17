from itertools import zip_longest
import copy

import pandas as pd
import numpy as np
from scipy.stats import mode


def Bootstrap(df_x, df_y, rate=0.5, size=None):
    if size is None:
        size = df_x.shape[0]
    df_ind = np.random.choice(df_x.shape[0], int(df_x.shape[0] * rate), replace=False)
    df_sample_ind = np.random.choice(df_ind, size)
    return df_x.loc[df_sample_ind], df_y.loc[df_sample_ind]


def Gini(y):
    labels, freq = np.unique(y, return_counts=True)
    t = freq / np.sum(freq)
    return np.sum(t*(1-t))


def InfoGain(y, mask, imp_func):
    gini = imp_func(y[y.columns[0]])
    imps = []
    unique, counts = np.unique(mask, return_counts=True)
    for i in unique:
        imps.append(imp_func(y[y.columns[0]][mask == i]))
    gain = gini - (np.dot(counts, imps)) / y.shape[0]
    return gain


def Mode(z, axis=None):
    out = mode(z, axis=axis)
    return out[0]


class Tree:
    def __init__(self, height, imp_func, agg_func, num=50, min_points=1, gain_thresh=0.1, max_points=None):
        self.height = height
        self.imp_func = imp_func
        self.agg_func = agg_func
        self.num = num
        self.min_points = min_points
        self.gain_thresh = gain_thresh
        self.max_points = max_points

    def Fit(self, x, y):
        if self.height > 0 and x.shape[0] > self.min_points:
            if self.max_points is None:
                self.max_points = x.shape[0]
            self.leaf = False
            self.gain = -np.inf
            for feat in x.columns:
                if x[feat].dtype == 'O':
                    mask = x[feat]
                    gain = InfoGain(y, mask, self.imp_func)
                    if gain > self.gain:
                        self.gain = gain
                        self.feat = feat
                        best_mask = mask
                else:
                    thresholds = np.linspace(x[feat].min(), x[feat].max(), num=self.num)
                    for i, t in enumerate(thresholds):
                        mask = x[feat] > t
                        gain = InfoGain(y, mask, self.imp_func)
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
                self.children[-1].Fit(
                    x=x[best_mask == i],
                    y=y[best_mask == i],
                )

        else:
            self.leaf = True
            self.val = self.agg_func(y[y.columns[0]])

    def Predict(self, x):
        out = np.ones((x.shape[0], 1))
        if self.leaf:
            return out * self.val
        else:
            if x[self.feat].dtype != 'O':
                mask = x[self.feat] > self.t
            else:
                mask = x[self.feat]

            for i, c in enumerate(self.cats):
                out[c == mask] = self.children[i].Predict(x[c == mask])

            return out

    def GetSplits(self):
        if self.leaf:
            return []
        else:
            splits = []
            child_splits = []
            for child in self.children:
                child_splits.append(child.GetSplits())

            for i in zip_longest(*child_splits, fillvalue=[]):
                splits.append(sum(i, []))

            return [[{
                'feat': self.feat,
                'gain': self.gain,
                'scaled gain': self.gain_scaled,
                'norm scaled gain': self.gain_scaled_norm
            }]] + splits


class BaggedForest:
    def __init__(self, root_tree, num_trees=50, rate=0.5, size=None):
        self.root_tree = root_tree
        self.num_trees = num_trees
        self.rate = rate
        self.size = size

    def Fit(self, x, y):
        self.trees = []
        for i in range(self.num_trees):
            x_sample, y_sample = Bootstrap(x, y, self.rate, self.size)
            self.trees.append(copy.copy(self.root_tree))
            self.trees[-1].Fit(x_sample, y_sample)

    def Predict(self, x):
        preds = []
        for t in self.trees:
            preds.append(t.Predict(x))

        preds = np.hstack(preds)
        return self.root_tree.agg_func(preds, axis=1)


class Random_Forest:
    def __init__(self, root_tree, num_trees=50, bagg_rate=0.5, feat_prop=None, size=None):
        self.root_tree = root_tree
        self.num_trees = num_trees
        self.bagg_rate = bagg_rate
        self.size = size
        self.feat_prop = feat_prop

    def Fit(self, x, y):
        if self.feat_prop is None:
            feats = int(np.round(np.sqrt(x.shape[1]) + 6e-17))
        else:
            feats = int(np.round(self.feat_prop * x.shape[1] + 6e-17))
        self.trees = []

        for i in range(self.num_trees):
            x_sample, y_sample = Bootstrap(x, y, self.bagg_rate, self.size)
            x_sample = x_sample[np.random.choice(x.columns, size=feats, replace=False)]
            self.trees.append(copy.copy(self.root_tree))
            self.trees[-1].Fit(x_sample, y_sample)

    def Predict(self, x):
        preds = []
        for t in self.trees:
            preds.append(t.Predict(x))

        preds = np.hstack(preds)
        return self.root_tree.agg_func(preds, axis=1)


class Boosting:
    def __init__(self, model, num=20):
        self.num = num
        self.models = []
        for i in range(self.num):
            self.models.append(copy.copy(model))

    def Fit(self, x, y):
        resid = y
        y_hat = 0
        for model in self.models:
            model.Fit(x, resid - y_hat)
            y_hat = model.Predict(x)
            resid = resid - y_hat
        return resid

    def Predict(self, x):
        y_hat = 0
        for model in self.models:
            y_hat += model.Predict(x)
        return y_hat


