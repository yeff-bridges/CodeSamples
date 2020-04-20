import numpy as np

class Random_Forest:
    def __init__(self, root_tree, num_trees=50, bagg_rate=0.5, feat_prop=None, size=None):
        self.root_tree = root_tree
        self.num_trees = num_trees
        self.bagg_rate = bagg_rate
        self.size = size
        self.feat_prop = feat_prop

    def fit(self, x, y):
        if self.feat_prop is None:
            feats = int(np.round(np.sqrt(x.shape[1]) + 6e-17))
        else:
            feats = int(np.round(self.feat_prop * x.shape[1] + 6e-17))
        self.trees = []

        for i in range(self.num_trees):
            x_sample, y_sample = bootstrap(x, y, self.bagg_rate, self.size)
            x_sample = x_sample[np.random.choice(x.columns, size=feats, replace=False)]
            self.trees.append(copy.copy(self.root_tree))
            self.trees[-1].fit(x_sample, y_sample)

    def predict(self, x):
        preds = []
        for t in self.trees:
            preds.append(t.predict(x))

        preds = np.hstack(preds)
        return self.root_tree.agg_func(preds, axis=1)