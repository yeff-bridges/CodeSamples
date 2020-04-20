import copy

class BaggedForest:
    def __init__(self, root_tree, num_trees=50, rate=0.5, size=None):
        self.root_tree = root_tree
        self.num_trees = num_trees
        self.rate = rate
        self.size = size

    def fit(self, x, y):
        self.trees = []
        for i in range(self.num_trees):
            x_sample, y_sample = bootstrap(x, y, self.rate, self.size)
            self.trees.append(copy.copy(self.root_tree))
            self.trees[-1].fit(x_sample, y_sample)

    def predict(self, x):
        preds = []
        for t in self.trees:
            preds.append(t.predict(x))

        preds = np.hstack(preds)
        return self.root_tree.agg_func(preds, axis=1)