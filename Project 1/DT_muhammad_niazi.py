import pandas as pd
import numpy as np
import random
# https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb

from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeModel:

    def __init__(self, max_depth=100, criterion='gini', min_samples_split=2, impurity_stopping_threshold=0):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # TODO
        # call the _fit method
        X_array = X.to_numpy()
        Y_array = y.to_numpy()
        self._fit(X_array, Y_array)
        # end TODO
        print("Done fitting")

    def predict(self, X: pd.DataFrame):
        # TODO
        # call the predict method
        X_array = X.to_numpy()
        return self._predict(X_array)
        # end TODO

    def _fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

    def _is_finished(self, y, depth):
        # TODO: for graduate students only, add another stopping criteria
        # modify the signature of the method if needed
        if (depth >= self.max_depth
                or self.n_class_labels == 1
                or self.n_samples < self.min_samples_split
                or self._is_homogenous_enough(y)):
            return True
        # end TODO
        return False

    def _is_homogenous_enough(self, y):
        # TODO: for graduate students only
        loss = (self._gini(y) if self.criterion == 'gini' else self._entropy(y))
        result = loss < self.impurity_stopping_threshold
        # end TODO
        return result

    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self._is_finished(y, depth):
            u, counts = np.unique(y, return_counts=True)
            most_common_Label = u[np.argmax(counts)]
            return Node(value=most_common_Label)

        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)

    def _gini(self, y):
        # TODO
        u, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        gini = 1 - np.sum(np.square(proportions))
        # end TODO
        return gini

    def _entropy(self, y):
        # TODO: the following won't work if y is not integer
        u, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        # end TODO
        return entropy

    def _create_split(self, X, thresh):
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, thresh):
        # TODO: fix the code so it can switch between the two criterion: gini and entropy
        parent_loss = (self._gini(y) if self.criterion == 'gini' else self._entropy(y))
        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0:
            return 0

        if self.criterion == 'gini':
            child_loss = (n_left / n) * self._gini(y[left_idx]) + (n_right / n) * self._gini(y[right_idx])
        else:
            child_loss = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
        # end TODO
        return parent_loss - child_loss

    # TODO: add comments here
    def _best_split(self, X, y, features):
        '''TODO: add comments here
        for each feature, calculate the _information_gain for each unique threshold.
        find the best split feature and best threshold which can produce largest _information_gain.
        '''
        split = {'score': - 1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']

    # TODO: add comments here
    def _traverse_tree(self, x, node):
        '''TODO: add some comments here
        traverse the tree recursively with sample data point x,
        if x's node.feature value <= node's threshold, traverse the left child tree
        else traverse the left child tree
        Until reach to a leaf node, return the node value as sample data point's label
        '''
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestModel(object):

    def __init__(self, n_estimators):
        # TODO:
        models = []
        for _ in range(n_estimators):
            models.append(DecisionTreeModel(max_depth=10))
        self.models = models
        # end TODO

    def _fit_one_model(self, model, X: pd.DataFrame, y: pd.Series):
        random_rows = random.choices(range(X.shape[0]), k=X.shape[0])
        X_train = X.iloc[random_rows]
        y_train = y.iloc[random_rows]
        model.fit(X_train, y_train)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # TODO:
        for model in self.models:
            random_rows = random.choices(range(X.shape[0]), k=X.shape[0])
            X_train = X.iloc[random_rows]
            y_train = y.iloc[random_rows]
            model.fit(X_train, y_train)
        # end TODO

    def predict(self, X: pd.DataFrame):
        # TODO:
        predicts = pd.DataFrame()
        count = 0
        for model in self.models:
            predicts[count] = model.predict(X)
            count += 1

        predict = predicts.mode(axis=1).iloc[:, 0]
        return predict.to_numpy()
        # end TODO


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def _conf_matrix(y_test, y_pred):
    return pd.crosstab(y_test, y_pred)


def classification_report(y_test, y_pred):
    # calculate precision, recall, f1-score
    # TODO:
    df_confusion = _conf_matrix(y_test, y_pred)
    labels = df_confusion.columns
    result = {}
    for idx, label in enumerate(labels):
        alt_idx = 1 if idx == 0 else 0
        tp = df_confusion[labels[alt_idx]][labels[alt_idx]]
        tn = df_confusion[labels[idx]][labels[idx]]
        fn = df_confusion[labels[idx]][labels[alt_idx]]
        fp = df_confusion[labels[alt_idx]][labels[idx]]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        support = tn + fp
        result[label] = {'precision': precision,
                         'recall': recall,
                         'f1-score': f1_score,
                         'support': support
                         }
    # end TODO
    return result


def confusion_matrix(y_test, y_pred):
    # return the 2x2 matrix
    # TODO:
    result = _conf_matrix(y_test, y_pred).to_numpy()
    # end TODO
    return result


def _test():
    df = pd.read_csv('breast_cancer.csv')

    # X = df.drop(['diagnosis'], axis=1).to_numpy()
    # y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1).to_numpy()

    X = df.drop(['diagnosis'], axis=1)
    # y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    clf = DecisionTreeModel(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", acc)
    print("Confusion Matrix: \n", matrix)
    print("Classification report: \n", report)

def _test():
    y_test = pd.Series(['a', 'b', 'a', 'b', 'b'])
    y_pred = pd.Series(['a', 'a', 'a', 'b', 'b'])

    from sklearn.metrics import confusion_matrix as cm
    from sklearn.metrics import classification_report as cr

    print("sklearn confusion matrix")
    print(cm(y_test, y_pred))    
    print("Testing confusion matrix")
    print(confusion_matrix(y_test, y_pred))

    print(cr(y_test, y_pred))    
    print(classification_report(y_test, y_pred))

    
    df = pd.read_csv('breast_cancer.csv')

    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1)
    y = df['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    clf = DecisionTreeModel(max_depth=10, criterion='gini')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Gini Accracy: " + str(acc))

    print(classification_report(y_test,y_pred))

    clf = DecisionTreeModel(max_depth=10, criterion='entropy')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Entropy Accracy: " + str(acc))

    print(classification_report(y_test,y_pred))

    rfc = RandomForestModel(n_estimators=3)
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    print("RF Model")
    print(classification_report(y_test, rfc_pred))
    print(accuracy_score(y_test, rfc_pred))


if __name__ == "__main__":
    _test()
