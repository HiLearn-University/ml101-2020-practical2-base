import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


class RandomForestClassifier(object):
    def __init__(self, nb_trees=100, nb_samples=1000, max_depth=-1, max_workers=1):
        """
        :param  nb_trees:       Number of decision trees to use
        :param  nb_samples:     Number of samples to give to each tree
        :param  max_depth:      Maximum depth of the trees
        If needed you can add your own parametrs
        """
        self.trees = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth

    def fit(self, data):
        """
        Trains self.nb_trees number of decision trees.
        :param  data:   A list of lists with the last element of each list being
                        the value to predict: "label"
        """
        raise NotImplementedError()

    def train_tree(self, data):
        """
        Trains a single tree and returns it.
        Use your own DecisionTreeClassifier model.
        :param  data:  The data to train it
        """
        raise NotImplementedError()

    def predict(self, data):
        """
        Returns a prediction for the given feature. The result is the value that
        gets the most votes.
        :param  feature:    The features used to predict
        """
        raise NotImplementedError()


def main():
    clas = RandomForestClassifier()

    # Read data
    train_data = pd.read_csv("data/train.csv")

    # Separate labels from data
    labels = train_data['label'].values
    x = np.array(train_data.drop('label', axis=1))
    y = labels
    x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                      test_size=0.2,
                                                      random_state=42)
    # train data
    clas.fit([x_train, y_train])
    y_predict = clas.predict(x_val)

    # calculate score
    print(f1_score(y, y_predict))


if __name__ == "__main__":
    main()
