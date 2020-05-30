from practical2.io import load_beta, load_data
from practical2.random_forest import RandomForestClassifier
from pathlib import Path
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np


def test_random_forest():
    clas = RandomForestClassifier().fit()

    # Read data
    train_data = pd.read_csv("data/train.csv")
    train_data = train_data[:300]

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

    assert f1_score(y, y_predict) > 0.5
