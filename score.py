import argparse
from pathlib import Path
import json
import numpy as np
from save_beta import load_data
from practical2.regression import target, featurize, y_hat
from practical2.search import loss
from practical2.io import load_beta


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/first10.csv', type=Path,
                        help='CSV file with the apartment data.')
    parser.add_argument('--input-beta', default='beta.json',
                        help='json file to save the beta in.')
    args = parser.parse_args(*argument_array)
    return args



if __name__ == '__main__':
    # NOTE: This is **not** the code we will use for grading.
    #       This code is provided for your convenience.
    args = parse_args()
    raw_data = list(load_data(args.data))
    X = np.array([featurize(x) for x in raw_data])
    Y = np.array([target(x) for x in raw_data])

    beta = load_beta(args.input_beta)
    print(f'Mean Error = {np.sqrt(loss(Y, y_hat(X, beta))):,.3f}')
