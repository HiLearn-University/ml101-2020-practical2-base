import argparse
import numpy as np
import csv
import json
from collections import defaultdict
from pathlib import Path

from practical2.regression import target, featurize
from practical2.regression import fit_linear_regression, y_hat
from practical2.search import cross_val_splits, loss
from practical2.io import load_data, export_beta


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path,
                        default=Path('data/yerevan_may_2020.csv.gz'),
                        help='CSV file with the apartment data.')
    parser.add_argument('--output-beta', default='beta.json',
                        help='json file to save the beta in.')
    args = parser.parse_args(*argument_array)
    return args


def find_beta(X, Y) -> np.ndarray:
    # FIXME: Search for a good beta by finding a good lambda.
    return fit_linear_regression(X, Y, l=0.1)


if __name__ == '__main__':
    args = parse_args()
    raw_data = list(load_data(args.data))
    # NOTE: Feel free to modify this part as well, but make sure you call
    # `export_beta` in the end, to save the beta you found.
    X = np.array([featurize(x) for x in raw_data])
    Y = np.array([target(x) for x in raw_data])
    print(X.shape, Y.shape)
    beta = find_beta(X, Y)
    export_beta(args.output_beta, beta)
