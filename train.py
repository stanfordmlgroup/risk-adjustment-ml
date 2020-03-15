"""Train model on a given dataset."""
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import r2_score, mean_absolute_error
import argparse
import collections
from pathlib import Path

from preprocess import preprocess
from saver import ModelSaver
from predictor import predict
import models


def get_model(model_args):
    """Get model object to train."""
    model_fn = models.__dict__[model_args.model]
    
    model = model_fn(tuning_metric=model_args.tuning_metric, 
                     type=model_args.regularizer,
                     bottom_coding=model_args.bottom_coding,
                     gpu=model_args.gpu)
    return model


def train(args):
    """Train the model on the dataset."""
    Dataset = collections.namedtuple('Dataset', 'X y')

    assert args.random or args.csv_path, "Please choose either random data or pass a data path"

    if args.random:
        train_features = sp.random(100, 100)
        train_costs = np.random.random((100,))

        test_features = np.random.random((100, 100))
        test_costs = np.random.random((100,))
    else:
        train_path = Path(args.train_path)
        train_df = pd.read_csv(train_path)
        train_features, train_costs = preprocess(train_df, args.sdh)

        test_path = Path(args.test_path)
        test_df = pd.read_csv(test_path)
        test_features, test_costs = preprocess(test_df, args.sdh)

    train_dataset = Dataset(train_features, train_costs)
    test_dataset = Dataset(test_features, test_costs)

    # Load the model.
    saver = ModelSaver(args.save_dir)
    model = get_model(args)

    # Instantiate the model saver.
    # Train model on dataset, cross-validating on validation set.
    model.fit(train_dataset)
    saver.save(model)

    preds, targets = predict(model, test_dataset)

    metrics = {"R2-score": r2_score(targets, preds),
               "MAE": mean_absolute_error(targets, preds)}

    # Print metrics to stdout.
    print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML for Risk Adjustment Training')
    parser.add_argument('--model',
                        required=True,
                        type=str,
                        help='Type of model to use',
                        choices=['LightGBM', 'LinReg'])
    parser.add_argument('--regularizer',
                        type=str,
                        default='regular',
                        help='Type of regularization to use with lin reg (default none).',
                        choices=['regular', 'l1', 'l2'])
    parser.add_argument('--tuning_metric',
                        type=str,
                        default='mse',
                        help='Which metric to optimize in param search.',
                        choices=['mse', 'mae', 'huber'])
    parser.add_argument('--train_path',
                        type=str,
                        help='Path to csv with patient information for training.',)
    parser.add_argument('--test_path',
                        type=str,
                        help='Path to csv with patient information for testing.',)
    parser.add_argument('--save_dir',
                        default=Path("model_chkpts", ),
                        type=str,
                        help='Directory to output model weight checkpoint.')
    parser.add_argument('--random',
                        action='store_true',
                        help='Use random data.')
    parser.add_argument('--bottom_coding',
                        type=str,
                        default=0,
                        help='Amount by which to bottom code costs.')
    parser.add_argument('--gpu',
                        action='store_true',
                        help='Use GPU (LightGBM only)')
    train(parser.parse_args())
