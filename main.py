import argparse
import csv
import os

from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression
from art.estimators.classification.pytorch import PyTorchClassifier
from art.attacks.evasion import LowProFool

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix 
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

import warnings 
warnings.filterwarnings(action='ignore')

def standardize(data):
    """
    Get both the standardized data and the used scaler.
    """
    columns = data.columns
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(data)

    return pd.DataFrame(data=x_scaled, columns=columns), scaler

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

def get_train_and_valid(design_matrix, labels):
    """
    Split dataset into training and validation sets.
    """
    for train_idx, valid_idx in split.split(design_matrix, labels):
        X_train = design_matrix.iloc[train_idx].copy()
        X_valid = design_matrix.iloc[valid_idx].copy()
        y_train = labels.iloc[train_idx].copy()
        y_valid = labels.iloc[valid_idx].copy()

    return X_train, y_train, X_valid, y_valid

def standerdize_custom_data(data):
    std_feat = ['Amount', 'Time']
    columns=data.columns
    
    std_pipeline = Pipeline([
        ('std_scaler', StandardScaler())
    ])
    
    full_pipeline = ColumnTransformer([
        ('std_feat', std_pipeline, std_feat)
        ], remainder='passthrough')
    
    x_scaled = full_pipeline.fit_transform(data)
    return pd.DataFrame(data=x_scaled, columns=columns), None

def get_custom_dataset(filename, target_column=None, column_names=None):
    custom_dataset = pd.read_csv(filename)
    custom_dataset = custom_dataset.sample(frac=1).reset_index(drop=True)
    
    if column_names:
        custom_dataset = custom_dataset[column_names]

    if target_column:
        X = custom_dataset.drop(target_column, axis=1)
        Y = custom_dataset[target_column]
    else:
        X = custom_dataset.iloc[:, :-1]
        Y = custom_dataset.iloc[:, -1]
    # print(X.describe())
    
    
    X_scaled =X
    
    X_scaled, _scaler = standardize(X)
    # print(X_scaled.describe())

    # X_scaled, _scaler = standerdize_custom_data(X)
    # print(X_scaled.describe())

    # print(Y.value_counts())
    X_train, y_train, X_valid, y_valid = \
        get_train_and_valid(X_scaled, Y)

    return X_train, y_train, X_valid, y_valid
    

def get_cancer_dataset():
    cancer = datasets.load_breast_cancer()
    design_matrix_cancer = pd.DataFrame(data=cancer['data'], columns=cancer['feature_names'])
    labels_cancer = pd.Series(data=cancer['target'])

    design_matrix_cancer_scaled, cancer_scaler = standardize(design_matrix_cancer)

    X_train_cancer, y_train_cancer, X_valid_cancer, y_valid_cancer =\
        get_train_and_valid(design_matrix_cancer_scaled, labels_cancer)

    return X_train_cancer, y_train_cancer, X_valid_cancer, y_valid_cancer


def lowprofool_generate_adversaries_test_lr(lowprofool, classifier, x_valid, y_valid):
    """
    Testing utility.
    """
    n_classes = lowprofool.n_classes

    # Generate targets
    target = np.eye(n_classes)[np.array(
        y_valid.apply(
            lambda x: np.random.choice([i for i in range(n_classes) if i != x]))
    )]

    # Generate adversaries
    adversaries = lowprofool.generate(x=x_valid, y=target)

    # Test - check the success rate
    expected = np.argmax(target, axis=1)
    predicted = np.argmax(classifier.predict_proba(adversaries), axis=1)
    correct = (expected == predicted)

    success_rate = np.sum(correct) / correct.shape[0]

    return adversaries, success_rate



def test_3(
        n_steps=100,
        eta=0.2,
        lambd=0.2,
        eta_decay=0.9,
        n_train_samples=None,
        n_val_samples=None,
        dataset=None,
        column_names=None,
        target_column=None,
        success_on_class=None,
    ):
    # print("---------- Test 3 ----------")
    if dataset:
        X_train, y_train, X_valid, y_valid = get_custom_dataset(dataset,
                                                                target_column=target_column,
                                                                column_names=column_names)
    else:
        
        X_train, y_train, X_valid, y_valid = get_cancer_dataset()

    scaled_clip_values_cancer = (-1., 1.)

    # Take only some samples
    if n_train_samples is not None:
        X_train = X_train[:n_train_samples]
        y_train = y_train[:n_train_samples]
    if n_val_samples is not None:
        X_valid = X_valid[:n_val_samples]
        y_valid = y_valid[:n_val_samples]

    # SVC = Model(scaled_clip_values_cancer)
    # SVC.train(X_train_cancer, y_train_cancer)

    log_regression_clf_cancer = LogisticRegression()
    log_regression_clf_cancer.fit(X_train.values, y_train)

    
    # y_pred = log_regression_clf_cancer.predict(X_valid_0)
    # print(f'Accuracy class: [0]: {accuracy_score(y_valid_0, y_pred):.4f}')

    # y_pred = log_regression_clf_cancer.predict(X_valid_1)
    # print(f'Accuracy class: [1]: {accuracy_score(y_valid_1, y_pred):.4f}')
    # exit()
    # print(f'F1-Score: {f1_score(y_valid, y_pred):.4f}')
    # print(f'Confusion Matrix: {confusion_matrix(y_valid, y_pred)}')
    # exit()    

    if success_on_class:
        X_valid = X_valid[y_valid == success_on_class].reset_index(drop=True)
        y_valid = y_valid[y_valid == success_on_class].reset_index(drop=True)

    # print(y_valid.value_counts())


    # Wrapping classifier into appropriate ART-friendly wrapper
    logistic_regression_cancer_wrapper = ScikitlearnLogisticRegression(
        model       = log_regression_clf_cancer,
        clip_values = scaled_clip_values_cancer
    )

    # Creating LowProFool instance
    lpf_logistic_regression_cancer = LowProFool(
        n_steps    = n_steps,
        classifier = logistic_regression_cancer_wrapper,
        eta        = eta,
        lambd      = lambd,
        eta_decay  = eta_decay
    )

    # Fitting feature importance
    lpf_logistic_regression_cancer.fit_importances(X_train, y_train)

    # Testing
    results_lr_bc, success_rate = lowprofool_generate_adversaries_test_lr(
        lowprofool = lpf_logistic_regression_cancer,
        classifier = log_regression_clf_cancer,
        x_valid    = X_valid,
        y_valid    = y_valid
    )

    return success_rate

def get_nn_model(input_dimensions, hidden_neurons, output_dimensions):
    """
    Prepare PyTorch (torch) neural network.
    """
    return torch.nn.Sequential(
        nn.Linear(input_dimensions, hidden_neurons),
        nn.ReLU(),
        nn.Linear(hidden_neurons, output_dimensions),
        nn.Softmax(dim=1)
    )

loss_fn = torch.nn.MSELoss(reduction='sum')

def train_nn(nn_model, X, y, learning_rate, epochs):
    """
    Train provided neural network.
    """
    optimizer = optim.SGD(nn_model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        y_pred = nn_model.forward(X)
        loss = loss_fn(y_pred, y)
        nn_model.zero_grad()
        loss.backward()

        optimizer.step()

def lowprofool_generate_adversaries_test_nn(lowprofool, classifier, x_valid, y_valid):
    """
    Testing utility.
    """
    n_classes = lowprofool.n_classes

    # Generate targets
    target = np.eye(n_classes)[np.array(
        y_valid.apply(
            lambda x: np.random.choice([i for i in range(n_classes) if i != x]))
    )]

    # Generate adversaries
    adversaries = lowprofool.generate(x=x_valid, y=target)

    # Test - check the success rate
    expected = np.argmax(target, axis=1)
    x = Variable(torch.from_numpy(adversaries.astype(np.float32)))

    out = classifier.forward(x).detach().numpy()
    predicted = np.argmax(out, axis=1)
    correct = (expected == predicted)

    success_rate = np.sum(correct) / correct.shape[0]
    print("Success rate: {:.2f}%".format(100*success_rate))

    return adversaries

def test_5():
    print("---------- Test 5 ----------")

    X_train_cancer, y_train_cancer, X_valid_cancer, y_valid_cancer = get_cancer_dataset()

    scaled_clip_values_cancer = (-1., 1.)

    X = Variable(torch.FloatTensor(np.array(X_train_cancer.values)))
    y = Variable(torch.FloatTensor(np.eye(2)[y_train_cancer]))
    nn_model_cancer = get_nn_model(30, 50, 2)
    train_nn(nn_model_cancer, X, y, 1e-4, 1000)

    print("le")

    # Wrapping classifier into appropriate ART-friendly wrapper
    # (in this case it is PyTorch NN classifier wrapper from ART)
    neural_network_cancer_wrapper = PyTorchClassifier(
        model       = nn_model_cancer,
        loss        = loss_fn,
        input_shape = (30,),
        nb_classes  = 2,
        clip_values = scaled_clip_values_cancer,
        device_type = "cpu"
    )

    # Creating LowProFool instance
    lpf_neural_network_cancer = LowProFool(
        classifier = neural_network_cancer_wrapper,
        n_steps    = 200,
        eta        = 10,
        lambd      = 2,
        eta_decay  = 0.99
    )

    # Fitting feature importance
    lpf_neural_network_cancer.fit_importances(X_train_cancer, y_train_cancer)

    # Testing
    results_nn_bc = lowprofool_generate_adversaries_test_nn(
        lowprofool = lpf_neural_network_cancer,
        classifier = nn_model_cancer, 
        x_valid    = X_valid_cancer, 
        y_valid    = y_valid_cancer
    )

def write_to_csv(csv_file, config, success_rate):
    config_dict = vars(config)
    headers = sorted(config_dict.keys())
    if os.path.isfile(csv_file):
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            write_headers = (sum([1 for _ in reader]) == 0)
    else:
        write_headers = True

    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        if write_headers:
            writer.writerow(headers+['success_rate'])
        writer.writerow([config_dict[header] for header in headers] + [success_rate])

def main(config):
    # There are multiple tests
        # 3. Logistic regression with cancer dataset
        # 5. NN with cancer dataset

    success_rate = test_3(
        n_steps=config.n_steps,
        eta=config.eta,
        lambd=config.lambd,
        eta_decay=config.eta_decay,
        n_train_samples=config.n_train_samples,
        n_val_samples=config.n_val_samples,
        dataset=config.dataset,
        column_names=config.column_names,
        target_column=config.target_column,
        success_on_class=config.success_on_class
    )
    write_to_csv(config.csv_file, config, success_rate)
    print("Success rate: {:.2f}%".format(100*success_rate))

    #test_5()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cf', '--csv-file', default='results.csv')
    parser.add_argument('-nts', '--n_train_samples', type=int)
    parser.add_argument('-nvs', '--n_val_samples', type=int)
    parser.add_argument('-ns', '--n_steps', default=100, type=int)
    parser.add_argument('-e', '--eta', default=0.2, type=float)
    parser.add_argument('-l', '--lambd', default=0.2, type=float)
    parser.add_argument('-ed', '--eta_decay', default=0.9, type=float)
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-cn', '--column_names', nargs='+', default=None)
    parser.add_argument('-tc', '--target_column', type=str)
    parser.add_argument('-soc', '--success_on_class', type=int)
    args = parser.parse_args()
    main(args)
