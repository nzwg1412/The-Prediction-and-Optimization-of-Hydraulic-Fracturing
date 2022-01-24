"""
Date: 2021/05/11
Author: worith

"""

import numpy as np
import pandas as pd
import os
import argparse
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from config.config import global_config
from dataset.fracture_dataset import PDDataset

parser = argparse.ArgumentParser(description='agent model of frature')

parser.add_argument('--trainer_name', default=global_config.getRaw('config', 'model_name'))
parser.add_argument('--gpu', type=int, default=[0], nargs='+', help='used gpu')
parser.add_argument('--six_stages', action='store_true', help='2 stages data or 6 stages')

args = parser.parse_args()

# global config
data_path = global_config.getRaw('config', 'data_base_path')
random_seed = int(global_config.getRaw('train', 'random_seed'))


def my_DecisionTree_Regressor(x_train, y_train):
    # x_train, y_train = data
    dt_regressor = DecisionTreeRegressor()
    dt_regressor.fit(x_train, y_train)

    return dt_regressor


def my_Adaboost_Regressor(x_train, y_train):
    ada_regressor = ensemble.AdaBoostRegressor(n_estimators=50)
    ada_regressor.fit(x_train, y_train)
    return ada_regressor


def my_RandomForest_Regressor(x_train, y_train):
    rf_regressor = ensemble.RandomForestRegressor()
    rf_regressor.fit(x_train, y_train)
    return rf_regressor


def my_KNN_Regressor(x_train, y_train):
    knn_regressor = KNeighborsRegressor()
    knn_regressor.fit(x_train, y_train)
    return knn_regressor


def my_SVM_Regressor(x_train, y_train):
    svm_regressor = SVR()
    svm_regressor.fit(x_train, y_train)
    return svm_regressor


def main():
    # load data
    if args.six_stages:
        file_path = os.path.join(data_path, '6_stages.csv')

    else:
        file_path = os.path.join(data_path, '2_stages.csv')


    data = pd.read_csv(file_path)
    data.dropna(axis=0, how='any', inplace=True)

    data = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))


    train_data, test_data = train_test_split(data, test_size=0.1, random_state=random_seed)
    if args.six_stages:
        x_train, y_train = train_data.iloc[:, 0:13], train_data.iloc[:, 13]
        x_test, y_test = test_data.iloc[:, 0:13], test_data.iloc[:, 13]
    else:
        x_train, y_train = train_data.iloc[:, 0:5], train_data.iloc[:, 5]
        x_test, y_test = test_data.iloc[:, 0:5], test_data.iloc[:, 5]


    # decision tree
    dt_reg = my_DecisionTree_Regressor(x_train, y_train)
    dt_y_preds = dt_reg.predict(x_test)
    dt_error = np.linalg.norm(y_test - dt_y_preds, 2) / np.linalg.norm(y_test, 2)
    print('l2 error of decision tree: %f' % dt_error)

    # adaboost
    ada_reg = my_Adaboost_Regressor(x_train, y_train)
    ada_y_preds = ada_reg.predict(x_test)
    ada_error = np.linalg.norm(y_test - ada_y_preds, 2) / np.linalg.norm(y_test, 2)
    print('l2 error of adaboost: %f' % ada_error)
    
    # random forest
    rf_reg = my_RandomForest_Regressor(x_train, y_train)
    rf_y_preds = rf_reg.predict(x_test)
    rf_error = np.linalg.norm(y_test - rf_y_preds, 2) / np.linalg.norm(y_test, 2)
    print('l2 error of random forest: %f' % rf_error)

    # knn
    knn_reg = my_KNN_Regressor(x_train, y_train)
    knn_y_preds = knn_reg.predict(x_test)
    knn_error = np.linalg.norm(y_test - knn_y_preds, 2) / np.linalg.norm(y_test, 2)
    print('l2 error of knn: %f' % knn_error)

    # svm
    svm_reg = my_SVM_Regressor(x_train, y_train)
    svm_y_preds = svm_reg.predict(x_test)
    svm_error = np.linalg.norm(y_test - svm_y_preds, 2) / np.linalg.norm(y_test, 2)
    print('l2 error of svm: %f' % svm_error)


if __name__ == '__main__':
    main()