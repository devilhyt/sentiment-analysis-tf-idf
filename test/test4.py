# coding=utf-8

import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

if __name__ == "__main__":
    board_list = ["love", "sad"]
    score_combine_dicts = []
    score_order_list = []
    train_combine_dicts = []
    train_order_list = []

    sore_num = 2400
    train_num = 600
    for order, board in enumerate(board_list):  # 合成看板 dict_list 以及建立order_list
        with open('./data/' + board + ".pickle", 'rb') as pkl:
            tmp_list = pickle.load(pkl)
            score_order_list += [order]*train_num
            score_combine_dicts += tmp_list[0:train_num]
            del tmp_list[0:train_num]

            train_order_list += [order]*sore_num
            train_combine_dicts += tmp_list[0:sore_num]
            del tmp_list[0:sore_num]

    conv_DV = DictVectorizer()
    calc_Tfidf = TfidfTransformer()
    DictVect = conv_DV.fit_transform(train_combine_dicts)
    Tfidf_DV = calc_Tfidf.fit_transform(DictVect)

    score_DictVect = conv_DV.transform(score_combine_dicts)
    score_Tfidf_DV = calc_Tfidf.transform(score_DictVect)

    # from sklearn.model_selection import GridSearchCV
    # param_grid = {'C': [300]}
    # L_SVC = GridSearchCV(LinearSVC(max_iter=10000), param_grid, cv=5, return_train_score=True)
    # L_SVC.fit(Tfidf_DV, train_order_list)
    # print("L_SVC.best_estimator_")
    # print(L_SVC.best_estimator_)
    # print("L_SVC.best_score_")
    # print(L_SVC.best_score_)
    # print("L_SVC.best_params_")
    # print(L_SVC.best_params_)
    # print("L_SVC.best_index_")
    # print(L_SVC.best_index_)
    # print("L_SVC.scorer_")
    # print(L_SVC.scorer_)
    

    from skopt import BayesSearchCV
    from sklearn.datasets import load_digits
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split


    # log-uniform: understand as search over p = exp(x) by varying x
    opt = BayesSearchCV(
        SVC(),
        {
            'C': (1e-1, 1e+1, 'log-uniform'),
            'gamma': (1e-1, 1e+1, 'log-uniform'),
            'degree': (1, 8),  # integer valued parameter
            'kernel': ['rbf'],  # categorical parameter
        },
        n_iter=32,
        cv=3
    )

    opt.fit(Tfidf_DV, train_order_list)

    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(score_Tfidf_DV, score_order_list))