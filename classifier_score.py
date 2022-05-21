# coding=utf-8
import sys

if __name__ == "__main__" and len(sys.argv) <= 2:
    print("錯誤！請從命令行輸入至少 2 個 PTT 看板參數")
    sys.exit()

#===================================================

import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC

if __name__ == "__main__":
    board_list = [name for name in sys.argv[1:]]
    score_combine_dicts = []
    score_order_list = []
    train_combine_dicts = []
    train_order_list = []

    train_num = int(input('請輸入訓練篇數: '))
    score_num = int(input('請輸入測試篇數: '))
    for order, board in enumerate(board_list):  # 合成看板 dict_list 以及建立order_list
        with open('./data/' + board + ".pickle", 'rb') as pkl:
            tmp_list = pickle.load(pkl)
            score_order_list += [order]*score_num
            score_combine_dicts += tmp_list[0:score_num]
            del tmp_list[0:score_num]

            train_order_list += [order]*train_num
            train_combine_dicts += tmp_list[0:train_num]
            del tmp_list[0:train_num]

    conv_DV = DictVectorizer()
    calc_Tfidf = TfidfTransformer()
    DictVect = conv_DV.fit_transform(train_combine_dicts)
    Tfidf_DV = calc_Tfidf.fit_transform(DictVect)

    score_DictVect = conv_DV.transform(score_combine_dicts)
    score_Tfidf_DV = calc_Tfidf.transform(score_DictVect)

    L_SVC = LinearSVC(C=0.27)
    L_SVC.fit(Tfidf_DV, train_order_list)
    L_SVC_result = L_SVC.score(X=score_Tfidf_DV, y=score_order_list)
    print('\nLinearSVC    準確度：'+str(round(L_SVC_result*100, 2))+'%')
