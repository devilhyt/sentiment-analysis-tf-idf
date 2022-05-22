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
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    board_list = [name for name in sys.argv[1:]]

    test_combine_dicts = []
    test_order_list = []
    train_combine_dicts = []
    train_order_list = []

    for order, board in enumerate(board_list):  # 合成看板 dict_list 以及建立order_list
        with open('./data/' + board + ".pickle", 'rb') as pkl:
            tmp_list = pickle.load(pkl)
            tmp_train_data, tmp_test_data = train_test_split(
                tmp_list, random_state=1410832008, train_size=0.8)

            train_combine_dicts += tmp_train_data
            test_combine_dicts += tmp_test_data

            train_order_list += [order]*len(tmp_train_data)
            test_order_list += [order]*len(tmp_test_data)

    conv_DV = DictVectorizer()
    calc_Tfidf = TfidfTransformer()
    DictVect = conv_DV.fit_transform(train_combine_dicts)
    Tfidf_DV = calc_Tfidf.fit_transform(DictVect)

    score_DictVect = conv_DV.transform(test_combine_dicts)
    score_Tfidf_DV = calc_Tfidf.transform(score_DictVect)

    L_SVC = LinearSVC(C=0.27)
    L_SVC.fit(Tfidf_DV, train_order_list)
    L_SVC_result = L_SVC.score(X=score_Tfidf_DV, y=test_order_list)
    print('\nLinearSVC    準確度：'+str(round(L_SVC_result*100, 2))+'%')
