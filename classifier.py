# coding=utf-8
import sys

if __name__ == "__main__" and len(sys.argv) <= 2:
    print("錯誤！請從命令行輸入至少 2 個 PTT 看板參數")
    sys.exit()

#===================================================

import numpy as np
import pickle
import ptt_spider
import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

def article_classifier(L_SVC, conv_DV, calc_Tfidf, url_str, mode):
    if mode == 0:
        d = preprocessing.segmentation(
            [ptt_spider.get_articles_content(url_str)])
    elif mode == 1:
        d = preprocessing.segmentation([url_str])
    article_DictVect = conv_DV.transform(d)
    article_Tfidf_DV = calc_Tfidf.transform(article_DictVect)
    result = L_SVC.predict(article_Tfidf_DV)
    print('文章屬於' + board_list[result[0]] + '\n')


if __name__ == "__main__":
    board_list = [name for name in sys.argv[1:]]

    # ======合成看板 dict_list、建立order_list======
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

    # ======轉為Vector、Tfidf======
    conv_DV = DictVectorizer()
    calc_Tfidf = TfidfTransformer()
    DictVect = conv_DV.fit_transform(train_combine_dicts)
    Tfidf_DV = calc_Tfidf.fit_transform(DictVect)

    test_DictVect = conv_DV.transform(test_combine_dicts)
    test_Tfidf_DV = calc_Tfidf.transform(test_DictVect)

    # ======訓練模型======
    L_SVC = LinearSVC(C=0.65)
    L_SVC.fit(Tfidf_DV, train_order_list)

    # ======預測======
    L_SVC_result = L_SVC.score(X=test_Tfidf_DV, y=test_order_list)
    print('\nLinearSVC    準確度：'+str(round(L_SVC_result*100, 2))+'%')

    # ======輸入功能======
    mode = 0
    dp_str = ['PTT文章網址', '文字']
    while True:
        url = input('情緒分析，請輸入' + dp_str[mode] + '：\n')
        print()
        if url == '/exit':
            sys.exit()
        elif url == '/str':
            mode = 1
        elif url == '/url':
            mode = 0
        else:
            article_classifier(L_SVC, conv_DV, calc_Tfidf, url, mode)
