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
    combine_dicts = []
    order_list = []
    for order, board in enumerate(board_list):
        with open('./data/' + board + ".pickle", 'rb') as pkl:
            tmp_list = pickle.load(pkl)
            combine_dicts += tmp_list
            order_list += [order]*len(tmp_list)

    # ======轉為Vector、Tfidf======
    conv_DV = DictVectorizer()
    calc_Tfidf = TfidfTransformer()
    DictVect = conv_DV.fit_transform(combine_dicts)
    Tfidf_DV = calc_Tfidf.fit_transform(DictVect)

    # ======訓練模型======
    L_SVC = LinearSVC(C=0.27)
    L_SVC.fit(Tfidf_DV, order_list)

    mode = 0
    dp_str = ['PTT文章網址', '文字']
    while True:
        url = input('情緒分析，請輸入' + dp_str[mode] + '：\n')
        print()
        if url == '-exit':
            sys.exit()
        elif url == '-str':
            mode = 1
        elif url == '-url':
            mode = 0
        else:
            article_classifier(L_SVC, conv_DV, calc_Tfidf, url, mode)
