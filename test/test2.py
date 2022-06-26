# coding=utf-8

import numpy as np
import pickle
import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC



if __name__ == "__main__":
    board_list = ["happy", "hate", "sad"]

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
    L_SVC = LinearSVC()
    L_SVC.fit(Tfidf_DV, order_list)



    #-----------------------------------------------

    # url_str="鄰居養了臘腸狗，腿短短的，好可愛，每次經過他們家門口，我就會蹲下來招招手，10次裡有8次牠不會理我，只有2次會開心的跑來讓我摸摸，我感覺像抽獎一樣，牠讓我摸摸的時候我就覺得很開心~~~，每天下班的小確幸"

    url_str = "剛剛去買點心的時候，老闆額外再多送一份，真開心，今天工作上的不愉快都無所謂啦！"
    # url_str = "世界會亂，有部分應歸咎於無聊人士的小心眼，心胸狹窄得只住得進自己，何其可悲。"
    # url_str ="心情糟晚睡 頭很痛又更不舒服，雖然早就有心理準備，聽到當下還是會難過，本來就察覺到可能在保持距離，那就繼續保持距離吧"

    print(f"原文：")
    print(url_str)
    print("")

    #-----------------------------------------------

    d = preprocessing.segmentation([url_str])
    print(d)
    article_DictVect = conv_DV.transform(d)
    article_Tfidf_DV = calc_Tfidf.transform(article_DictVect)
    result = L_SVC.predict(article_Tfidf_DV)
    print('文章屬於' + board_list[result[0]] + '\n')

    #-----------------------------------------------

    print('TF IDF 結果：')
    top_n = 25
    tfidf_sorted_list = sorted(list(zip(conv_DV.get_feature_names(), article_Tfidf_DV.sum(
        0).getA1())), key=lambda x: x[1], reverse=True)[:top_n]
    for i in tfidf_sorted_list:
        print(i)

    #-----------------------------------------------

    feature_sorted_list = sorted(list(zip(conv_DV.get_feature_names(),
                                   article_DictVect.sum(0).getA1())),
                          key=lambda x: x[1], reverse=True)[:top_n]

    global_feature_sorted_list = sorted(list(zip(conv_DV.get_feature_names(),
                                   DictVect.sum(0).getA1())),
                          key=lambda x: x[1], reverse=True)
    
    print('單詞出現的次數：')
    for i in feature_sorted_list:
        print(i)

    print('單詞在文章庫中出現的次數')
    for name_g in global_feature_sorted_list:
        for name in feature_sorted_list:
            if name[0] == name_g[0]:
                print(name_g)
