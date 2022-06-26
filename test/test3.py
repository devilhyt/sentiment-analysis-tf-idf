# coding=utf-8

import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    board_list = ["happy", "hate", "sad"]
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

    test_DictVect = conv_DV.transform(test_combine_dicts)
    test_Tfidf_DV = calc_Tfidf.transform(test_DictVect)

    L_SVC = LinearSVC(C=0.65)
    L_SVC.fit(Tfidf_DV, train_order_list)
    L_SVC_result = L_SVC.score(X=test_Tfidf_DV, y=test_order_list)
    print('\nLinearSVC    準確度：'+str(round(L_SVC_result*100, 2))+'%')

    from sklearn.metrics import classification_report
    pre = L_SVC.predict(test_Tfidf_DV)
    print(classification_report(test_order_list, pre))
    print("\n")

    from sklearn.metrics import hinge_loss
    pred_decision = L_SVC.decision_function(test_Tfidf_DV)
    loss = hinge_loss(test_order_list, pred_decision)
    print(loss)


    # from sklearn.model_selection import GridSearchCV
    # param_grid = {'C': [300]}
    # L_SVC = GridSearchCV(LinearSVC(), param_grid, cv=5, return_train_score=True)
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

    # x = []
    # y = []
    # for testc in range(1, 200):
    #     x.append(testc*0.01)
    #     print(testc*0.01)
    #     L_SVC = LinearSVC(C=testc*0.01)
    #     L_SVC.fit(Tfidf_DV, train_order_list)
    #     L_SVC_result = L_SVC.score(X=test_Tfidf_DV, y=test_order_list)
    #     y.append(L_SVC_result)
    #     print('\nLinearSVC    準確度：'+str(round(L_SVC_result*100, 2))+'%')

    # plt.plot(x, y, color='b')
    # plt.xlabel('C')  # 設定x軸標題
    # plt.ylabel('score')  # 設定x軸標題
    # # plt.xticks([0.0, 0.27,  0.4,  0.6, 0.8, 1.0,1.2,1.4,1.6,1.8,2.0],)  # 設定x軸label以及垂直顯示
    # plt.title('result')  # 設定圖表標題
    # plt.show()

    # from sklearn.metrics import hinge_loss

    # x = []
    # y = []
    # for testc in range(1, 200):
    #     x.append(testc*0.01)
    #     print(testc*0.01)
    #     L_SVC = LinearSVC(C=testc*0.01)
    #     L_SVC.fit(Tfidf_DV, train_order_list)
    #     L_SVC_result = hinge_loss(test_order_list, L_SVC.decision_function(test_Tfidf_DV), labels=np.array([0, 1, 2]))
    #     y.append(L_SVC_result)
    #     print('\nLinearSVC    準確度：'+str(round(L_SVC_result*100, 2))+'%')

    # plt.plot(x, y, color='b')
    # plt.xlabel('C')  # 設定x軸標題
    # plt.ylabel('loss')  # 設定x軸標題
    # # plt.xticks([0.0, 0.27,  0.4,  0.6, 0.8, 1.0,1.2,1.4,1.6,1.8,2.0],)  # 設定x軸label以及垂直顯示
    # plt.title('result')  # 設定圖表標題
    # plt.show()
