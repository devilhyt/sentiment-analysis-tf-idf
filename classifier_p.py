# coding=utf-8
import sys

if __name__ == "__main__" and len(sys.argv) <= 2:
    print("錯誤！請從命令行輸入至少 2 個 PTT 看板參數")
    sys.exit()

import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

if __name__ == "__main__":
    board_list = [name for name in sys.argv[1:]]

    # score_combine_dicts = []
    # score_order_list = []
    train_combine_dicts = []
    train_order_list = []
    train_num = 2400
    score_num = 600
    for order, board in enumerate(board_list):  # 合成看板 dict_list 以及建立order_list
        with open('./data/' + board + ".pickle", 'rb') as pkl:
            tmp_list = pickle.load(pkl)
            # score_order_list += [order]*score_num
            # score_combine_dicts += tmp_list[0:score_num]
            del tmp_list[0:score_num]

            train_order_list += [order]*train_num
            train_combine_dicts += tmp_list[0:train_num]
            del tmp_list[0:train_num]

    conv_DV = DictVectorizer()
    calc_Tfidf = TfidfTransformer()
    DictVect = conv_DV.fit_transform(train_combine_dicts)
    Tfidf_DV = calc_Tfidf.fit_transform(DictVect)

    import umap
    import umap.plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    td = Tfidf_DV.todense()
    train_order_list_np = np.asarray(train_order_list)

    fit = umap.UMAP(n_components=3, n_neighbors=25, min_dist=0.3)
    u = fit.fit_transform(td)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_axis_off()

    color = ["blue", "yellow", "green"]
    for order, label in enumerate(board_list):
        arr_start = order*train_num
        arr_end = (order+1)*train_num
        ax.scatter(u[arr_start:arr_end, 0], u[arr_start:arr_end, 1], u[arr_start:arr_end,
                   2], c=color[order], s=8, label=label)
    ax.legend(loc='upper left')
    plt.show()
