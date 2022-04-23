# coding=utf-8
import sys

if __name__ == "__main__" and len(sys.argv) <= 1:
    print("錯誤！請從命令行輸入至少 1 個 PTT 看板參數")
    sys.exit()

#===================================================

from ckiptagger import WS
import tensorflow as tf
from tensorflow.python.util import deprecation
import os
import pickle
import string
from opencc import OpenCC
from collections import defaultdict  # 使用 dict 儲存資料

print('正在載入ckip model...')
# ===========Suppress as many warnings as possible============
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# =============================================================
ws = WS("./ckip_data")  # Load model without GPU
print('載入完成！\n')


def segmentation(content_list):
    global ws
    cc = OpenCC('s2t')

    # ======ckip斷詞======
    # print('開始斷詞...')
    punctuation_list = [p for p in string.punctuation]
    segmented_list = []

    for cnt, content in enumerate(content_list):
        # 使用opencc簡轉繁 再用ckip斷詞
        segmented_list.append(ws([cc.convert(content)],
                                 sentence_segmentation=True,
                                 segment_delimiter_set=punctuation_list)[0])
        if (cnt+1) % 100 == 0:
            print('ckip: 已斷詞 ' + str(cnt+1) + ' 篇')
    # print('斷詞完成！開始轉換dict...')

    # ======載入stopword======
    with open("./data/stopwords.txt", 'r', encoding="utf-8") as sw:  # stopwords.txt 存入list
        stopwords_list = [line.rstrip('\n') for line in sw]

    # ======分別將各篇的斷詞結果整理成dict======
    dict_list = []
    for cnt, segmented in enumerate(segmented_list):
        tmp_dict = defaultdict(int)
        for word in segmented:
            if word not in stopwords_list:
                tmp_dict[word] += 1
        dict_list.append(tmp_dict)
        if (cnt+1) % 100 == 0:
            print('dict: 已轉換 ' + str(cnt+1) + ' 篇')
    return dict_list


if __name__ == "__main__":
    board_list = [name for name in sys.argv[1:]]

    for board in board_list:
        # ======讀.txt 存入list======
        with open('./data/' + board + ".txt", 'r', encoding="utf-8") as f:
            content_list = [line.rstrip('\n') for line in f]

        # ======預處理======
        print(board)
        dict_list = segmentation(content_list)
        print('轉換完成！')
        with open('./data/' + board + ".pickle", 'wb') as pkl:
            pickle.dump(dict_list, pkl)
