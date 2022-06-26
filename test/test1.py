#coding=utf-8
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

str1="邻居养了腊肠狗，腿短短的，好可爱，每次经过他们家门口，我就会蹲下来招招手，10次里有8次它不会理我，只有2次会开心的跑来让我摸摸，我感觉像抽奖一样，它让我摸摸的时候我就觉得很开心~~~，每天下班的小确幸"

print(f"原文（簡體版）：")
print(str1)
print("")

# -------------------------------------
from opencc import OpenCC
cc = OpenCC('s2t')
str2="鄰居養了臘腸狗，腿短短的，好可愛，每次經過他們家門口，我就會蹲下來招招手，10次裡有8次牠不會理我，只有2次會開心的跑來讓我摸摸，我感覺像抽獎一樣，牠讓我摸摸的時候我就覺得很開心~~~，每天下班的小確幸"

print(f"簡體轉繁體：")
print(cc.convert(str2))
print("")

# -------------------------------------
from ckiptagger import WS
ws = WS("./ckip_data")
import string
punctuation_list = [p for p in string.punctuation]

print(f"斷詞：")
segmented_list = ws([str2], sentence_segmentation=True, segment_delimiter_set = punctuation_list)
print(segmented_list)
print("")

# -------------------------------------
from collections import defaultdict
#======載入stopword======
with open("./data/stopwords.txt", 'r', encoding = "utf-8") as sw: # stopwords.txt 存入list
    stopwords_list = [line.rstrip('\n') for line in sw]

tmp_list = []
for cnt,segmented in enumerate(segmented_list):
    for cnt2, word in enumerate(segmented):
     if word not in stopwords_list:
         tmp_list.append(word)

print(f"排除停用字：")
print(tmp_list)
print("")

#======分別將各篇的斷詞結果整理成dict======
dict_list=[]
for cnt,segmented in enumerate(segmented_list):
    tmp_dict= defaultdict(int)
    for word in segmented:
        if word not in stopwords_list:
            tmp_dict[word]+=1
    dict_list.append(tmp_dict)        
print(f"計算單詞出現次數：")
print(dict_list)
print("")

