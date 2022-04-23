# coding=utf-8
import sys
import requests
import time
from bs4 import BeautifulSoup

PTT_URL = "https://www.ptt.cc"
session = requests.session()
PTT_over18_URL = PTT_URL + '/ask/over18'
Form_Data = {'from': '/bbs/index.html', 'yes': 'yes'}

def get_articles_content(article_href):  # 取得單篇內文
    r = session.get(article_href)
    soup = BeautifulSoup(r.text, "html.parser")
    try:
        time = soup.select('span.article-meta-value')[3].text
        content = soup.find(id="main-content").text  # content 文章內文
        target_content = (u'※ 發信站: 批踢踢實業坊(ptt.cc)')
        content = content.split(target_content)  # 去除掉 target_content
        content = content[0].split(time)
        main_content = content[1].replace('\n', '，')  # 逗號代替換行
        return main_content
    except:
        return None

def get_page_article_href_list(page_URL):  # 取得本頁所有文章的網址
    page_article_href_list = []
    r = session.get(page_URL)
    soup = BeautifulSoup(r.text, "html.parser")  # 使用html.parser解析
    results = soup.findAll("div", {"class": "title"})
    for item in results:
        try:
            item_href = item.find("a").attrs["href"]
            page_article_href_list.append(item_href)
        except:
            pass
    return page_article_href_list


def write_file(mode, data):
    with open('./data/' + board + ".txt", mode, encoding="UTF-8") as f:
        if data is not None:
            f.write(data + "\n")


def main_function(URL):
    global cnt
    r = session.get(URL)
    soup = BeautifulSoup(r.text, "html.parser")  # 使用html.parser解析
    page_article_href_list = get_page_article_href_list(page_URL=URL)

    for url_sub in page_article_href_list:
        articles_content = get_articles_content(PTT_URL + url_sub)
        if articles_content is not None:
            write_file("a", articles_content)
            cnt += 1
            if cnt % 100 == 0:
                print("目前已處理 "+str(cnt)+" 篇文章")
                # time.sleep(5)
            if cnt == target:
                break

    # 找btn-group-paging
    btn = soup.select('div.btn-group.btn-group-paging > a')
    if(str(btn[1]) == '<a class="btn wide disabled">‹ 上頁</a>') or cnt == target:
        print("完成")
    else:
        main_function(PTT_URL + btn[1]['href'])

if __name__ == "__main__":
    res = session.post(PTT_over18_URL, data=Form_Data) # 通過ask over 18

    while True:
        cnt = 0
        board = input("請輸入看板名稱: ")
        r = session.get(PTT_URL + "/bbs/" + board + "/index.html")
        if board == '-exit':
            sys.exit()
        elif (r.status_code == requests.codes.ok):
            soup = BeautifulSoup(r.text, "html.parser")
            # 找btn-group-paging
            btn = soup.select('div.btn-group.btn-group-paging > a')
            # 顯示總頁數
            print(
                'Get 成功！共有' + str(int((btn[1]['href'].split('index')[1]).replace('.html', '')) + 1) + '頁')
            page = input("請輸入起始頁(向前爬取): ")
            target = int(input("請輸入爬取文章數量: "))
            write_file("w", None)  # 創建.txt
            main_function(PTT_URL + "/bbs/" + board +
                          "/index" + page + ".html")
        else:
            print("Get 失敗")
