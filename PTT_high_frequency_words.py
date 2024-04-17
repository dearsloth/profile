import requests
from bs4 import BeautifulSoup
import datetime
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import jieba
import jieba.analyse
from collections import Counter # 次數統計
import re


url="https://www.ptt.cc/bbs/Stock/index.html"


# 存N天內的文章標題與文章連結
def get_all_href(url,N):
    article={}
    today = datetime.datetime.now()
    last_month =  today - datetime.timedelta(days=N)
    last_month_date = last_month.strftime("%m/%d")
    terminal = False #若數到了N天，則結束整個迴圈
    for page in range(1,1000):     
        if terminal == False:
            #print(url)
            r = requests.get(url)
            soup = BeautifulSoup(r.text, "html.parser")  #擷取整個網頁的html
            results = soup.select("div.title")  #選取div的class="title"標籤
            dates = soup.select("div.date")
            #print(dates)
            for date in dates:  #該頁的日期
                #print(date.text)
                if int(date.text.split('/')[0])<10:
                    new_date = "0"+date.text[1:]
                    if new_date != last_month_date:  #若無超出N天，則存入文章標題與連結
                        #print(page, new_date)
                        for item in results:  #該頁的文章標題與連結 
                            a_item = item.find_all("a")
                            #print(a_item)
                            title = item.text
                            title = title.replace('\n','')
                            if a_item:  # 刪除掉的文章略過
                                article[title]='https://www.ptt.cc'+ item.select_one('a').get('href')
                                #print(title, 'https://www.ptt.cc'+ item.select_one('a').get('href'))
                                #print(date.text)
                    else:
                        terminal = True
                        break
                else:
                    if date.text != last_month_date:
                       # btn = soup.select('div.btn-group > a')
                        #print(btn)
                       # up_page_href = btn[3]['href']
                        #next_page_url = 'https://www.ptt.cc' + up_page_href
                        #url = next_page_url
                        for item in results:
                            a_item = item.find_all("a")
                            #print(a_item)
                            title = item.text
                            title = title.replace('\n','')
                            if a_item:  # 刪除掉的文章略過
                                article[title]='https://www.ptt.cc'+ item.select_one('a').get('href')
                                #print(title, 'https://www.ptt.cc'+ item.select_one('a').get('href'))
                                #print(date.text)
                    else:
                        terminal = True
                        break
            btn = soup.select('div.btn-group > a')
            #print(btn)
            up_page_href = btn[3]['href']
            next_page_url = 'https://www.ptt.cc' + up_page_href
            url = next_page_url
               
        else:
            break
    return article


record = get_all_href(url, 30)
#print(record)





# 開啟文章
def get_article_content(article_dic):
    article={}
    #i=0  #驗證數量正確
    for title in article_dic:
        #i+=1
        #print(len(article_dic),i)
        url=article_dic[title]
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "lxml")
        #print(url)
        #print(soup.text)
        article[title]=soup.text
    return  article  
        
article_dic = get_article_content(record)   
#print(article_dic)
 



 
# 文字雲
dictfile = "D:\\Desktop\\WordCloud\\dict.txt"  # 字典檔
stopfile = "D:\\Desktop\\WordCloud\\stopwords.txt"  # stopwords
fontpath = "C:\\Windows\\Fonts\\msjhbd.ttc"  # 字型檔
pngfile = "D:\\Desktop\\WordCloud\\circle.png"  # 剛才下載存的底圖

alice_mask = np.array(Image.open(pngfile))

jieba.set_dictionary(dictfile)
jieba.analyse.set_stop_words(stopfile)

text=""
for title in article_dic:
    text+=article_dic[title]
text=re.sub("[A-Za-z0-9標題看板踢踢時間網址\.]",'',text)
tags = jieba.analyse.extract_tags(text, topK=50)

seg_list = jieba.lcut(text, cut_all=False)
dictionary = Counter(seg_list)

freq = {}
for ele in dictionary:
    if ele in tags:
        freq[ele] = dictionary[ele]
#print(freq) # 計算出現的次數

wordcloud = WordCloud(background_color="white", mask=alice_mask, contour_width=3, contour_color='steelblue', font_path= fontpath).generate_from_frequencies(freq)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()