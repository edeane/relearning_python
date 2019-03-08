"""
Web Scraping
https://realpython.com/python-web-scraping-practical-introduction/

"""

from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
from web_scrape_funs import simple_get
from pprint import pprint
import pandas as pd
import re
from time import sleep
import random
import seaborn as sns



raw_math_url = 'http://www.fabpedigree.com/james/mathmen.htm'
raw_math_html = simple_get(raw_math_url)

if raw_math_html is not None:
    math_html = BeautifulSoup(raw_math_html, 'html.parser')
    names = []
    for i in math_html.select('li'):
        names.append(i.text.split('\n')[0].strip())

names

math_df = pd.DataFrame({'math_names': names})
math_df.head()
math_df.shape

url = []
for math_name in math_df['math_names']:
    goog_sear = math_name.replace(' ', '%20')
    goog_sear = goog_sear + '%20wikipedia'
    goog_str = (f'https://www.google.com/search?q={goog_sear}')
    print(goog_str)
    goog_res = get(goog_str)

    goog_html = BeautifulSoup(goog_res.content)

    for idx, val in enumerate(goog_html.select('a')):
        if ' - Wikipedia' in val.text:
            print(idx, val)
            val_url = val['href'].split('&sa=U')[0].split('/url?q=')[1]
            url.append(val_url)
            break

    sleep_time = random.randint(5, 10)
    print(f'sleeping {sleep_time} seconds')
    sleep(sleep_time)

url
math_df['url'] = url
math_df.head()

math_df['wiki_page'] = math_df.loc[:, 'url'].str.split('https://en.wikipedia.org/wiki/', expand=True).iloc[:, 1]
math_df.head()
# math_df.to_csv(r'C:\Users\edeane\PycharmProjects\relearning_python\data\math_names.csv', index=False)

math_df = pd.read_csv(r'C:\Users\edeane\PycharmProjects\relearning_python\data\math_names.csv')
math_df

wiki_views = []
for wiki_page in math_df['wiki_page']:

    wiki_url = f'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/' \
               f'en.wikipedia.org/all-access/all-agents/' \
               f'{wiki_page}/monthly/20130101/20190101'

    wiki_res = get(wiki_url)
    wiki_res_json = wiki_res.json()

    tot_views = 0
    try:
        for i_month in wiki_res_json['items']:
            tot_views += int(i_month['views'])
    except:
        pass
    print(wiki_page, tot_views)
    wiki_views.append(tot_views)


len(wiki_views)
wiki_views
math_df['tot_wiki_views'] = wiki_views
math_df.head()

math_df['website_ranking'] = range(1, 100)
math_df.sort_values('tot_wiki_views', ascending=False, inplace=True)
math_df.head()
math_df.to_csv(r'C:\Users\edeane\PycharmProjects\relearning_python\data\math_names_final.csv', index=False)

math_df['tot_wiki_views']
math_df.columns

sns.distplot(math_df['tot_wiki_views'], kde=False)
sns.barplot(data=math_df, y='wiki_page', x='tot_wiki_views')






