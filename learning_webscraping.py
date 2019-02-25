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

raw_html = simple_get('https://realpython.com/blog/')
len(raw_html)

no_html = simple_get('https://realypython.com/blog/nope-not-gonna-find-it')

url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/' \
      'en.wikipedia.org/all-access/all-agents/' \
      'Python_(programming_language)/monthly/20130101/20190101'

wiki_res = get(url)
wiki_res_json = wiki_res.json()

len(wiki_res_json['items'])

tot_views = 0
for i_month in wiki_res_json['items']:
    tot_views += int(i_month['views'])

tot_views
wiki_res_json['items'][41]

wiki_res_df = pd.DataFrame(wiki_res_json['items'])
wiki_res_df.columns
wiki_res_df.head()

wiki_res_df.loc[:, 'views'].sum()


math_html
len(math_html.select('li'))


math_df.to_csv(r'C:\Users\edeane\PycharmProjects\relearning_python\data\math_names.csv', index=False)


math_html.select('li')[0].select('li')[0]
math_html.select('li')[1].select('li')



