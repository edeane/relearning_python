"""
Convert Google Chrome Bookmarks to csv

"""

from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
from pprint import pprint

bm_path = Path('data/')
bm_fp = bm_path / 'new_bookmarks.html'

with open(bm_fp, mode='r', encoding='UTF-8') as fp:
    soup = BeautifulSoup(fp)

soup
all_links = soup.body.find_all('a')

len(all_links)
type(all_links)
pprint(all_links)

string_list = []
link_list = []

for a_link in all_links:
    string_list.append(a_link.string)
    link_list.append(a_link.get('href'))


df = pd.DataFrame({'strings': string_list, 'links': link_list})

bm_fp = bm_path / 'bookmarks.csv'
df.to_csv(bm_fp, index=False, encoding='utf-8-sig')




