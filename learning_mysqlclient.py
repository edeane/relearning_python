# -*- coding: utf-8 -*-
"""

MySQL
https://dev.mysql.com/downloads/installer/
TCP/IP Port: 3306 X Protocol Port: 33060

mysqlclient

download the wheel  https://www.lfd.uci.edu/~gohlke/pythonlibs/#mysql-python
pip install C:/Users/edeane/Downloads/mysqlclient-1.4.2-cp37-cp37m-win_amd64.whl

"""

import numpy as np
import pandas as pd
import sys
import MySQLdb
import getpass

print('-' * len(sys.version), '\n', sys.version, '\n', '-' * len(sys.version), sep='')

local_con_det = {'host': 'localhost',
                 'user': 'root',
                 'passwd': getpass.getpass('passwd'),
                 'db': 'sakila',
                 'port': 3306}

local_con = MySQLdb.connect(**local_con_det)

customer_star = pd.read_sql('''
SELECT
	*
FROM customer c
LEFT JOIN address a USING(address_id)
LEFT JOIN city ci USING(city_id)
LEFT JOIN country co USING(country_id)
''', con=local_con)

local_con.close()
customer_star.head()

# multiline strings with
# 1. ''' abc ''' or
# 2.
# s = ('abasdfasfdkj asdfl '
#      'aslfkaskldf;jasf'
#      'asldfkas loiwuer '
#      'asodfiuasdofi')
# 3.
# s = 'alskdfj sdf sdfoiasdf' \
#     'f;lasjdfoif asldfklsf ' \
#     'adsfljaslfd oiwero asdf' \
#     'aosdfweoir ' \
#     'asdfjlfasdf' \
#     'lfjoiwurr'




