# -*- coding: utf-8 -*-
"""

MySQL
https://dev.mysql.com/downloads/installer/
TCP/IP Port: 3306 X Protocol Port: 33060

mysqlclient

download the wheel  https://www.lfd.uci.edu/~gohlke/pythonlibs/#mysql-python
pip install C:/Users/edeane/Downloads/mysqlclient-1.4.2-cp37-cp37m-win_amd64.whl

multiline strings with
1. ''' abc ''' or
2.
s = ('abasdfasfdkj asdfl '
     'aslfkaskldf;jasf'
     'asldfkas loiwuer '
     'asodfiuasdofi')
3.
s = 'alskdfj sdf sdfoiasdf' \
    'f;lasjdfoif asldfklsf ' \
    'adsfljaslfd oiwero asdf' \
    'aosdfweoir ' \
    'asdfjlfasdf' \
    'lfjoiwurr'

"""

# data manip
import pandas as pd

# mysql client
import MySQLdb

# need sqlalchemy for df.to_sql
from sqlalchemy import create_engine

# insert pass hidden
from getpass import getpass

# toy datasets
from sklearn import datasets

print('MySQLdb verions:', MySQLdb.__version__)


def create_con_det(db_name='sklearn_toy_data'):

    return {'host': 'localhost',
            'user': 'root',
            'passwd': getpass('passwd'),
            'db': db_name,
            'port': 3306}


def create_sqlalch_str(db_name='sklearn_toy_data'):
    password = getpass('password')
    return 'mysql+mysqldb://root:' + password + '@localhost:3306/' + db_name


use_mysql_or_sqlalchemy = 'sqlalchemy'

if use_mysql_or_sqlalchemy == 'mysql':
    # connect with MySQLdb package
    print('using mysqlclient')
    local_con_det = create_con_det('sakila')
    use_eng_con = MySQLdb.connect(**local_con_det)
elif use_mysql_or_sqlalchemy == 'sqlalchemy':
    # or connect with sqlalchemy create_engine
    print('using sqlalchemy')
    sqlalch_str = create_sqlalch_str()
    use_eng_con = create_engine(sqlalch_str, echo=False)


boston_df = pd.read_sql('''
SELECT
	*
FROM boston
''', con=use_eng_con)

print(boston_df.head())

sk_datasets = dir(datasets)
load_not_needed = ['load_files', 'load_sample_image', 'load_sample_images', 'load_svmlight_file',
                   'load_svmlight_files', 'load_svmlight_files', 'load_mlcomp']
sk_datasets = [sk_dataset for sk_dataset in sk_datasets if 'load_' in sk_dataset and sk_dataset not in load_not_needed]
print('sk_datasets:', sk_datasets)

sk_datasets_dict = dict()

for i in range(len(sk_datasets)):
    data_name = sk_datasets[i]
    print(data_name)
    data_fun = getattr(datasets, data_name)
    data = data_fun()

    if 'feature_names' in data.keys():
        df = pd.DataFrame(data.data, columns=data.feature_names)
    else:
        df = pd.DataFrame(data.data, columns=['p' + str(i) for i in range(data.data.shape[1])])
    if len(data.target.shape) > 1:
        df = pd.concat([df, pd.DataFrame(data.target, columns=data.target_names)], axis=1)
    elif 'target_names' in data.keys():
        target_df = pd.DataFrame({'target': data.target})
        target_names_df = pd.DataFrame({'target_names': data.target_names})
        target_names_df['target'] = range(len(target_names_df))
        target_df = pd.merge(target_df, target_names_df, how='left', on='target')
        df = pd.concat([df, target_df], axis=1)
    else:
        df = pd.concat([df, pd.DataFrame({'target': data.target})], axis=1)
    print(df.head())

    sk_datasets_dict[data_name.split('load_')[1]] = df


for db_name in sk_datasets_dict.keys():
    print('uploading:', db_name)
    sk_datasets_dict[db_name].to_sql(db_name, con=use_eng_con, if_exists='replace', index=False)

