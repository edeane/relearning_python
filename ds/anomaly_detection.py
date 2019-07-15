"""
Anomaly Detection for Dummies
https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.ensemble import IsolationForest
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
plt.interactive(False)

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 1000)

# Univariate Anomaly Detection

df = pd.read_excel('C:/Users/edeane/Downloads/Superstore.xls')
df

df.describe()

df['Sales'].describe()

def create_index(df, col):
    df.sort_values(by=col, inplace=True, ascending=True)
    df = df.reset_index(drop=True)
    df = df.reset_index()
    df = df.rename(columns={'index': f'{col.lower()}_index'})
    return df

df = create_index(df, 'Sales')
df = create_index(df, 'Profit')
df

col = 'Profit'

sns.regplot(data=df, x=f'{col.lower()}_index', y=col, fit_reg=False, ci=None)
plt.show()

sns.distplot(df[col])
plt.show()


def print_skew_kurt(df, col):
    print(f"{col} skewness: {df[col].skew()}")
    print(f"{col} kurt: {df[col].kurt()}")



print_skew_kurt(df, 'Sales')
print_skew_kurt(df, 'Profit')

# Anomaly Detection with Isolation Forest

col = 'Sales'

iso_for = IsolationForest()
iso_for.fit(df[[col]])

xx = np.linspace(df[col].min(), df[col].max(), len(df)).reshape(-1,1)

yy_score = iso_for.decision_function(xx)
yy_pred = iso_for.predict(xx)

iso_df = pd.DataFrame({'xx': xx.T[0], 'yy_score': yy_score, 'yy_pred': yy_pred})
iso_df

sns.lineplot(data=iso_df, x='xx', y='yy_score', estimator=None)
plt.axvline(iso_df.loc[iso_df['yy_pred']==1, 'xx'].min())
plt.axvline(iso_df.loc[iso_df['yy_pred']==1, 'xx'].max())
plt.show()

iso_df[iso_df['yy_pred'] == -1].describe()
iso_df[iso_df['yy_pred'] == 1].describe()

sns.regplot(data=df, x='Sales', y='Profit')
plt.show()


# Cluster-based Local Outlier Factor (CBLOF)

from sklearn.preprocessing import MinMaxScaler
sca = MinMaxScaler()
for i in ('Sales', 'Profit'):
    df[f'{i}_scale'] = sca.fit_transform(df[[i]])

out_frac = .01

classers = {
    'cblof': CBLOF(contamination=out_frac, check_estimator=False, random_state=42),
    'hbos': HBOS(contamination=out_frac),
    'isof': IForest(contamination=out_frac, random_state=0),
    'knn': KNN(contamination=out_frac)
}


clf = classers['isof']
xx, yy = np.meshgrid(np.linspace(0, 1, 1000),
                     np.linspace(0, 1, 1000))
x = df[['Sales_scale', 'Profit_scale']]
x
clf.fit(x)
df['clf_pred'] = clf.predict(x)
df['clf_score'] = clf.decision_function(x) * -1
df.describe()

thresh = np.percentile(df['clf_score'], 100 * out_frac)
z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
z = z.reshape(xx.shape)

plt.figure(figsize=(8,8))
plt.contourf(xx, yy, z, levels=np.linspace(z.min(), thresh, 7), cmap=plt.cm.Blues_r)
plt.contour(xx, yy, z, levels=[thresh], linewidths=2, colors='red')
plt.contourf(xx, yy, z, levels=[thresh, z.max()], colors='orange')
plt.scatter(df.loc[df['clf_pred']==0, 'Sales_scale'], df.loc[df['clf_pred']==0, 'Profit_scale'], c='white', s=20,
            edgecolors='k', alpha=.5)
plt.scatter(df.loc[df['clf_pred']==1, 'Sales_scale'], df.loc[df['clf_pred']==1, 'Profit_scale'], c='black', s=20,
            edgecolors='k', alpha=.5)
plt.show()







































