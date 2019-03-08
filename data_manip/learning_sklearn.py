'''
sklearn
- https://scikit-learn.org/stable/tutorial/index.html
- https://www.dataquest.io/blog/sci-kit-learn-tutorial/

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pd.set_option('display.max_columns', 100)

# X https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html

def poly_inter(x):
    return x * np.sin(x)

x = np.linspace(0, 10, 100)
rng = np.random.RandomState(0)
rng.shuffle(x)
x_samp = np.sort(x[:20])
x = np.sort(x)
y_samp = poly_inter(x_samp)
x_samp
y_samp

x_samp = x_samp[:, np.newaxis]
x = x[:, np.newaxis]

plt.ioff()

plt.plot(x, poly_inter(x), label='truth')
plt.scatter(x_samp, y_samp, label='sample')

for count, degree in enumerate([3, 4, 5, 6, 7]):
    model = make_pipeline(PolynomialFeatures(degree), ElasticNet())
    model.fit(x_samp, y_samp)
    y_pred = model.predict(x)
    plt.plot(x, y_pred, label=f'deg: {degree}')

plt.legend(loc='best')

plt.show()

poly = PolynomialFeatures(3)
poly.fit_transform(x_samp)[0]
x_samp[0, 0]
x_samp[0, 0] ** 2
x_samp[0, 0] ** 3


df = pd.DataFrame({'x1': np.arange(1,11), 'x2': np.arange(11, 21)})
df

poly = PolynomialFeatures(2)
poly.fit_transform(df)


# X https://towardsdatascience.com/polynomial-regression-bbe8b9d97491

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(0)

x = 2 - 3 * np.random.normal(0, 1, 20)
y = x - 2 * (x ** 2) + .5 * (x ** 3) + np.random.normal(-3, 3, 20)

x
y

df = pd.DataFrame({'x': x, 'y': y})
df
df.sort_values('x', ascending=True, inplace=True)
df.reset_index(inplace=True)
df

plt.scatter(data=df, x='x', y='y', s=10)
plt.show()

lin_mod = LinearRegression()
lin_mod.fit(df[['x']], df['y'])
df['lin_y_pred'] = lin_mod.predict(df[['x']])
df

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(df[['x']])
poly_mod = LinearRegression()
poly_mod.fit(x_poly, df['y'])
df['poly_y_pred'] = poly_mod.predict(x_poly)


poly3 = PolynomialFeatures(degree=3)
x_poly3 = poly3.fit_transform(df[['x']])
poly3_mod = LinearRegression()
poly3_mod.fit(x_poly3, df['y'])
df['poly3_y_pred'] = poly3_mod.predict(x_poly3)

poly20 = PolynomialFeatures(degree=20)
x_poly20 = poly20.fit_transform(df[['x']])
poly20_mod = LinearRegression()
poly20_mod.fit(x_poly20, df['y'])
df['poly20_y_pred'] = poly20_mod.predict(x_poly20)
poly20_mod.coef_


plt.scatter(data=df, x='x', y='y', s=10)
plt.plot(df['x'], df['lin_y_pred'], color='r')
plt.plot(df['x'], df['poly_y_pred'], color='g')
plt.plot(df['x'], df['poly3_y_pred'], color='b')
plt.plot(df['x'], df['poly20_y_pred'], color='c')
plt.show()

df.head()


mod.coef_
mod.intercept_



mean_squared_error(df['y'], y_pred)
r2_score(df['y'], y_pred)




