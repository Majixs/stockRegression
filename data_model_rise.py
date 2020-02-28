import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import ensemble
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet



qqq = pd.DataFrame.from_csv('QQQ.csv')
i_qqq = list(range(len(qqq['Open'])))
qqq = qqq.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], 1)
time = qqq.index.values
all_time = time
time = time[4137:4637]
qqq = qqq.set_index([i_qqq])

aapl = pd.DataFrame.from_csv('AAPL.csv')
aapl = aapl.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], 1)
aapl = aapl.set_index([i_qqq])

adbe = pd.DataFrame.from_csv('ADBE.csv')
adbe = adbe.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], 1)
adbe = adbe.set_index([i_qqq])

amgn = pd.DataFrame.from_csv('AMGN.csv')
amgn = amgn.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], 1)
amgn = amgn.set_index([i_qqq])

amzn = pd.DataFrame.from_csv('AMZN.csv')
amzn = amzn.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], 1)
amzn = amzn.set_index([i_qqq])

cmcsa = pd.DataFrame.from_csv('CMCSA.csv')
cmcsa = cmcsa.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], 1)
cmcsa = cmcsa.set_index([i_qqq])

csco = pd.DataFrame.from_csv('CSCO.csv')
csco = csco.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], 1)
csco = csco.set_index([i_qqq])

intc = pd.DataFrame.from_csv('INTC.csv')
intc = intc.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], 1)
intc = intc.set_index([i_qqq])

msft = pd.DataFrame.from_csv('MSFT.csv')
msft = msft.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], 1)
msft = msft.set_index([i_qqq])

nvda = pd.DataFrame.from_csv('NVDA.csv')
nvda = nvda.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], 1)
nvda = nvda.set_index([i_qqq])

txn = pd.DataFrame.from_csv('TXN.csv')
txn = txn.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], 1)
txn = txn.set_index([i_qqq])

all_data = pd.concat([txn, msft, intc, csco, cmcsa, amzn, adbe, aapl], axis=1)
target = qqq['Close'].tolist()
all_train = all_data.as_matrix()
all_target = target


train = all_data.drop(all_data.index[4137:4637])
test = all_data.drop(all_data.index[:4137])
qqq_train = target[:4137]
qqq_test = target[4137:4637]


train = train.as_matrix()
test = test.as_matrix()
qqq_train = np.asarray(qqq_train)
qqq_test = np.asarray(qqq_test)
compare = qqq_test.tolist()

#models
#decision tree
clf = tree.DecisionTreeRegressor()
clf.fit(train, qqq_train)
predict_tree = clf.predict(test)

#forest
regr = ensemble.RandomForestRegressor()
regr.fit(train, qqq_train)
pred_forest = regr.predict(test)

#lasso rise prise
lasso = Lasso(alpha = 0.01)
pred_lasso = lasso.fit(train, qqq_train).predict(test)
all_pred_lasso = lasso.predict(all_train)
pred_lasso = pred_lasso.tolist()
all_pred_lasso = all_pred_lasso.tolist()
pred_lasso = [x * 0.9 for x in pred_lasso]
all_pred_lasso = [x * 0.9 for x in all_pred_lasso]

#ridge rise prise
ridge = Ridge(alpha = 0.01)
pred_ridge = ridge.fit(train, qqq_train).predict(test)
pred_ridge = pred_ridge.tolist()
#pred_ridge = [x * 0.9 for x in pred_ridge]

#elastic n rise prise
en = ElasticNet(alpha = -4)
pred_en = en.fit(train, qqq_train).predict(test)
pred_en = pred_en.tolist()
#pred_en = [x * 0.9 for x in pred_en]

success_day_lasso = []
for i in range(len(compare)):
    if compare[i] >= pred_lasso[i]:
        success_day_lasso.append(1)
p_success_trade_lasso = len(success_day_lasso)/len(time)

success_day_ridge = []
for i in range(len(compare)):
    if compare[i] >= pred_ridge[i]:
        success_day_ridge.append(1)
p_success_trade_ridge = len(success_day_ridge)/len(time)

success_day_en = []
for i in range(len(compare)):
    if compare[i] >= pred_en[i]:
        success_day_en.append(1)
p_success_trade_en = len(success_day_en)/len(time)

success_day__all_lasso = []
for i in range(len(all_target)):
    if all_target[i] >= all_pred_lasso[i]:
        success_day__all_lasso.append(1)
p_success_all_trade_lasso = len(success_day__all_lasso)/len(all_time)


plt.figure()
plt.scatter(time, qqq_test, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(time, pred_lasso, color="cornflowerblue", label="model", linewidth=2)
plt.xlabel("time")
plt.ylabel("цена")
plt.title("Lasso")
plt.legend()

plt.figure()
plt.scatter(time, qqq_test, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(time, pred_ridge, color="cornflowerblue", label="model", linewidth=2)
plt.xlabel("time")
plt.ylabel("price")
plt.title("Ridge")
plt.legend()

plt.figure()
plt.scatter(time, qqq_test, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(time, pred_en, color="cornflowerblue", label="model", linewidth=2)
plt.xlabel("time")
plt.ylabel("price")
plt.title("Elastic_New")
plt.legend()

plt.figure()
plt.scatter(all_time, all_target, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(all_time, all_pred_lasso, color="cornflowerblue", label="model", linewidth=2)
plt.xlabel("time")
plt.ylabel("price")
plt.title("All data Lasso")
plt.legend()




#mean_squared
sq_er_lasso= mean_squared_error(qqq_test , pred_lasso)
rmse_lasso = sqrt(sq_er_lasso)
r2_score_er_lasso = r2_score(qqq_test , pred_lasso)

sq_er_ridge= mean_squared_error(qqq_test , pred_ridge)
rmse_ridge = sqrt(sq_er_ridge)
r2_score_er_ridge = r2_score(qqq_test , pred_ridge)

sq_er_en= mean_squared_error(qqq_test , pred_en)
rmse_en = sqrt(sq_er_en)
r2_score_er_en = r2_score(qqq_test , pred_en)

print('p_success_trade_en: ', p_success_trade_en)
print('rmse_en: ', rmse_en)
print('r2_score_er_en: ', r2_score_er_en)
print('-   -   -')

print('p_success_trade_ridge: ', p_success_trade_ridge)
print('rmse_ridge: ', rmse_ridge)
print('r2_score_er_ridge: ', r2_score_er_ridge)
print('-   -   -')

print('p_success_trade_lasso: ', p_success_trade_lasso)
print('rmse_lasso: ', rmse_lasso)
print('r2_score_er_lasso: ', r2_score_er_lasso)



