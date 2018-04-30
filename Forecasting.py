import quandl
import pandas as pd
import numpy as np
import datetime
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing, cross_validation, svm
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
quandl.ApiConfig.api_key = "Hyqv1_kd5xeAzSemDUeH"
get_ipython().magic(u'matplotlib inline')
def Forecast(ticker):
    df = quandl.get("WIKI/"+ticker)
    df = df.iloc[-500:]
    func = ['mean','std','skew','kurt']
    time = [2,4,7,30,60,120,250]
    Z = pd.DataFrame()
    for f in func:
        for t in time:
            aux = pd.DataFrame()
            for i in range(249):
                aux[df.index[i+251]] = list(df.iloc[i-t+251:i+251]['Adj. Close'])
            Z[f+str(t)] = aux.apply(f,axis=0)  
    Z.drop(['skew2','kurt2'],axis=1, inplace =True)
    W = pd.merge(Z, df[['Adj. Close']],left_index=True,right_index=True)
    X = W.drop(['Adj. Close'],1).fillna(0)
    y = W[['Adj. Close']]
    mm1 = MinMaxScaler()
    mm2 = MinMaxScaler()
    mm1.fit(X)
    mm2.fit(y)
    Xm = pd.DataFrame(mm1.transform(X))
    ym = pd.DataFrame(mm2.transform(y))
    Xt, Xv, yt, yv = cross_validation.train_test_split(Xm, ym, test_size = 0.3)
    model = MLPRegressor()
    param_grid = dict(
    activation = ['identity', 'logistic', 'tanh', 'relu'],
    solver = ['lbfgs', 'sgd', 'adam'],
    learning_rate = ['constant', 'invscaling', 'adaptive'],)
    grid = GridSearchCV(cv = 3, 
                        estimator = model,
                        n_jobs = -1,
                        param_grid=param_grid,)
    grid.fit(Xm,ym)
    model = grid.best_estimator_
    model.fit(Xt,yt)
    print 'ACC Validate {:.2%}'.format(model.score(Xv,yv))
    print 'ACC Train    {:.2%}'.format(model.score(Xt,yt))
    model.fit(Xm,ym)
    pred = pd.DataFrame(model.predict(Xm[-150:]))
    x = Z.index[range(1,151)]
    plt.plot(x, list(y['Adj. Close'][-151:-1]), label='Real')
    plt.plot(x, mm2.inverse_transform(pred), label='Prediccion')
    plt.title(ticker)
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(15.5, 8.5)
    plt.show()