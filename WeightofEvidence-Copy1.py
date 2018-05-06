from __future__ import division
import pandas as pd
import numpy as np
from math import log
class WoE:
    def __init__(self, disc=None, cont=None):
        self.maps = None
        self.disc = disc
        self.cont = cont
        self.IV = None
    def fit(self, Z, y,bins=None):
        X = Z.copy()
        self.IV = pd.DataFrame([np.zeros(len(X.columns))],columns = X.columns)
        self.maps = pd.DataFrame()
        cols = X.columns
        X['var'] = y
        X['ID'] = range(len(X))
        for col in self.disc:
            a = X.pivot_table(aggfunc='count',columns='var',fill_value=0, index=col,values='ID').reset_index()
            a.loc[-1] =["TOTAL", sum(a[0]), sum(a[1])]
            lis = []
            for y in set(X[col].values):
                g = int(a[a[col]==y][1])/int(a[a[col]=='TOTAL'][1])
                b = int(a[a[col]==y][0])/int(a[a[col]=='TOTAL'][0])
                if g*b == 0 :
                    d = log((g+0.5)/(b+0.5))
                else:
                    d = log(g/b)
                self.IV[col] += float((g-b)*d)
                lis.append((y,d))
            lis1 = pd.DataFrame(columns=[col])
            lis1[col] = lis
            self.maps = pd.concat([self.maps, lis1],axis=1) 
        for col in self.cont:
            IV = []
            for i in bins:
                IV.append(0)
                X[col] = pd.cut(Z[col], bins = i)
                a = X.pivot_table(aggfunc='count',columns='var',fill_value=0, index=col,values='ID').reset_index()
                a.loc[-1] =["TOTAL", sum(a[0]), sum(a[1])]
                for y in set(X[col].values):
                    goods = float(int(a[a[col]==y][1])/int(a[a[col]=='TOTAL'][1]))
                    bads = float(a[a[col]==y][0]/int(a[a[col]=='TOTAL'][0]))
                    if (bads != 0)&(goods !=0):
                        d = log(bads/goods)
                        IV[-1] += float((bads-goods)*d)
                    else:
                        IV[-1] += -np.inf 
            IV = np.array(IV)
            armax = np.argmax(IV[IV <np.inf])
            M = int(bins[armax])
            y1 = min(Z[col])
            y2 = max(Z[col])
            B = [-np.inf]+[y1 + n*(y2-y1)/M for n in range(1,M)]+[np.inf]
            X[col] = pd.cut(Z[col], bins = M,include_lowest=True,right=True,labels= [x for x in range(1,M+1)])
            a = X.pivot_table(aggfunc='count',columns='var',fill_value=0, index=col,values='ID').reset_index()
            a.loc[-1] =["TOTAL", sum(a[0]), sum(a[1])]
            lis = []
            for y in set(X[col].values):
                g = int(a[a[col]==y][1])/int(a[a[col]=='TOTAL'][1])
                b = int(a[a[col]==y][0])/int(a[a[col]=='TOTAL'][0])
                if g*b == 0 :
                    d = log((g+0.5)/(b+0.5))
                else:
                    d = log(g/b)
                self.IV[col] += float((g-b)*d)
                lis.append((B[y-1],B[y],d))
            lis1 = pd.DataFrame(columns=[col])
            lis1[col] = lis
            self.maps = pd.concat([self.maps, lis1],axis=1) 
    def transform(self, W):
        Z = W.copy()
        for col in self.disc:
            for value in Z[col].values:
                Aux = [x for x in self.maps[col] if type(x)==tuple]
                if value in [x[0] for x in Aux]:
                        aux = [x[1] for x in Aux if x[0]==value]
                        Z[col].replace(value,aux[0]*100,inplace=True)
                else:
                    print str(value)+" No se observo en la variable original " + str(col)
        for col in self.cont:
            for pairs in [x for x in self.maps[col] if type(x)==tuple ]:
                for value in Z[col].values:
                    if (pairs[0]<= value) & (value<= pairs[1]):
                        Z[col].replace(value,pairs[2]*100,inplace=True)
        return Z
