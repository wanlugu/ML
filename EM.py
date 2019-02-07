%load_ext watermark
import warnings
warnings.filterwarnings("ignore") 
from IPython.core.display import display, HTML

import time
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import scipy.stats as scs
from scipy.stats import multivariate_normal as mvn
import sklearn.mixture as mix

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

%watermark

# import class heights
f = 'https://raw.githubusercontent.com/BlackArbsCEO/Mixture_Models/K-Means%2C-E-M%2C-Mixture-Models/Class_heights.csv'

data = pd.read_csv(f)
# data.info()

height = data['Height (in)']
data


# EM
def em_gmm_orig(xs,pis,mus,sigmas, tol = 0.01, max_iter = 100):
    n, p = xs.shape
    k = len(pis)

    ll_old = 0
    for i in range(max_iter):
        print("\nIteration:", i)
        print()
        ll_new =0

        # E-step
        ws = np.zeros((k,n)) # indicators
        for j in range(len(mus)):
            for i in range(n):
                ws[j,i] = pis[j]*mvn(mus[j],sigmas[j])*pdf(xs[i])

        ws /= ws.sum(0) # column sum

        # M-step
        pis = np.zeros(k)
        for j in range(len(mus)):
            for i in range(n):
                pis[j] += ws[j,i]
            pis[j] /= n

        mus = np.zeros(k,p)
        for j in range(len(mus)):
            for i in range(n):
                mus[j] = ws[j,i]*xs[i]
        mus /= ws.sum(1)

        sigmas = np.zeros((k,p,p))
        for j in range(len(mus)):
            for i in range(n):
                ys = np.shape((xs[i]-mus[i]),(p,1))
                sigma[j] = ws[j,i]* np.dot(ys,ys.T)
        sigma /= ws.sum(1)

        # plots of fitted distribution of two clusters
        new_mus = (np.diag(mus)[0], np.diag(mus)[1])
        xx = np.linspace(0,100,100)
        yy = scs.multivariate_normal.pdf(xx, mean = new_mus[0], cov = new_sigs[0])

        colors = sns.color_palette('Dark2',3)
        fig, ax = plt.subplots(figsize = (9,7))
        ax.set_ylim(-0.001, np.max(yy))
        ax.plot(xx, yy, color = colors[1])
        ax.axvline(new_mus[0], ymin = 0., color = volors[1])
        lo, hi = ax.get_ylim()
        ax.annotate(f'$\mu_1$: {new_mus[0]:3.2f})',
                    fontsize = 12, fontweight = 'demi',
                    xy =(new_mus[0], (hi-lo)/2),
                    
    return 
    




            
