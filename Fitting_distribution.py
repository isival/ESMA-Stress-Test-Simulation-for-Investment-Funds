"""
Created by Isidore Valette
15/10/2019

Prends le set de donnée et trouve la loi qui fit le plus les données
Boucle sur toutes les distributions possibles, et prends celle dont l'erreur SSE est la plus faible
"""
###################### Imports

import scipy
from scipy import *
import scipy.stats as st

import numpy as np
from numpy import *

import matplotlib
from matplotlib import *
import matplotlib.pylab as plt

from lmfit.models import *

import pandas as pd
from pandas import *

import statistics
from statistics import *
from fractions import *
from math import *

import warnings

import statsmodels as sm

######################## Traitement des données 


data = np.loadtxt('test.dat') #mets toutes le données sous forme d'un tableau

dates = []
for i in range(len(data)):
    dates.append(data[i, 0])
dates = np.asarray(dates)

fund_tot_asset = [data[i, 1] for i in range(len(data))]
fund_tot_asset = np.asarray(fund_tot_asset)

nav = [data[i, 2] for i in range(len(data))]
nav = np.asarray(nav)

#récupère les données sur un interval de dates précis
ind = []
for i in range(0 ,len(data)) :
    if data[i][0] < 20140705 or data[i][0] > 20180705 : #date à modifier
        ind.append(i)

#On enlève les dates indésirables
dates = np.delete(np.asarray(dates), ind)
fund_tot_asset = np.delete(np.asarray(fund_tot_asset), ind)
nav = np.delete(np.asarray(nav), ind)

######################### Calcul des rendements
########### NAV ##
r_nav=[None]*len(nav)
for l in range(0, len(nav)-1) :
    r_nav[l] = (nav[l+1]/nav[l])
r_nav = np.flip(r_nav)
r_nav = r_nav[1:]
########### Fund total asset - r_fta ##
r_fta=[None]*len(fund_tot_asset)
for m in range(0, len(fund_tot_asset)-1) :
    r_fta[m] = (fund_tot_asset[m+1]/fund_tot_asset[m])
r_fta = np.flip(r_fta)
r_fta = r_fta[1:]
########### rachat souscription ##
r_rachat_sous=[None]*len(r_nav)
for k in range(994) :
    r_rachat_sous[k] = r_fta[k]-r_nav[k]
r_rachat_sous = np.flip(r_rachat_sous)
## rachat (valeurs de r_rachat_sous qui sont négatives) ##
## souscription (valeurs de r_rachat_sous qui sont positives) ##

r_rachat=[]
r_souscription=[]

for elmt in r_rachat_sous :
    if elmt < 0 :
        r_rachat.append(elmt)
    if elmt > 0 :
        r_souscription.append(elmt)

#########################

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')


# Creéation du modèle depuis les données
def best_fit_distribution(data, bins=200, ax=None):

    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions
    DISTRIBUTIONS = [        
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma, st.dweibull,st.erlang,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.genlogistic,st.gennorm,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,
        st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,
        st.invweibull,st.laplace,st.levy, st.t,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.ncx2,st.ncf,st.nct,
        st.norm,st.powerlaw,st.powerlognorm,st.powernorm,
		st.rayleigh,st.recipinvgauss,st.semicircular,st.truncnorm,
		st.uniform,st.weibull_min,st.weibull_max
        ]

    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimation des paramètres de distribution^depuis les données
    for distribution in DISTRIBUTIONS:

        #on fit la distribution
        try:
            #warnins
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                #on fit la distribution aux données
                params = distribution.fit(data)

                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # calcul le pdf fitted et l'erreur
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                #identifie la meilleure distribution
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse
                    print("error : ", best_sse)

        except Exception:
            pass
    print(best_params)
    return (best_distribution.name, best_params)
    

def make_pdf(dist, params, size=10000):
    """Génère la fonction de distribution de probabilité"""

    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # récupère le début et la fin, des points de la distribution, en fonction du passage d'un argument
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Crée le pdf, puis on transforme en série pandas
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


# remplace par rendement r-s ou rend rachat ou rend souscr.
data = r_rachat_sous

# Comparaison
plt.figure(figsize=(10,7))
ax = plt.hist(data, bins=50, density=True, alpha=0.5)
#plt.show()

# On trouve la meilleure distribution qui fit
best_fit_name, best_fit_params = best_fit_distribution(data, 2000, ax)
best_dist = getattr(st, best_fit_name)


# On crée le pdf avec ces meilleurs paramètres
pdf = make_pdf(best_dist, best_fit_params)
plt.plot(pdf)
plt.show()

p = np.asarray(pdf)
p = [p[10*i] for i in range(994)]
diff = p - data

######################## DURBIN WATSON
######################## Traitement des données 

element={"Rendement Rachat Souscription" : r_rachat_sous,
            "Rendement NAV" : r_nav,
            "Rendement FTA" : r_fta,
            "Rendement Rachat" : r_rachat,
            "Rendement Souscription" : r_souscription}

for key, value in element.items() :
    test = sm.stats.stattools.durbin_watson(diff)
    print(key, " :  ", test)

########################

# On visualise
plt.figure(figsize=(10,7))
ax = pdf.plot(lw=2, label='PDF', legend=True)
plt.hist(data, bins=500, density=True, alpha=0.5, label='Data')

param_names = (best_dist.shapes + ', loc (moyenne) , scale (stdeviation) ').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
dist_str = '{}({})'.format(best_fit_name, param_str)

print(dist_str)

ax.set(xlim=(-3, 3))
plt.show()