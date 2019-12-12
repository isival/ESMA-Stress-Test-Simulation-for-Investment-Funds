"""
Created by Isidore Valette
16/10/2019

Stats sur tous les rendements des sets de données, avec traitement des outliers
Autocorrélation pour la stationnarité
Ljungi Box pour le bruit blanc
Etude de la tendance
"""
###################### Imports

import scipy
from scipy import *
import scipy.stats as st

import numpy as np
from numpy import *

import matplotlib.pylab as plt
from matplotlib import *
import matplotlib

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

print("")
print("pourcentage de rachat : ", double(100 * len(r_rachat)/(len(r_rachat) + len(r_souscription))), "%")
print("pourcentage de souscription : ", double(100 *len(r_souscription)/(len(r_rachat) + len(r_souscription))), "%")

print("sous : ", mean(r_souscription))
print("rac : ", mean(r_rachat))
print("r_rachat_sous : ", mean(r_rachat_sous))
print("fta : ", mean(r_fta))
print("nav : ", mean(r_nav))

##################### outliers --> enlève les valeurs sup à la abs de la MAD

def mav(data) :
	df = pd.DataFrame(data)
	a = df.mad()[0]

	data = np.asarray(data)
	index = []
	
	for i in range(len(data)) :
		if abs(data[i]) > a :
			index.append(i)

	data = np.delete(np.asarray(data), index)
	return a, data

r =len(r_fta)
s = len(r_nav)

#enlève les outliers - à vérifier car pas bon
#r_nav = mav(r_nav)[1]
#r_fta = mav(r_fta)[1]

#récupère la mav
mav_rach = mav(r_rachat)[0]
mav_sous = mav(r_souscription)[0]

print("")
print("Outliers for fta : ", r-len(r_fta))
print("Outliers for nav : ", s-len(r_nav))
print("")
print("MAD rachat : ", mav_rach)
print("MAD souscription : ", mav_sous)
print("")
###################### fitting des modèles

from scipy.stats import norm,rayleigh, gamma
from statsmodels.graphics.tsaplots import plot_acf

param = rayleigh.fit(r_rachat_sous) # distribution fitting
param2 = rayleigh.fit(r_rachat_sous)
param3 = gamma.fit(r_rachat_sous)

x = linspace(-5, 5,1000)
a=1.99 #paramètre loi gamma

pdf_fitted = rayleigh.pdf(x,loc=param[0],scale=param[1])
pdf_fitted_2= norm.pdf(x,loc=param2[0],scale=param2[1])
pdf_fitted_3= gamma.pdf(x, a)

pdf = rayleigh.pdf(x,loc=5,scale=2)
pdf2 = norm.pdf(x,loc=5,scale=2)
pdf3 = gamma.pdf(x,a)

plt.title('Rayleigh, normal, gamma distribution')
#plt.hist(r_rachat_sous,alpha=.3)
#plt.plot(x,pdf_fitted,'r-', x, pdf2, 'b-', x, pdf3, 'g-')

axes = plt.gca()
axes.set_xlim(-2, 2)

############################## statistiques sur toutes les données

element={"Rendement Rachat Souscription" : r_rachat_sous,
			"Rendement NAV" : r_nav,
			"Rendement FTA" : r_fta,
			"Rendement Rachat" : r_rachat,
			"Rendement Souscription" : r_souscription}
"""
for key, value in element.items() :
	print("")
	print(key)
	print("")
	print("mean : ", mean(value))
	print("median : ", median(value))
	print("pvariance : ", pvariance(value))
	print("stdev : ", stdev(value))
	print("min : ",min(value))
	print("max : ",max(value))
	print("")
"""
############################## autocorrélation
from statsmodels.tsa.stattools import pacf

for key, value in element.items() :
	plot_acf(value, lags=50)
	pcor = pacf(value, nlags = 50)
	plt.plot(pcor)
	#plt.show()

################################# tendance (sert pas)
from statsmodels.api import OLS
from statsmodels.graphics.regressionplots import abline_plot

X = np.ones((len(r_rachat_sous), 2))
X[:,1] = np.arange(0,len(r_rachat_sous))
reg = OLS(r_rachat_sous,X)
results = reg.fit()
results.params

fig = abline_plot(model_results=results)
ax = fig.axes[0]
ax.plot(X[:,1], r_rachat_sous, 'r')
ax.margins(.1)
#plt.show()

############## test de Ljung Box - Bruit blanc

import statsmodels as sm
from statsmodels import *

#on enlève rachat et souscription qui ne sont pas stationaires

for key, value in element.items() :
	try :
		res = sm.tsa.arima_model.ARMA(value, (1,1)).fit(disp=-1)
		test = sm.stats.diagnostic.acorr_ljungbox(res.resid, lags = [5])
		#print("(Ljung Box, pvalue Khi2) - ", key, " : ", test)
	except ValueError:
		print(key, " est non stationnaire")

##################### test autocorrélation - dickey fuller
#ok if pvalue is inferior to 0.05

import statsmodels.formula.api as smt
print("")
for key, value in element.items() :
	try :
		test = sm.tsa.stattools.adfuller(value)
		print(key, " :  pvalue =", test[1]) #pvalue
		print(key, " :  lag =", test[2]) #lags
	except ValueError:
		print("error_df")


#################### test égalité des variances

data1 = r_rachat_sous[:int(len(r_rachat_sous)/3)]
data2 = r_rachat_sous[int(2*len(r_rachat_sous)/3):]


##############################################################################################################

############### test autocorrelation - durbin watson
######################## Traitement des données 

data = np.loadtxt('test.dat') #mets toutes le données sous forme d'un tableau

dates = [data[i,0] for i in range(len(data))]
dates = np.asarray(dates)

fund_tot_asset = [data[i, 1] for i in range(len(data))]
fund_tot_asset = np.asarray(fund_tot_asset)

nav = [data[i, 2] for i in range(len(data))]
nav = np.asarray(nav)

#récupère les dnnées sur un interval de dates précis
ind = []
for i in range(0 ,len(data)) :
	if data[i][0] < 20140705 or data[i][0] > 20180705 : #date à modifier
		ind.append(i)

#On enlève les dates indésirables
dates = np.delete(np.asarray(dates), ind)
fund_tot_asset = np.delete(np.asarray(fund_tot_asset), ind)
nav = np.delete(np.asarray(nav), ind)

######################### Calcul des rendements

## rachat souscription ##

r_rachat_sous=[None]*len(nav)
for k in range(1, len(nav)) :
	r_rachat_sous[k] = ((fund_tot_asset[k]/fund_tot_asset[k-1]) - (nav[k]/nav[k-1]))*100
r_rachat_sous[0] = 0
r_rachat_sous = np.flip(r_rachat_sous)

r_rachat=[]
r_souscription=[]

for elmt in r_rachat_sous :
	if elmt < 0 :
		r_rachat.append(-elmt)
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

r_rachat = np.asarray(r_rachat)
r_souscription = np.asarray(r_souscription)

Data = {"Rachat - Souscription" : r_rachat_sous,
		"NAV" : r_nav,
		"FTA" : r_fta,
		"Rachats" : r_rachat,
		"Souscription" : r_souscription}

for key, elmt in Data.items() :
	data = elmt

	# On trouve la meilleure distribution qui fit
	best_fit_name, best_fit_params = best_fit_distribution(data, 2000, ax)
	best_dist = getattr(st, best_fit_name)


	# On crée le pdf avec ces meilleurs paramètres
	pdf = make_pdf(best_dist, best_fit_params)

	p = np.asarray(pdf)
	p = [p[10*i] for i in range(len(data))]
	diff = p - data

	test = sm.stats.stattools.durbin_watson(diff)
	print("test durbin watson :  ", key, " ", test)
	print("")

##################
