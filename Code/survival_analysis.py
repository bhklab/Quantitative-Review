#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 29 2023
Last updated Feb 16 2024

@author: caryn-geady
"""

"""
Models:
- Cox Proportional Hazards
- LASSO-Cox
- Random Survival Forest
"""

import numpy as np
# from lifelines import CoxPHFitter
# from lifelines.utils import concordance_index
from sklearn.utils import resample
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import GridSearchCV 
from lifelines.utils.sklearn_adapter import sklearn_adapter

def CPH_bootstrap(df, name='agg/selection name', outcome='OS', trainFlag=True,param_grid=None):
    '''
    Compute CPH with bootstrapping

    Parameters:    
    - df (DataFrame): selected features + survival data
    - name (str): feature aggregation / lesion selection identifier
    - outcome (str): outcome modelled (default overall survival (OS))
    - trainFlag (bool): flag indicating whether to perform training or testing
    - param_grid (dict): hyperparameter grid for CoxPHFitter (optional)
    
    Returns:
    - best_params (dict): optimal hyperparameters for the model (when trainFlag is True)
    - print (str): C-index (95% confidence interval when trainFlag is True)
    '''

    if trainFlag:
        
        dat = df.copy()
        Y = Surv.from_arrays(dat['E_OS'],dat['T_OS'])
        X = dat.drop(['E_'+outcome, 'T_'+outcome], axis=1)
        
        params   = {
                    'alpha': [1.0, 10.0, 100.0, 1000.0],  
                    'tol': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]     
                    }
        
        clf = GridSearchCV(CoxPHSurvivalAnalysis(), params, cv=5)
        clf.fit(X,Y)

        # configure bootstrap (sampling 50% of data)
        n_iterations = 100
        n_size = int(len(df) * 0.50)

        metrics = []

        for i in range(n_iterations):
            sample = resample(df, n_samples=n_size, random_state=i)
            dat = sample.copy()
            Y = Surv.from_arrays(dat['E_OS'],dat['T_OS'])
            X = dat.drop(['E_'+outcome, 'T_'+outcome], axis=1)

            # calculate c-index and append to list
            cph = CoxPHSurvivalAnalysis(alpha = clf.best_params_['alpha'], tol = clf.best_params_['tol']).fit(X,Y)
            score = cph.score(X,Y)
            metrics.append(score)

        # calculate confidence interval
        alpha = 0.95
        p = ((1.0 - alpha) / 2.0) * 100
        lower = max(0.0, np.percentile(metrics, p))
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        upper = min(1.0, np.percentile(metrics, p))
        med = np.percentile(metrics, 50)

        return clf.best_params_, print(name, 'CPH training: ', '%.3f (%.3f-%.3f)' % (med, lower, upper))

    else:
        
        dat = df.copy()
        Y = Surv.from_arrays(dat['E_OS'],dat['T_OS'])
        X = dat.drop(['E_'+outcome, 'T_'+outcome], axis=1)
        
        cph = CoxPHSurvivalAnalysis(alpha = param_grid[0]['alpha'], tol = param_grid[0]['tol']).fit(X,Y)
        score = cph.score(X,Y)

        return print(name, 'CPH testing: {:.3f}'.format(score))
    

def LASSO_COX_bootstrap(df,name='agg/selection name',outcome='OS',trainFlag=True,param_grid=None):
    
    '''
	Compute Lasso-Cox with bootstrapping

	:param df: (pandas DataFrame) selected features + survival data
    :param name: (str) feature aggregation / lesion selection identifier
    :param outcome: (str) outcome modelled (default overall survival (OS))
	:return: (str) C-index (95% confidence interval)
	'''
    
    if trainFlag:
        
        dat = df.copy()
        Y = Surv.from_arrays(dat['E_OS'],dat['T_OS'])
        X = dat.drop(['E_'+outcome, 'T_'+outcome], axis=1)
                
        params   = {
                    # 'alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'tol': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]     
                    }
        
        clf = GridSearchCV(CoxPHSurvivalAnalysis(), params, cv=5)
        clf.fit(X,Y)
        
        # configure bootstrap (sampling 50% of data)
        n_iterations = 100
        n_size = int(len(df) * 0.50)
        
        metrics = []
        
        for i in range(n_iterations):
            sample = resample(df,n_samples=n_size,random_state=i)#.reset_index(True)
            X = sample.copy().iloc[:,:-2]
            
            X = X.to_numpy()
            y = sample[['E_'+outcome, 'T_'+outcome]].copy()
            y['E_'+outcome] = y['E_'+outcome].astype('bool')
            y = y.to_records(index=False)
            
            # calculate c-index and append to list
            estimator = CoxnetSurvivalAnalysis(l1_ratio = 0.5, tol = clf.best_params_['tol'])
            estimator.fit(X, y)
            score = estimator.score(X, y)
            metrics.append(score)
        
        # calculate confidence interval
        alpha = 0.95
        p = ((1.0 - alpha) / 2.0) * 100
        lower = max(0.0, np.percentile(metrics, p))
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        upper = min(1.0, np.percentile(metrics, p))
        med = np.percentile(metrics, 50)
    
        return clf.best_params_, print(name, 'Lasso-Cox training: ', '%.3f (%.3f-%.3f)' % (med, lower, upper))
	
    else:
        X = df.copy().iloc[:,:-2]
        
        X = X.to_numpy()
        y = df[['E_'+outcome, 'T_'+outcome]].copy()
        y['E_'+outcome] = y['E_'+outcome].astype('bool')
        y = y.to_records(index=False)
        
        # calculate c-index and append to list
        estimator = CoxnetSurvivalAnalysis(l1_ratio = 0.5, tol = param_grid[0]['tol'])
        estimator.fit(X, y)
        score = estimator.score(X, y)
        
        return print(name, 'Lasso-Cox testing: {:.3f}'.format(score))
        

def RSF_bootstrap(df,name='agg/selection name',outcome='OS',trainFlag=True):
    
    '''
	Compute RSF with bootstrapping

	:param df: (pandas DataFrame) selected features + survival data
    :param name: (str) feature aggregation / lesion selection identifier
    :param outcome: (str) outcome modelled (default overall survival (OS))
	:return: (str) C-index (95% confidence interval)
	'''
    
    # parameters
    NUMESTIMATORS = 100
    TESTSIZE = 0.20
    random_state = 20
    
    if trainFlag:
    
        # configure bootstrap (sampling 50% of data)
        n_iterations = 100
        n_size = int(len(df) * 0.50)
        
        metrics = []
    
        for i in range(n_iterations):
            sample = resample(df,n_samples=n_size,random_state=i)
            X = sample.copy().iloc[:,:-2]
            
            X = X.to_numpy().astype('float64')
            y = sample[['E_'+outcome, 'T_'+outcome]].copy()
            y['E_'+outcome] = y['E_'+outcome].astype('bool')
            y['T_'+outcome] = y['T_'+outcome].astype('float64')
            y = y.to_records(index=False)
        
            # calculate c-index and append to list
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TESTSIZE, random_state=random_state)
            rsf = RandomSurvivalForest(n_estimators=NUMESTIMATORS,
                                   min_samples_split=15,
                                   min_samples_leaf=8,
                                   max_features="sqrt",
                                   n_jobs=-1,
                                   random_state=random_state)
            rsf.fit(X_train, y_train)
            score = rsf.score(X_test, y_test)
            metrics.append(score)
        
        # calculate confidence interval
        alpha = 0.95
        p = ((1.0 - alpha) / 2.0) * 100
        lower = max(0.0, np.percentile(metrics, p))
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        upper = min(1.0, np.percentile(metrics, p))
        med = np.percentile(metrics, 50)
    
        return print(name, 'RSF training: ', '%.3f (%.3f-%.3f)' % (med, lower, upper))
    
    else:
        X = df.copy().iloc[:,:-2]
        
        X = X.to_numpy().astype('float64')
        y = df[['E_'+outcome, 'T_'+outcome]].copy()
        y['E_'+outcome] = y['E_'+outcome].astype('bool')
        y['T_'+outcome] = y['T_'+outcome].astype('float64')
        y = y.to_records(index=False)
    
        # calculate c-index and append to list
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TESTSIZE, random_state=random_state)
        rsf = RandomSurvivalForest(n_estimators=NUMESTIMATORS,
                               min_samples_split=15,
                               min_samples_leaf=8,
                               max_features="sqrt",
                               n_jobs=-1,
                               random_state=random_state)
        rsf.fit(X_train, y_train)
        score = rsf.score(X_test, y_test)
        
        return print(name, 'RSF testing: {:.3f}'.format(score))