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

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.utils import resample
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest


def CPH_bootstrap(df,name='agg/selection name',outcome='OS',trainFlag=True):
    
    '''
	Compute CPH with bootstrapping

	:param df: (pandas DataFrame) selected features + survival data
    :param name: (str) feature aggregation / lesion selection identifier
    :param outcome: (str) outcome modelled (default overall survival (OS))
	:return: (str) C-index (95% confidence interval)
	'''
    
    if trainFlag:
        # configure bootstrap (sampling 50% of data)
        n_iterations = 100
        n_size = int(len(df) * 0.50)
    
        metrics = []
    
        for i in range(n_iterations):
            sample = resample(df,n_samples=n_size,random_state=i)#.reset_index(True)
            
            # calculate c-index and append to list
            cph = CoxPHFitter(penalizer=0.0001).fit(sample, 'T_'+outcome, 'E_'+outcome)
            score = concordance_index(sample['T_'+outcome], -cph.predict_partial_hazard(sample), sample['E_'+outcome])
            metrics.append(score)
        
        # calculate confidence interval
        alpha = 0.95
        p = ((1.0 - alpha) / 2.0) * 100
        lower = max(0.0, np.percentile(metrics, p))
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        upper = min(1.0, np.percentile(metrics, p))
        med = np.percentile(metrics, 50)
        
        return print(name, 'CPH training: ', '%.3f (%.3f-%.3f)' % (med, lower, upper))
    
    else:
        cph = CoxPHFitter().fit(df, 'T_'+outcome, 'E_'+outcome)
        score = concordance_index(df['T_'+outcome], -cph.predict_partial_hazard(df), df['E_'+outcome])
        
        return print(name, 'CPH testing: {:.3f}'.format(score))
    

def LASSO_COX_bootstrap(df,name='agg/selection name',outcome='OS',trainFlag=True):
    
    '''
	Compute Lasso-Cox with bootstrapping

	:param df: (pandas DataFrame) selected features + survival data
    :param name: (str) feature aggregation / lesion selection identifier
    :param outcome: (str) outcome modelled (default overall survival (OS))
	:return: (str) C-index (95% confidence interval)
	'''
    
    if trainFlag:
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
            estimator = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=None)
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
    
        return print(name, 'Lasso-Cox training: ', '%.3f (%.3f-%.3f)' % (med, lower, upper))
	
    else:
        X = df.copy().iloc[:,:-2]
        
        X = X.to_numpy()
        y = df[['E_'+outcome, 'T_'+outcome]].copy()
        y['E_'+outcome] = y['E_'+outcome].astype('bool')
        y = y.to_records(index=False)
        
        # calculate c-index and append to list
        estimator = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=None)
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