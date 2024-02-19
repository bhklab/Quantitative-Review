#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 29 2023
Last updated Jan 4 2024

@author: cgeady
"""

'''
Lesion Selection Methods:

    - Primary tumor (if available)
    - Primary tumor (if available) + number of metastases
    - Smallest lesion
    - Smallest lesion + number of metastases
    - Largest lesion
    - Largest lesion + number of metastases
    - Largest lung metastasis
    - Largest lung metastasis + number of metastases
    

Lesion-Level Aggregation Methods:
    
    - Unweighted Average
    - Volume-Weighted Average
    - Volume-Weighted Average of 3 Largest lesions
    - Concatenation (choose 2)
    - Cosine Similarity Metrics (X2)
    - Inter-site similarity matrix (??? to be figured out ???)
    - Unsupervised clustering (??? to be figured out ???)

'''

# %% IMPORTS
# Change working directory to be whatever directory this script is in
#import os
#os.chdir(os.path.dirname(__file__))
import numpy as np, pandas as pd
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from pymrmre import mrmr
#from lifelines import CoxPHFitter
#from lifelines.utils import concordance_index
from sklearn.utils import resample
#from sksurv.linear_model import CoxnetSurvivalAnalysis
#from sksurv.ensemble import RandomSurvivalForest

import Code.lesion_selection as fs



# %% MISC.

def calcNumMets(radiomics):
    
    ids,counts = np.unique(radiomics.USUBJID,return_counts=True)
    numMets = pd.DataFrame([ids,counts]).T
    numMets.columns = ['USUBJID','NumMets']
    
    return numMets

def printRadiomicsReport(radiomics):
    
    patients,numLesions = np.unique(radiomics.USUBJID,return_counts=True)
    instCodes = [p[-6:-3] for p in patients]
    institutions,numPatients = np.unique(instCodes,return_counts=True)
    
    if 'RADCURE' in radiomics.USUBJID[0]:  # update for CRLM dataset
        
        print('----------')
        print('No. of Institutions: 1')
        
    else:
    
        print('----------')
        print('No. Institutions: {}'.format(len(institutions)))
        print('----------')
        print('Patients per Institution: ')
        print('Mean: {:.2f}'.format(np.mean(numPatients)))
        print('Median: {}'.format(np.median(numPatients)))
        print('Range: [{}, {}]'.format(np.min(numPatients),np.max(numPatients)))
        print('IQR: [{}, {}]'.format(np.percentile(numPatients,25),np.percentile(numPatients,75)))
    
    if len(radiomics) > len(patients):
        
        print('----------')
        print('Lesions per Patient: ')
        print('Mean: {:.2f}'.format(np.mean(numLesions)))
        print('Median: {}'.format(np.median(numLesions)))
        print('Range: [{}, {}]'.format(np.min(numLesions),np.max(numLesions)))
        print('IQR: [{}, {}]'.format(np.percentile(numLesions,25),np.percentile(numLesions,75)))
        
def randomSplit(radiomics,clinical,train_size = 0.7,tuneFlag = True):
     
    test_size = 1 - train_size
    ids_to_keep = np.intersect1d(clinical.USUBJID, np.unique(radiomics.USUBJID))
    train_ids,test_ids = train_test_split(ids_to_keep,test_size=test_size,random_state=42)
    
    if tuneFlag:
        tune_size = 0.5
        tune_ids,test_ids = train_test_split(ids_to_keep,test_size=tune_size,random_state=42)
        return train_ids, tune_ids, test_ids
    else:
        return train_ids, test_ids
    
def singleInstValidationSplit(radiomics,clinical,train_size = 0.7):
    
    test_size = 1 - train_size
    ids_to_keep = np.intersect1d(clinical.USUBJID, np.unique(radiomics.USUBJID))
    
    instCodes = np.array([p[-6:-3] for p in ids_to_keep])

    institutions,numPatients = np.unique(instCodes,return_counts=True)  
    largestSingleInst = institutions[np.where(numPatients==np.max(numPatients))]

    validation_ids = ids_to_keep[instCodes==largestSingleInst]

    train_ids,test_ids = train_test_split(ids_to_keep[instCodes!=largestSingleInst],test_size=test_size,random_state=42)
    
    return train_ids, test_ids, validation_ids

      

# %% LESION AGGREGATION METHODS

def calcUnweightedAverage(radiomics,clinical,outcome='OS',scaleFlag=False,numMetsFlag=True,multipleFlag=True):
    
    startColInd = np.where(radiomics.columns.str.find('original')==0)[0][0]
    
    df_UnweightedAverage = radiomics.copy().iloc[:,startColInd:]
    df_UnweightedAverage.insert(0, "USUBJID", radiomics.USUBJID, True)
    df_Sum = df_UnweightedAverage.loc[:,df_UnweightedAverage.columns.str.contains('original_shape')].copy()
    df_Sum.insert(0,"USUBJID",radiomics.USUBJID,True)
    df_Sum = df_Sum.groupby('USUBJID').sum().reset_index(drop=True)
    df_UnweightedAverage = df_UnweightedAverage.groupby('USUBJID').mean().reset_index(drop=True)
    df_UnweightedAverage.loc[:,df_UnweightedAverage.columns.str.contains('original_shape')] = df_Sum
    df_UnweightedAverage.insert(0,"USUBJID",np.unique(radiomics.USUBJID),True)
    
    df_UnweightedAverage = df_UnweightedAverage.merge(clinical[['USUBJID','T_'+outcome,'E_'+outcome]],on='USUBJID').reset_index(drop=True)
    
    if scaleFlag:
        scaledFeatures = StandardScaler().fit_transform(df_UnweightedAverage.iloc[:,1:-2])
        df_UnweightedAverage.iloc[:,1:-2] = scaledFeatures
    
    if numMetsFlag:
        df_UnweightedAverage = calcNumMets(radiomics).merge(df_UnweightedAverage,on='USUBJID').reset_index(drop=True)
        
        if multipleFlag:
            df_UnweightedAverage = df_UnweightedAverage.loc[df_UnweightedAverage.NumMets>1,:]
        
    return df_UnweightedAverage

def calcVolumeWeightedAverage(radiomics,clinical,outcome='OS',scaleFlag=False,numMetsFlag=True,multipleFlag=True):
    
    startColInd = np.where(radiomics.columns.str.find('original')==0)[0][0]
    
    df_Volumes = radiomics[['USUBJID','original_shape_VoxelVolume']].copy()
    totalvol = df_Volumes.copy().groupby('USUBJID').sum()
    totalvoldict = pd.Series(totalvol['original_shape_VoxelVolume'].values,index=totalvol.index).to_dict()
    df_Volumes['total volume'] = df_Volumes.USUBJID.map(totalvoldict)
    weight = df_Volumes['original_shape_VoxelVolume'] / df_Volumes['total volume']
    
    df_WeightedAverage = radiomics.copy().iloc[:,startColInd:]
    df_WeightedAverage.loc[:,~df_WeightedAverage.columns.str.contains('original_shape')] = df_WeightedAverage.loc[:,~df_WeightedAverage.columns.str.contains('original_shape')].multiply(weight.values,axis='index')
    df_WeightedAverage.insert(0, "USUBJID", radiomics.USUBJID, True)
    df_WeightedAverage = df_WeightedAverage.groupby('USUBJID').sum().reset_index(drop=True)
    df_WeightedAverage.insert(0,"USUBJID",np.unique(radiomics.USUBJID),True)
    
    df_WeightedAverage = df_WeightedAverage.merge(clinical[['USUBJID','T_'+outcome,'E_'+outcome]],on='USUBJID').reset_index(drop=True)
    
    if scaleFlag:
        scaledFeatures = StandardScaler().fit_transform(df_WeightedAverage.iloc[:,1:-2])
        df_WeightedAverage.iloc[:,1:-2] = scaledFeatures
    
    if numMetsFlag:
        df_WeightedAverage = calcNumMets(radiomics).merge(df_WeightedAverage,on='USUBJID').reset_index(drop=True)
        
        if multipleFlag:
            df_WeightedAverage = df_WeightedAverage.loc[df_WeightedAverage.NumMets>1,:]
        
    return df_WeightedAverage

def calcVolumeWeightedAverageNLargest(radiomics,clinical,numLesions=3,outcome='OS',scaleFlag=False,numMetsFlag=True):
    
    id_counts = radiomics['USUBJID'].value_counts()
    valid_ids = id_counts[id_counts >= numLesions].index
    df_radiomics = radiomics[radiomics['USUBJID'].isin(valid_ids)]
    
    df_filtered = df_radiomics.groupby('USUBJID').apply(lambda group: group.nlargest(numLesions, 'original_shape_VoxelVolume')).reset_index(drop=True)
    
    df_VolWeightNLargest = calcVolumeWeightedAverage(df_filtered,clinical,numMetsFlag=False)
    
    if scaleFlag:
        scaledFeatures = StandardScaler().fit_transform(df_VolWeightNLargest.iloc[:,1:-2])
        df_VolWeightNLargest.iloc[:,1:-2] = scaledFeatures
    
    if numMetsFlag:
        df_VolWeightNLargest.insert(1,"NumMets",calcNumMets(df_radiomics).NumMets,True)
        
    return df_VolWeightNLargest

def concatenateNLargest(radiomics,clinical,numLesions=3,outcome='OS',scaleFlag=False,numMetsFlag=True):
    
    startColInd = np.where(radiomics.columns.str.find('original')==0)[0][0]
    
    id_counts = radiomics['USUBJID'].value_counts()
    valid_ids = id_counts[id_counts >= numLesions].index
    df_radiomics = radiomics[radiomics['USUBJID'].isin(valid_ids)]
    
    df_radiomics = df_radiomics.iloc[:,startColInd:]
    df_radiomics.insert(0, "USUBJID", radiomics.USUBJID[radiomics['USUBJID'].isin(valid_ids)], True)
    
    df_filtered = df_radiomics.groupby('USUBJID').apply(lambda group: group.nlargest(numLesions, 'original_shape_VoxelVolume')).reset_index(drop=True)
    df_Concatenated = df_filtered[df_filtered.groupby('USUBJID').cumcount() == 0].reset_index(drop=True)
    total_volume = df_Concatenated.original_shape_VoxelVolume.copy()
    df_Concatenated = df_Concatenated.rename(columns={c: c+'_Lesion1' for c in df_Concatenated.columns if c not in ['USUBJID']})
    
    for i in range(1,numLesions):
        df_temp = df_filtered[df_filtered.groupby('USUBJID').cumcount() == i].reset_index(drop=True)
        total_volume += df_temp.original_shape_VoxelVolume.copy()
        df_temp = df_temp.rename(columns={c: c+'_Lesion'+str(i+1) for c in df_temp.columns if c not in ['USUBJID']})
        df_Concatenated = df_Concatenated.merge(df_temp,on='USUBJID')
    
    # here, original_shape_VoxelVolume represents total volume for the N largest lesions
    # then, original_shape_VoxelVolume_Lesion1 represents the volume for Lesion1 and so on...
    df_Concatenated.insert(1,"original_shape_VoxelVolume",total_volume,True)    
    df_Concatenated = df_Concatenated.merge(clinical[['USUBJID','T_'+outcome,'E_'+outcome]],on='USUBJID').reset_index(drop=True)
    
    if scaleFlag:
        scaledFeatures = StandardScaler().fit_transform(df_Concatenated.iloc[:,1:-2])
        df_Concatenated.iloc[:,1:-2] = scaledFeatures
    
    if numMetsFlag:
        df_Concatenated.insert(1,"NumMets",calcNumMets(df_radiomics).NumMets,True)
    
    return df_Concatenated

def calcCosineMetrics(radiomics,clinical,numLesions=2,outcome='OS',scaleFlag=False,numMetsFlag=True):
        
    startColInd = np.where(radiomics.columns.str.find('original')==0)[0][0]
    
    id_counts = radiomics['USUBJID'].value_counts()
    valid_ids = id_counts[id_counts >= numLesions].index
    df_radiomics = radiomics[radiomics['USUBJID'].isin(valid_ids)]
    
    df_radiomics = df_radiomics.iloc[:,startColInd:]
    df_radiomics.insert(0, "USUBJID", radiomics.USUBJID[radiomics['USUBJID'].isin(valid_ids)], True)

    
    # df_radiomics contains only those patients with 2+ lesions
    # for each patient, we need to isolate the features and scale them
    # then for each lesion-lesion combination, we calculate cos_sim using the scaled features
    avgTumorHetero = []
    maxTumorDiverg = []
    df_scale = df_radiomics.copy()

    for p in np.unique(df_radiomics.USUBJID):
        
        inds = np.where(df_radiomics.USUBJID == p)[0]
        scaledFeatures = StandardScaler().fit_transform(df_radiomics.iloc[inds,1:])
        df_scale.iloc[inds,1:] = scaledFeatures
        combos = list(combinations(inds,2))
        cos_dissim = np.zeros((len(combos),))
        
        for i in range(len(combos)):
            
            cos_dissim[i] = 1 - cos_sim([df_scale.iloc[combos[i][0],1:],df_scale.iloc[combos[i][1],1:]])[0][1]
            
        avgTumorHetero.append(np.mean(cos_dissim))
        maxTumorDiverg.append(np.max(cos_dissim))
        
    df_CosineMetrics = pd.DataFrame([np.unique(df_radiomics.USUBJID),avgTumorHetero,maxTumorDiverg]).T
    df_CosineMetrics.columns = ["USUBJID","AVGTHETERO","MAXTDIVERG"]    
    df_CosineMetrics = df_CosineMetrics.merge(clinical[['USUBJID','T_'+outcome,'E_'+outcome]],on='USUBJID').reset_index(drop=True)
    
    if scaleFlag:
        scaledFeatures = StandardScaler().fit_transform(df_CosineMetrics.iloc[:,1:-2])
        df_CosineMetrics.iloc[:,1:-2] = scaledFeatures
    
    if numMetsFlag:
        df_CosineMetrics.insert(1,"NumMets",calcNumMets(df_radiomics).NumMets,True)
        
    return df_CosineMetrics
    


# %% FEATURE REDUCTION/SELECTION

def varianceFilter(radiomics,varThresh=10,outcome='OS',returnColsFlag=True):
    
    var = radiomics.var()
    cols_to_drop = radiomics.columns[np.where(var>=varThresh)]
    
    if cols_to_drop.isin(['original_shape_VoxelVolume']).any():
        cols_to_drop = cols_to_drop.drop('original_shape_VoxelVolume')
    
    if cols_to_drop.isin(['T_'+outcome]).any():
        cols_to_drop = cols_to_drop.drop('T_'+outcome)
        
    if cols_to_drop.isin(['E_'+outcome]).any():
        cols_to_drop = cols_to_drop.drop('E_'+outcome)
      
    if returnColsFlag:
        return cols_to_drop,radiomics.drop(cols_to_drop,axis=1)
    else:
        return radiomics.drop(cols_to_drop,axis=1)
    

def volumeFilter(radiomics,volThresh=0.1,outcome='OS',returnColsFlag=True):

    cor = radiomics.corr(method = 'spearman')['original_shape_VoxelVolume']
    cols_to_drop = cor[abs(cor)>volThresh].index
    
    if cols_to_drop.isin(['original_shape_VoxelVolume']).any():
        cols_to_drop = cols_to_drop.drop('original_shape_VoxelVolume')
    
    if cols_to_drop.isin(['T_'+outcome]).any():
        cols_to_drop = cols_to_drop.drop('T_'+outcome)
        
    if cols_to_drop.isin(['E_'+outcome]).any():
        cols_to_drop = cols_to_drop.drop('E_'+outcome)
        
    if returnColsFlag:
        return cols_to_drop,radiomics.drop(cols_to_drop,axis=1)
    else:
        return radiomics.drop(cols_to_drop,axis=1)

def featureReduction(radiomics,varThresh=10,volThresh=0.1,outcome='OS',returnColsFlag=False,scaleFlag=False):

    df_varReduced = varianceFilter(radiomics,varThresh,outcome,returnColsFlag)
    df_volReduced = volumeFilter(df_varReduced,volThresh,outcome,returnColsFlag)
    
    if scaleFlag:
        scaledFeatures = StandardScaler().fit_transform(df_volReduced.iloc[:,1:-2])
        df_volReduced.iloc[:,1:-2] = scaledFeatures

    return df_volReduced    

def featureSelection(df,numFeatures=10,numMetsFlag=False,volFlag=False,scaleFlag=False):
    
    if numMetsFlag:
        numFeatures -= 1
        solutions = mrmr.mrmr_ensemble_survival(features=df.iloc[:,2:-2],
                                                targets=df.iloc[:,-2:],
                                                solution_length=numFeatures)[0]
        df_Selected = df.copy()[solutions[0]+["NumMets","T_OS","E_OS"]]
        
    else:
        solutions = mrmr.mrmr_ensemble_survival(features=df.iloc[:,1:-2],
                                            targets=df.iloc[:,-2:],
                                            solution_length=numFeatures)[0]
        df_Selected = df.copy()[solutions[0]+["T_OS","E_OS"]]
      
    if volFlag and df.columns.isin(['original_shape_VoxelVolume']).any():
        numFeatures -= 1
        solutions = mrmr.mrmr_ensemble_survival(features=df.iloc[:,2:-2],
                                                targets=df.iloc[:,-2:],
                                                solution_length=numFeatures)[0]
        df_Selected = df.copy()[solutions[0]+["original_shape_VoxelVolume","T_OS","E_OS"]]
        
    else:
        solutions = mrmr.mrmr_ensemble_survival(features=df.iloc[:,1:-2],
                                            targets=df.iloc[:,-2:],
                                            solution_length=numFeatures)[0]
        df_Selected = df.copy()[solutions[0]+["T_OS","E_OS"]]
    
    
    if scaleFlag:
        scaledFeatures = StandardScaler().fit_transform(df_Selected.iloc[:,:-2])
        df_Selected.iloc[:,:-2] = scaledFeatures
    
    return df_Selected

def prepTestData(df_test,featureData):
    
    df_test = df_test[featureData]
    
    scaledFeatures = StandardScaler().fit_transform(df_test.iloc[:,:-2])
    df_test.iloc[:,:-2] = scaledFeatures
    
    return df_test





# %% SURVIVAL ANALYSIS

"""
Models:
- Cox Proportional Hazards
- LASSO-Cox
- Random Survival Forest
"""

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
        

# %% DATA LOADING

# SARC021
sarc021_radiomics = pd.read_csv('Data/SARC021/SARC021_radiomics.csv')
sarc021_clinical = pd.read_csv('Data/SARC021/SARC021_clinical.csv')

# optional ? -- look only at those patients with 2+ lesions contoured
id_counts = sarc021_radiomics['USUBJID'].value_counts()
valid_ids = id_counts[id_counts >= 2].index
sarc021_radiomics2Plus = sarc021_radiomics[sarc021_radiomics['USUBJID'].isin(valid_ids)].reset_index()

# RADCURE
radcure_radiomics = pd.read_csv('Data/RADCURE/RADCURE_radiomics.csv')
radcure_clinical = pd.read_csv('Data/RADCURE/RADCURE_clinical.csv')

del(id_counts,valid_ids)



# %% TESTING SELECTION/AGGREGATION

sarc021_largestlung = selectLargestLungLesion(sarc021_radiomics2Plus, sarc021_clinical)
radiomics = sarc021_radiomics2Plus
clinical = sarc021_clinical

startColInd = np.where(radiomics.columns.str.find('original')==0)[0][0]
rowInds = np.where(radiomics.LABEL=='LUNG')[0]

df_LargestLung = radiomics.copy().iloc[:,startColInd:] 
df_LargestLung.insert(0, "USUBJID", radiomics.USUBJID, True)
df_LargestLung = df_LargestLung.iloc[rowInds,:]
df_LargestLung = df_LargestLung.groupby('USUBJID').max('original_shape_VoxelVolume').reset_index(drop=True)
df_LargestLung.insert(0, "USUBJID", np.unique(radiomics.USUBJID[rowInds]), True)

# %%
df = radcure_concat

# Assuming df is your original DataFrame

# Find unique suffixes in the column names
# suffixes = df.columns.str.extract(r'_(\w+)$')[0].unique()

# Create a dictionary to store DataFrames
dfs_dict = {}
numLesions = 7
# Iterate through unique suffixes and filter columns
for i in range(1,numLesions+1):
    suffix = 'Lesion'+str(i)
    filtered_cols = [col for col in df.columns if col.endswith(suffix)]
    dfs_dict[suffix] = df[filtered_cols]

# Display the resulting DataFrames in the dictionary
# for suffix, df_suffix in dfs_dict.items():
#     print(f"DataFrame with '{suffix}' suffix:")
#     print(df_suffix)
#     print("\n")


# %% TESTING


aggName = 'weighted 3 largest'

# RADCURE
df_imaging = radcure_radiomics
df_clinical = radcure_clinical
train,test = randomSplit(df_imaging,df_clinical,0.7,False)
val = np.nan

# SARC021
# df_imaging = sarc021_radiomics2Plus
# df_clinical = sarc021_clinical
# train,test,val = singleInstValidationSplit(df_imaging,df_clinical,0.7)

pipe_dict = {
                'train' : [train,True],
                'test'  : [test,False],
                'val'   : [val,False]
    }

# ----- TRAINING SET -----
# isolate the patients in the defined split (i.e., train/test/val)
df_imaging_train = df_imaging[df_imaging.USUBJID.isin(pipe_dict['train'][0])].reset_index()
df_clinical_train = df_clinical[df_clinical.USUBJID.isin(pipe_dict['train'][0])].reset_index()

# aggregate, reduce (variance and volume adjustment) and select features
trainingSet = featureSelection(featureReduction(calcVolumeWeightedAverageNLargest(df_imaging_train,df_clinical_train,numLesions=2)),volFlag=False,scaleFlag=True,numFeatures=10)
# trainingSet = calcCosineMetrics(df_imaging_train,df_clinical_train)

CPH_bootstrap(trainingSet,aggName,'OS',pipe_dict['train'][1])
LASSO_COX_bootstrap(trainingSet,aggName,'OS',pipe_dict['train'][1])
# RSF_bootstrap(selectedFeatures,aggName,'OS',pipe_dict[split][1])

# ----- TESTING SET -----
# isolate the patients in the defined split (i.e., train/test/val)
df_imaging_test = df_imaging[df_imaging.USUBJID.isin(pipe_dict['test'][0])].reset_index()
df_clinical_test = df_clinical[df_clinical.USUBJID.isin(pipe_dict['test'][0])].reset_index()

testingSet = prepTestData(calcVolumeWeightedAverageNLargest(df_imaging_test,df_clinical_test,numLesions=2), trainingSet.columns)
# testingSet = calcCosineMetrics(df_imaging_test,df_clinical_test)
CPH_bootstrap(trainingSet,aggName,'OS',pipe_dict['test'][1])
LASSO_COX_bootstrap(trainingSet,aggName,'OS',pipe_dict['test'][1])

# %%
# ----- SARC021 SINGLE INSTITUTION VALIDATION -----
# isolate the patients in the defined split (i.e., train/test/val)
df_imaging_val = df_imaging[df_imaging.USUBJID.isin(pipe_dict['val'][0])].reset_index()
df_clinical_val = df_clinical[df_clinical.USUBJID.isin(pipe_dict['val'][0])].reset_index()

validationSet = prepTestData(calcVolumeWeightedAverageNLargest(df_imaging_val,df_clinical_val,numLesions=2), trainingSet.columns)
# validationSet = calcCosineMetrics(df_imaging_val,df_clinical_val)
CPH_bootstrap(validationSet,aggName,'OS',pipe_dict['val'][1])
LASSO_COX_bootstrap(validationSet,aggName,'OS',pipe_dict['val'][1])


# %%
# Fit a Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(selectedFeatures, duration_col="T_OS", event_col="E_OS")

cph2 = CoxPHFitter()
cph2.fit(radcure_largest[["original_shape_VoxelVolume","T_OS","E_OS"]], duration_col="T_OS", event_col="E_OS")

# Display the results
print(cph.summary)


	

#%%
# ------------------------ #
# Cox Proportional Hazards
# ------------------------ #
# CPH_bootstrap(UWA_top40_fp)
# CPH_bootstrap(WA_top40_fp)
# CPH_bootstrap(big3_top40_fp)
# CPH_bootstrap(big1_top40_fp)
# CPH_bootstrap(big1_top40_fp, num=True)
# CPH_bootstrap(smallest_top40_fp)

# #%%
# # ---------------------- #
# # Lasso-Cox
# # ---------------------- #
# LASSO_COX_bootstrap(UWA_top40_fp)
# LASSO_COX_bootstrap(WA_top40_fp)
# LASSO_COX_bootstrap(big3_top40_fp)
# LASSO_COX_bootstrap(big1_top40_fp)
# LASSO_COX_bootstrap(big1_top40_fp, num=True)
# LASSO_COX_bootstrap(smallest_top40_fp)

# # ----------------------- #
# # Random Survival Forest
# # ----------------------- #
# RSF_bootstrap(UWA_top40_fp)
# RSF_bootstrap(WA_top40_fp)
# RSF_bootstrap(big3_top40_fp)
# RSF_bootstrap(big1_top40_fp)
# RSF_bootstrap(big1_top40_fp, num=True)
# RSF_bootstrap(smallest_top40_fp)

# # -------------------- #
# # Sub-Analysis: Num Mets < 5, 5-10, 11+
# # -------------------- #

# df = pd.read_csv(UWA_top40_fp, index_col=0)

# # map labels to radiomic data
# df['nummets'] = df.index.to_series().map(nummetsdic)
# df['nummets'] = df['nummets'].astype(int)

# # create sub-groups
# df_5 = df.loc[df['nummets'] < 5].copy()
# df_5_10 = df.loc[(df['nummets'] >= 5) & (df['nummets'] <= 10)].copy()
# df_11 = df.loc[df['nummets'] >= 11].copy()

# CPH_bootstrap(UWA_top40_fp, sub=df_5)
# CPH_bootstrap(UWA_top40_fp, sub=df_5_10)
# CPH_bootstrap(UWA_top40_fp, sub=df_11)


# df = pd.read_csv(WA_top40_fp, index_col=0)

# # map labels to radiomic data
# df['nummets'] = df.index.to_series().map(nummetsdic)
# df['nummets'] = df['nummets'].astype(int)

# # create sub-groups
# df_5 = df.loc[df['nummets'] < 5].copy()
# df_5_10 = df.loc[(df['nummets'] >= 5) & (df['nummets'] <= 10)].copy()
# df_11 = df.loc[df['nummets'] >= 11].copy()

# CPH_bootstrap(WA_top40_fp, sub=df_5)
# CPH_bootstrap(WA_top40_fp, sub=df_5_10)
# CPH_bootstrap(WA_top40_fp, sub=df_11)

# df = pd.read_csv(big3_top40_fp, index_col=0)

# # map labels to radiomic data
# df['nummets'] = df.index.to_series().map(nummetsdic)
# df['nummets'] = df['nummets'].astype(int)

# # create sub-groups
# df_5 = df.loc[df['nummets'] < 5].copy()
# df_5_10 = df.loc[(df['nummets'] >= 5) & (df['nummets'] <= 10)].copy()
# df_11 = df.loc[df['nummets'] >= 11].copy()

# CPH_bootstrap(big3_top40_fp, sub=df_5)
# CPH_bootstrap(big3_top40_fp, sub=df_5_10)
# CPH_bootstrap(big3_top40_fp, sub=df_11)

# df = pd.read_csv(big1_top40_fp, index_col=0)

# # map labels to radiomic data
# df['nummets'] = df.index.to_series().map(nummetsdic)
# df['nummets'] = df['nummets'].astype(int)

# # create sub-groups
# df_5 = df.loc[df['nummets'] < 5].copy()
# df_5_10 = df.loc[(df['nummets'] >= 5) & (df['nummets'] <= 10)].copy()
# df_11 = df.loc[df['nummets'] >= 11].copy()

# CPH_bootstrap(big1_top40_fp, sub=df_5)
# CPH_bootstrap(big1_top40_fp, sub=df_5_10)
# CPH_bootstrap(big1_top40_fp, sub=df_11)

# df = pd.read_csv(big1_top40_fp, index_col=0)

# # map labels to radiomic data
# df['nummets'] = df.index.to_series().map(nummetsdic)
# df['nummets'] = df['nummets'].astype(int)

# # create sub-groups
# df_5 = df.loc[df['nummets'] < 5].copy()
# df_5_10 = df.loc[(df['nummets'] >= 5) & (df['nummets'] <= 10)].copy()
# df_11 = df.loc[df['nummets'] >= 11].copy()

# CPH_bootstrap(big1_top40_fp, num=True, sub=df_5)
# CPH_bootstrap(big1_top40_fp, num=True, sub=df_5_10)
# CPH_bootstrap(big1_top40_fp, num=True, sub=df_11)

# df = pd.read_csv(smallest_top40_fp, index_col=0)

# # map labels to radiomic data
# df['nummets'] = df.index.to_series().map(nummetsdic)
# df['nummets'] = df['nummets'].astype(int)

# # create sub-groups
# df_5 = df.loc[df['nummets'] < 5].copy()
# df_5_10 = df.loc[(df['nummets'] >= 5) & (df['nummets'] <= 10)].copy()
# df_11 = df.loc[df['nummets'] >= 11].copy()

# CPH_bootstrap(smallest_top40_fp, sub=df_5)
# CPH_bootstrap(smallest_top40_fp, sub=df_5_10)
# CPH_bootstrap(smallest_top40_fp, sub=df_11)

# # -------------------- #
# # Sub-Analysis: Volume Largest Met <200, 200-700, >700
# # -------------------- #

# df = pd.read_csv(UWA_top40_fp, index_col=0)

# # map labels to radiomic data
# df['volume'] = df.index.to_series().map(big1voldic)
# df['volume'] = df['volume'].astype(int)

# # create sub-groups
# df_200 = df.loc[df['volume'] < 200].copy()
# df_200_700 = df.loc[(df['volume'] >= 200) & (df['volume'] <= 700)].copy()
# df_700 = df.loc[df['volume'] >= 700].copy()

# CPH_bootstrap(UWA_top40_fp, sub=df_200)
# CPH_bootstrap(UWA_top40_fp, sub=df_200_700)
# CPH_bootstrap(UWA_top40_fp, sub=df_700)

# df = pd.read_csv(WA_top40_fp, index_col=0)

# # map labels to radiomic data
# df['volume'] = df.index.to_series().map(big1voldic)
# df['volume'] = df['volume'].astype(int)

# # create sub-groups
# df_200 = df.loc[df['volume'] < 200].copy()
# df_200_700 = df.loc[(df['volume'] >= 200) & (df['volume'] <= 700)].copy()
# df_700 = df.loc[df['volume'] >= 700].copy()

# CPH_bootstrap(WA_top40_fp, sub=df_200)
# CPH_bootstrap(WA_top40_fp, sub=df_200_700)
# CPH_bootstrap(WA_top40_fp, sub=df_700)

# df = pd.read_csv(big3_top40_fp, index_col=0)

# # map labels to radiomic data
# df['volume'] = df.index.to_series().map(big1voldic)
# df['volume'] = df['volume'].astype(int)

# # create sub-groups
# df_200 = df.loc[df['volume'] < 200].copy()
# df_200_700 = df.loc[(df['volume'] >= 200) & (df['volume'] <= 700)].copy()
# df_700 = df.loc[df['volume'] >= 700].copy()

# CPH_bootstrap(big3_top40_fp, sub=df_200)
# CPH_bootstrap(big3_top40_fp, sub=df_200_700)
# CPH_bootstrap(big3_top40_fp, sub=df_700)

# df = pd.read_csv(big1_top40_fp, index_col=0)

# # map labels to radiomic data
# df['volume'] = df.index.to_series().map(big1voldic)
# df['volume'] = df['volume'].astype(int)

# # create sub-groups
# df_200 = df.loc[df['volume'] < 200].copy()
# df_200_700 = df.loc[(df['volume'] >= 200) & (df['volume'] <= 700)].copy()
# df_700 = df.loc[df['volume'] >= 700].copy()

# CPH_bootstrap(big1_top40_fp, sub=df_200)
# CPH_bootstrap(big1_top40_fp, sub=df_200_700)
# CPH_bootstrap(big1_top40_fp, sub=df_700)

# df = pd.read_csv(big1_top40_fp, index_col=0)

# # map labels to radiomic data
# df['volume'] = df.index.to_series().map(big1voldic)
# df['volume'] = df['volume'].astype(int)

# # create sub-groups
# df_200 = df.loc[df['volume'] < 200].copy()
# df_200_700 = df.loc[(df['volume'] >= 200) & (df['volume'] <= 700)].copy()
# df_700 = df.loc[df['volume'] >= 700].copy()

# CPH_bootstrap(big1_top40_fp, num=True, sub=df_200)
# CPH_bootstrap(big1_top40_fp, num=True, sub=df_200_700)
# CPH_bootstrap(big1_top40_fp, num=True, sub=df_700)

# df = pd.read_csv(smallest_top40_fp, index_col=0)

# # map labels to radiomic data
# df['volume'] = df.index.to_series().map(big1voldic)
# df['volume'] = df['volume'].astype(int)

# # create sub-groups
# df_200 = df.loc[df['volume'] < 200].copy()
# df_200_700 = df.loc[(df['volume'] >= 200) & (df['volume'] <= 700)].copy()
# df_700 = df.loc[df['volume'] >= 700].copy()

# CPH_bootstrap(smallest_top40_fp, sub=df_200)
# CPH_bootstrap(smallest_top40_fp, sub=df_200_700)
# CPH_bootstrap(smallest_top40_fp, sub=df_700)




