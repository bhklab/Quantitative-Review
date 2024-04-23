#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 29 2023
Last updated Feb 16 2024

@author: caryn-geady
"""

''' 

Lesion-Level Aggregation Methods:
    
    - Unweighted Average
    - Volume-Weighted Average
    - Volume-Weighted Average of 3 Largest lesions
    - Concatenation (choose 2)
    - Cosine Similarity Metrics (X2)
    - Inter-site similarity matrix (??? to be figured out ???)
    - Unsupervised clustering (??? to be figured out ???)

'''

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import Code.feature_handling as fh
import Code.lesion_selection as ls


def calcNumMets(radiomics):
    """
    Calculate the number of lesions for each unique USUBJID in the radiomics dataset.
    
    Parameters:
    - radiomics (DataFrame): The radiomics dataset containing USUBJID information.
    
    Returns:
    - numMet (DataFrame): DataFrame with two columns - 'USUBJID' and 'NumMets', representing the unique USUBJID and the corresponding number of metastases.
    """

    ids,counts = np.unique(radiomics.USUBJID,return_counts=True)
    numMets = pd.DataFrame([ids,counts]).T
    numMets.columns = ['USUBJID','NumMets']
    
    return numMets

def calcUnweightedAverage(radiomics, clinical, outcome='OS', scaleFlag=False, numMetsFlag=False, multipleFlag=False):
    """
    Calculate the unweighted average of radiomics features for each subject.

    Parameters:
    - radiomics (DataFrame): DataFrame containing radiomics features.
    - clinical (DataFrame): DataFrame containing clinical data.
    - outcome (str, optional): Outcome variable. Defaults to 'OS'.
    - scaleFlag (bool, optional): Flag indicating whether to scale the features. Defaults to False.
    - numMetsFlag (bool, optional): Flag indicating whether to calculate the number of metastases. Defaults to True.
    - multipleFlag (bool, optional): Flag indicating whether to include only subjects with multiple metastases. Defaults to True.

    Returns:
    - df_UnweightedAverage (DataFrame): Unweighted average of radiomics features and the specified outcome variable for each subject.
    """
    
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

def calcVolumeWeightedAverage(radiomics, clinical, outcome='OS', scaleFlag=False, numMetsFlag=False, multipleFlag=False):
    """
    Calculates the volume-weighted average of radiomics features for each subject.

    Parameters:
    - radiomics (DataFrame): DataFrame containing radiomics features for each subject.
    - clinical (DataFrame): DataFrame containing clinical data for each subject.
    - outcome (str, optional): The outcome variable to include in the result DataFrame. Defaults to 'OS'.
    - scaleFlag (bool, optional): Flag indicating whether to scale the features using StandardScaler. Defaults to False.
    - numMetsFlag (bool, optional): Flag indicating whether to calculate the number of metastases for each subject. Defaults to True.
    - multipleFlag (bool, optional): Flag indicating whether to include only subjects with more than one metastasis. Defaults to True.

    Returns:
    - df_WeightedAverage (DataFrame): Volume-weighted average of radiomics features and the specified outcome variable for each subject.
    """
    
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

def calcVolumeWeightedAverageNLargest(radiomics, clinical, numLesions=3, outcome='OS', scaleFlag=False, numMetsFlag=False):
    """
    Calculate the volume-weighted average of the n largest lesions for each subject.

    Parameters:
    - radiomics (DataFrame): DataFrame containing radiomics data.
    - clinical (DataFrame): DataFrame containing clinical data.
    - numLesions (int): Number of largest lesions to consider (default is 3).
    - outcome (str): Outcome variable to consider (default is 'OS').
    - scaleFlag (bool): Flag indicating whether to scale the features (default is False).
    - numMetsFlag (bool): Flag indicating whether to calculate the number of metastases (default is True).

    Returns:
    - df_VolWeightNLargest (DataFrame): Volume-weighted average of the N-largest lesions and the specified outcome variable for each subject.
    """
    
    # id_counts = radiomics['USUBJID'].value_counts()
    # valid_ids = id_counts[id_counts >= numLesions].index
    # df_radiomics = radiomics[radiomics['USUBJID'].isin(valid_ids)]
    
    df_filtered = radiomics.groupby('USUBJID').apply(lambda group: group.nlargest(numLesions, 'original_shape_VoxelVolume')).reset_index(drop=True)
    
    df_VolWeightNLargest = calcVolumeWeightedAverage(df_filtered, clinical, numMetsFlag=False)
    
    if scaleFlag:
        scaledFeatures = StandardScaler().fit_transform(df_VolWeightNLargest.iloc[:,1:-2])
        df_VolWeightNLargest.iloc[:,1:-2] = scaledFeatures
    
    if numMetsFlag:
        df_VolWeightNLargest.insert(1, "NumMets", calcNumMets(radiomics).NumMets, True)
        
    return df_VolWeightNLargest

def concatenateNLargest(radiomics, clinical, numLesions=3, outcome='OS', scaleFlag=False, numMetsFlag=False):
    """
    Concatenates the largest lesions from radiomics data with clinical data.
    
    Parameters:
    - radiomics (DataFrame): The radiomics data.
    - clinical (DataFrame): The clinical data.
    - numLesions (int, optional): The number of largest lesions to consider. Defaults to 3.
    - outcome (str, optional): The outcome variable. Defaults to 'OS'.
    - scaleFlag (bool, optional): Flag indicating whether to scale the features. Defaults to False.
    - numMetsFlag (bool, optional): Flag indicating whether to calculate the number of metastases. Defaults to True.
    
    Returns:
    - df_Concatenated (DataFrame): Concatenated radiomics data of the N-largest lesions and the specified outcome variable for each subject.
    """
    
    startColInd = np.where(radiomics.columns.str.find('original')==0)[0][0]
    
    # id_counts = radiomics['USUBJID'].value_counts()
    # valid_ids = id_counts[id_counts >= numLesions].index
    # df_radiomics = radiomics[radiomics['USUBJID'].isin(valid_ids)]
    
    df_radiomics = radiomics.iloc[:,startColInd:]
    df_radiomics.insert(0, "USUBJID", radiomics.USUBJID)
    
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
    # if numLesions == 1:
    #     df_Concatenated = df_Concatenated.drop('original_shape_VoxelVolume_Lesion1')
    df_Concatenated.insert(1,"original_shape_VoxelVolume",total_volume,True)    
    df_Concatenated = df_Concatenated.merge(clinical[['USUBJID','T_'+outcome,'E_'+outcome]],on='USUBJID').reset_index(drop=True)

    if scaleFlag:
        scaledFeatures = StandardScaler().fit_transform(df_Concatenated.iloc[:,1:-2])
        df_Concatenated.iloc[:,1:-2] = scaledFeatures
    
    if numMetsFlag:
        df_Concatenated.insert(1,"NumMets",calcNumMets(df_radiomics).NumMets,True)
    
    if numLesions == 1:
        return ls.selectLargestLesion(radiomics,clinical,scaleFlag=True)
    else: 
        return df_Concatenated
    # return df_Concatenated

def calcCosineMetrics(radiomics, clinical, numLesions=3, outcome='OS', scaleFlag=False, numMetsFlag=False):
    """
    Calculate cosine similarity metrics for radiomics data.

    Parameters:
    - radiomics (DataFrame): DataFrame containing radiomics data.
    - clinical (DataFrame): DataFrame containing clinical data.
    - numLesions (int): Minimum number of lesions required for a patient to be included in the analysis. Default is 3.
    - outcome (str): Outcome variable for survival analysis. Default is 'OS'.
    - scaleFlag (bool): Flag indicating whether to scale the features. Default is False.
    - numMetsFlag (bool): Flag indicating whether to calculate the number of metastases. Default is True.

    Returns:
    - df_CosineMetrics (DataFrame): Calculated cosine similarity metrics and the specified outcome variable for each subject.

    """
    startColInd = np.where(radiomics.columns.str.find('original')==0)[0][0]
    
    id_counts = radiomics['USUBJID'].value_counts()
    valid_ids = id_counts[id_counts >= numLesions].index
    df_radiomics = radiomics[radiomics['USUBJID'].isin(valid_ids)]
    
    df_radiomics = df_radiomics.iloc[:,startColInd:]
    
    # perform feature reduction
    df_radiomics = fh.featureReduction(df_radiomics,scaleFlag=False)
    df_radiomics.insert(0, "USUBJID", radiomics.USUBJID[radiomics['USUBJID'].isin(valid_ids)], True)
    
    df_radiomics = df_radiomics.groupby('USUBJID').apply(lambda x: x.sample(numLesions)).reset_index(drop=True)
    # df_radiomics contains only those patients with 3+ lesions
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

def interLesionRelationNetwork(radiomics):
    
    '''
    Algorithm:
        - using original features only:
            - separate by feature class (first-order, shape, GLCM, etc.)
            - calculate the data depth of each class on a patient-by-patient basis
            - this should reduce the feature set for each patient from 1218-->100-->6
        - using the reduced feature set of class-specific data depth for all lesions,
            - cluster lesions using k-means clustering
            - grid search for appropriate number of clusters to maintain intracluster homoegeneity
            - this reduces each lesion-specific feature set from 6-->1, where
            - the remaining number is a radiographic lesion class
        - "Qualitative Assessment of Inter-tumor Heterogeneity" -- separate patients into 2 groups:
            - homogeneous radiomic profiles
            - heterogeneous radiomic profiles
        - "Quantitative Assessment of Inter-tumor Heterogeneity" -- using the feature set of 6 data depths per lesion:
            - create a patient-specific dendrogram 
            - calculate derived metrics from the dendrogram:
                1. number of lesions
                2. sum of tree branch lengths
                3. dispersion among lesions
                4. number of different phenotypes
    
    
    '''
    feature_classes = ['firstorder','shape','glcm','glrlm','glszm','gldm']
    
    df_original = radiomics.iloc[:,np.where(radiomics.columns.str.find('original')==0)[0]]
    df_original.insert(0,"USUBJID",radiomics.USUBJID)
    # cols = cols[np.where(cols.str.contains('original'))[0]]
    
    return df_original


