#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 29 2023
Last updated Feb 16 2024

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

'''

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

# %% MISC

def calcNumMets(radiomics):
    
    ids,counts = np.unique(radiomics.USUBJID,return_counts=True)
    numMets = pd.DataFrame([ids,counts]).T
    numMets.columns = ['USUBJID','NumMets']
    
    return numMets


# %% LESION SELECTION METHODS 
 
# def selectPrimaryTumor (dataset-specific hypothesis)
def selectPrimaryTumor(radiomics,clinical,outcome='OS',scaleFlag=False,numMetsFlag=True,multipleFlag=True):
    
    startColInd = np.where(radiomics.columns.str.find('original')==0)[0][0]
    rowInds = np.where(radiomics.LABEL.str.find('GTVp')==0)[0]
    
    df_Primary = radiomics.copy().iloc[rowInds,startColInd:]
    df_Primary.insert(0, "USUBJID", radiomics.USUBJID.iloc[rowInds], True)
    
    df_Primary = df_Primary.merge(clinical[['USUBJID','T_'+outcome,'E_'+outcome]],on='USUBJID').reset_index(drop=True)
    
    if scaleFlag:
        scaledFeatures = StandardScaler().fit_transform(df_Primary.iloc[:,1:-2])
        df_Primary.iloc[:,1:-2] = scaledFeatures
    
    if numMetsFlag:
        df_Primary = calcNumMets(radiomics).merge(df_Primary,on='USUBJID').reset_index(drop=True)
        
        if multipleFlag:
            df_Primary = df_Primary.loc[df_Primary.NumMets>1,:]
        
    return df_Primary

def selectSmallestLesion(radiomics,clinical,outcome='OS',scaleFlag=False,numMetsFlag=True,multipleFlag=True):
    
    startColInd = np.where(radiomics.columns.str.find('original')==0)[0][0]
    
    df_Smallest = radiomics.copy().iloc[:,startColInd:] 
    df_Smallest.insert(0, "USUBJID", radiomics.USUBJID, True)
    df_Smallest = df_Smallest.groupby('USUBJID').min('original_shape_VoxelVolume').reset_index(drop=True)
    df_Smallest.insert(0, "USUBJID", np.unique(radiomics.USUBJID), True)
    
    df_Smallest = df_Smallest.merge(clinical[['USUBJID','T_'+outcome,'E_'+outcome]],on='USUBJID').reset_index(drop=True)
    
    if scaleFlag:
        scaledFeatures = StandardScaler().fit_transform(df_Smallest.iloc[:,1:-2])
        df_Smallest.iloc[:,1:-2] = scaledFeatures
    
    if numMetsFlag:
        df_Smallest = calcNumMets(radiomics).merge(df_Smallest,on='USUBJID').reset_index(drop=True)
        
        if multipleFlag:
            df_Smallest = df_Smallest.loc[df_Smallest.NumMets>1,:]
        
    return df_Smallest
        
def selectLargestLesion(radiomics,clinical,outcome='OS',scaleFlag=False,numMetsFlag=True,multipleFlag=True):
    
    startColInd = np.where(radiomics.columns.str.find('original')==0)[0][0]
    
    df_Largest = radiomics.copy().iloc[:,startColInd:] 
    df_Largest.insert(0, "USUBJID", radiomics.USUBJID, True)
    df_Largest = df_Largest.groupby('USUBJID').max('original_shape_VoxelVolume').reset_index(drop=True)
    df_Largest.insert(0, "USUBJID", np.unique(radiomics.USUBJID), True)
    
    df_Largest = df_Largest.merge(clinical[['USUBJID','T_'+outcome,'E_'+outcome]],on='USUBJID').reset_index(drop=True)
    
    if scaleFlag:
        scaledFeatures = StandardScaler().fit_transform(df_Largest.iloc[:,1:-2])
        df_Largest.iloc[:,1:-2] = scaledFeatures
    
    if numMetsFlag:
        df_Largest = calcNumMets(radiomics).merge(df_Largest,on='USUBJID').reset_index(drop=True)
        
        if multipleFlag:
            df_Largest = df_Largest.loc[df_Largest.NumMets>1,:]
        
    return df_Largest

# def selectLargestLungLesion (sarcoma-specific hypothesis)
def selectLargestLungLesion(radiomics,clinical,outcome='OS',scaleFlag=False,numMetsFlag=True,multipleFlag=True):
    
    startColInd = np.where(radiomics.columns.str.find('original')==0)[0][0]
    rowInds = np.where(radiomics.LABEL=='LUNG')[0]
    
    df_LargestLung = radiomics.copy().iloc[:,startColInd:] 
    df_LargestLung.insert(0, "USUBJID", radiomics.USUBJID, True)
    df_LargestLung = df_LargestLung.iloc[rowInds,:]
    df_LargestLung = df_LargestLung.groupby('USUBJID').max('original_shape_VoxelVolume').reset_index(drop=True)
    df_LargestLung.insert(0, "USUBJID", np.unique(radiomics.USUBJID[rowInds]), True)
    
    df_LargestLung = df_LargestLung.merge(clinical[['USUBJID','T_'+outcome,'E_'+outcome]],on='USUBJID').reset_index(drop=True)
    
    if scaleFlag:
        scaledFeatures = StandardScaler().fit_transform(df_LargestLung.iloc[:,1:-2])
        df_LargestLung.iloc[:,1:-2] = scaledFeatures
    
    if numMetsFlag:
        df_LargestLung = calcNumMets(radiomics).merge(df_LargestLung,on='USUBJID').reset_index(drop=True)
        
        if multipleFlag:
            df_LargestLung = df_LargestLung.loc[df_LargestLung.NumMets>1,:]
        
    return df_LargestLung