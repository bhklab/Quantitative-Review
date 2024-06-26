#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 29 2023
Last updated Feb 16 2024

@author: caryn-geady
"""

'''
The Python module "lesion_selection.py" defines several functions for lesion selection based on radiomics data. Here is a summary of the functions:

calcNumMets(radiomics): Calculates the number of lesions for each unique USUBJID in the radiomics dataset and returns a DataFrame with two columns - 'USUBJID' and 'NumMets'.

selectPrimaryTumor(radiomics, clinical, outcome='OS', scaleFlag=False, numMetsFlag=True, multipleFlag=False): Selects primary tumors from the radiomics data and merges them with the clinical outcome. It returns a DataFrame containing radiomic data from the primary tumor and the specified outcome variable for each subject.

selectSmallestLesion(radiomics, clinical, outcome='OS', scaleFlag=False, numMetsFlag=True, multipleFlag=False): Selects the smallest lesion based on radiomics data and merges it with the clinical outcome. It returns a DataFrame containing radiomic data from the smallest lesion and the specified outcome variable for each subject.

selectLargestLesion(radiomics, clinical, outcome='OS', scaleFlag=False, numMetsFlag=True, multipleFlag=False): Selects the largest lesion based on radiomics data and merges it with the clinical outcome. It returns a DataFrame containing radiomic data from the largest lesion and the specified outcome variable for each subject.

selectLargestLungLesion(radiomics, clinical, outcome='OS', scaleFlag=False, numMetsFlag=True, multipleFlag=False): Selects the largest lung lesion from the radiomics data and merges it with the clinical outcome. It returns a DataFrame containing radiomic data from the largest lung lesion and the specified outcome variable for each subject.

These functions take the radiomics and clinical datasets as input, along with optional parameters for outcome variable, scaling, number of metastases calculation, and multiple metastases filtering. They perform data manipulation and merging operations to extract the desired lesion information and return the resulting DataFrame.

'''

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler


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
 
# def selectPrimaryTumor (dataset-specific hypothesis)
def selectPrimaryTumor(radiomics, clinical, outcome='OS', scaleFlag=False, numMetsFlag=False, multipleFlag=False):
    """
    RADCURE-SPECIFIC: selects primary tumors from radiomics data and merges it with the clinical outcome.

    Parameters:
    - radiomics (DataFrame): DataFrame containing radiomics data.
    - clinical (DataFrame): DataFrame containing clinical data.
    - outcome (str, optional): Outcome variable to include in the selected primary tumors DataFrame. Defaults to 'OS'.
    - scaleFlag (bool, optional): Flag indicating whether to scale the features in the selected primary tumors DataFrame. Defaults to False.
    - numMetsFlag (bool, optional): Flag indicating whether to calculate the number of metastases and include it in the selected primary tumors DataFrame. Defaults to True.
    - multipleFlag (bool, optional): Flag indicating whether to include only primary tumors with multiple metastases in the selected primary tumors DataFrame. Defaults to True.

    Returns:
    - df_Primary (DataFrame): Radiomic data from the primary tumor and the specified outcome variable for each subject.
    """
    
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

def selectSmallestLesion(radiomics, clinical, outcome='OS', scaleFlag=False, numMetsFlag=False, multipleFlag=False):
    """
    Selects the smallest lesion based on radiomics data and merges it with the clinical outcome.

    Parameters:
    - radiomics (DataFrame): DataFrame containing radiomics data.
    - clinical (DataFrame): DataFrame containing clinical information.
    - outcome (str, optional): Outcome variable for survival analysis. Defaults to 'OS'.
    - scaleFlag (bool, optional): Flag indicating whether to scale the features. Defaults to False.
    - numMetsFlag (bool, optional): Flag indicating whether to calculate the number of metastases. Defaults to True.
    - multipleFlag (bool, optional): Flag indicating whether to include only cases with multiple metastases. Defaults to True.

    Returns:
    - df_Smallest (DataFrame): Radiomic data from the smallest lesion and the specified outcome variable for each subject.
    """
    
    startColInd = np.where(radiomics.columns.str.find('original')==0)[0][0]
    
    df_Smallest = radiomics.copy().iloc[:,startColInd:] 
    df_Smallest.insert(0, "USUBJID", radiomics.USUBJID, True)
    df_Smallest = df_Smallest.groupby('USUBJID').apply(lambda group: group.nsmallest(1, 'original_shape_VoxelVolume')).reset_index(drop=True)

    df_Smallest = df_Smallest.merge(clinical[['USUBJID','T_'+outcome,'E_'+outcome]],on='USUBJID').reset_index(drop=True)
    
    if scaleFlag:
        scaledFeatures = StandardScaler().fit_transform(df_Smallest.iloc[:,1:-2])
        df_Smallest.iloc[:,1:-2] = scaledFeatures
    
    if numMetsFlag:
        df_Smallest = calcNumMets(radiomics).merge(df_Smallest,on='USUBJID').reset_index(drop=True)
        
        if multipleFlag:
            df_Smallest = df_Smallest.loc[df_Smallest.NumMets>1,:]
        
    return df_Smallest
        
def selectLargestLesion(radiomics, clinical, outcome='OS', scaleFlag=False, numMetsFlag=False, multipleFlag=False):
    """
    Selects the largest lesion based on radiomics data and merges it with the clinical outcome.

    Parameters:
    - radiomics (DataFrame): DataFrame containing radiomics data.
    - clinical (DataFrame): DataFrame containing clinical information.
    - outcome (str, optional): Outcome variable to be included in the resulting DataFrame. Defaults to 'OS'.
    - scaleFlag (bool, optional): Flag indicating whether to scale the features. Defaults to False.
    - numMetsFlag (bool, optional): Flag indicating whether to calculate the number of metastases. Defaults to True.
    - multipleFlag (bool, optional): Flag indicating whether to include only cases with multiple metastases. Defaults to True.

    Returns:
    - df_Largest (DataFrame): Radiomic data from the largest lesion and the specified outcome variable for each subject.
    """
    
    startColInd = np.where(radiomics.columns.str.find('original')==0)[0][0]
    
    df_Largest = radiomics.copy().iloc[:,startColInd:] 
    df_Largest.insert(0, "USUBJID", radiomics.USUBJID, True)
    df_Largest = df_Largest.groupby('USUBJID').apply(lambda group: group.nlargest(1, 'original_shape_VoxelVolume')).reset_index(drop=True)

    df_Largest = df_Largest.merge(clinical[['USUBJID','T_'+outcome,'E_'+outcome]],on='USUBJID').reset_index(drop=True)
    
    if scaleFlag:
        scaledFeatures = StandardScaler().fit_transform(df_Largest.iloc[:,1:-2])
        df_Largest.iloc[:,1:-2] = scaledFeatures
    
    if numMetsFlag:
        df_Largest = calcNumMets(radiomics).merge(df_Largest,on='USUBJID').reset_index(drop=True)
        
    if numMetsFlag and multipleFlag:
        df_Largest = df_Largest.loc[df_Largest.NumMets>1,:]
    
    return df_Largest

# def selectLargestLungLesion (sarcoma-specific hypothesis)
def selectLargestLungLesion(radiomics, clinical, outcome='OS', scaleFlag=False, numMetsFlag=False, multipleFlag=False):
    """
    SARCOMA-SPECIFIC: selects the largest lung lesion from the radiomics data and merges it with the clinical outcome.
    
    Parameters:
    - radiomics (DataFrame): The radiomics data.
    - clinical (DataFrame): The clinical data.
    - outcome (str): The outcome variable to be used for merging the data. Default is 'OS'.
    - scaleFlag (bool): Flag indicating whether to scale the features. Default is False.
    - numMetsFlag (bool): Flag indicating whether to calculate the number of metastases. Default is True.
    - multipleFlag (bool): Flag indicating whether to filter out cases with multiple metastases. Default is True.
    
    Returns:
    - df_LargestLung (DataFrame): Radiomic data from the largest lung lesion and the specified outcome variable for each subject.
    """
    
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