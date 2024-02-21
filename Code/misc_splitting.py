#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 29 2023
Last updated Feb 16 2024

@author: caryn-geady
"""

'''
Miscellaneous helper and data-splitting functions.
'''

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

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

def printLesionReport(radiomics):
    """
    Prints a report based on the given radiomics data. Lists statistics on the number of institutions and patients per institution (where applicable), and the number of lesions per patient (where applicable).

    Parameters:
    - radiomics (DataFrame): PyRadiomics raw output.

    Returns:
    None
    """
    
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
        
def randomSplit(radiomics, clinical, train_size=0.8, tuneFlag=False):
    """
    Splits the data into train, tune, and test sets based on the given parameters.

    Parameters:
    - radiomics (DataFrame): The radiomics data.
    - clinical (DataFrame): The clinical data.
    - train_size (float): The proportion of data to be used for training. Default is 0.8.
    - tuneFlag (bool): Flag indicating whether to include a tune set. Default is False.

    Returns:
    - train_ids (array): The IDs of the samples in the training set.
    - test_ids (array): The IDs of the samples in the testing set.
    - tune_ids (array): The IDs of the samples in the tuning set (optional, if tuneFlag == True).
    """
    
    test_size = 1 - train_size
    ids_to_keep = np.intersect1d(clinical.USUBJID, np.unique(radiomics.USUBJID))
    train_ids, test_ids = train_test_split(ids_to_keep, test_size=test_size, random_state=42)

    if tuneFlag:
        tune_size = 0.5
        tune_ids, test_ids = train_test_split(ids_to_keep, test_size=tune_size, random_state=42)
        return train_ids, test_ids, tune_ids
    else:
        return train_ids, test_ids
    
def singleInstValidationSplit(radiomics, clinical, train_size=0.8):
    """
    Splits the data into training, testing, and validation sets based on the institution code. Validation set is the largest single institution.

    Parameters:
    - radiomics (DataFrame): The radiomics data.
    - clinical (DataFrame): The clinical data.
    - train_size (float): The proportion of data to be used for training (default is 0.8).

    Returns:
    - train_ids (array): The IDs of the samples in the training set.
    - test_ids (array): The IDs of the samples in the testing set.
    - validation_ids (array): The IDs of the samples in the validation set.
    """
    
    test_size = 1 - train_size
    ids_to_keep = np.intersect1d(clinical.USUBJID, np.unique(radiomics.USUBJID))
    
    instCodes = np.array([p[-6:-3] for p in ids_to_keep])

    institutions,numPatients = np.unique(instCodes,return_counts=True)  
    largestSingleInst = institutions[np.where(numPatients==np.max(numPatients))]

    validation_ids = ids_to_keep[instCodes==largestSingleInst]

    train_ids,test_ids = train_test_split(ids_to_keep[instCodes!=largestSingleInst],test_size=test_size,random_state=42)
    
    return train_ids, test_ids, validation_ids

def convert_to_datetime(input_str, parserinfo=None):
    """
    Convert a string representation of a date and time to a datetime object.

    Parameters:
    - input_str (str): The string representation of the date and time.
    - parserinfo (parserinfo, optional): The parserinfo object to use for parsing the input string. Defaults to None.

    Returns:
    - datetime: The datetime object representing the input string.

    """
    return parse(input_str, parserinfo=parserinfo)


def splitByScanDate(radiomics,train_test_ids,train_size = 0.8):
    """
    Splits the radiomics data by scan date into training and testing sets based on the specified train size.

    Parameters:
    - radiomics (DataFrame): The radiomics data.
    - train_test_ids (array-like): The IDs of the subjects to be split.
    - train_size (float, optional): The proportion of data to be used for training. Defaults to 0.8.

    Returns:
    - tuple: A tuple containing two arrays, the first one representing the IDs of the subjects in the training set,
               and the second one representing the IDs of the subjects in the testing set.
    """
    
    df_scanDates = radiomics.groupby('USUBJID').agg({'SCANDATE':'max'}).reset_index()
    dates = np.array([convert_to_datetime(df_scanDates.SCANDATE[i]).toordinal() for i in range(len(df_scanDates))])
    df_scanDates.insert(2,"DATENUM",dates)
    df_filter = df_scanDates.copy()[df_scanDates.USUBJID.isin(train_test_ids)]
    
    df_sorted = df_filter.sort_values(by='DATENUM').reset_index(drop=True)
    split_index = int(train_size * len(df_sorted))
    
    return df_sorted.USUBJID.iloc[:split_index].values, df_sorted.USUBJID.iloc[split_index:].values