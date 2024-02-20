#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 29 2023
Last updated Feb 16 2024

@author: caryn-geady
"""

'''
Two-stage unsupervised feature reduction: (1) variance filter, (2) volume correlation filter.
'''

from sklearn.preprocessing import StandardScaler


def varianceFilter(radiomics,varThresh=10,outcome='OS',returnColsFlag=True):
    """
    Filters the radiomics dataframe based on variance threshold.

    Parameters:
    - radiomics (DataFrame): The radiomics dataframe.
    - varThresh (float): The variance threshold. Default is 10.
    - outcome (str): The outcome variable. Default is 'OS'.
    - returnColsFlag (bool): Flag to indicate whether to return the dropped columns. Default is True.

    Returns:
    - cols_to_drop (Index): The dropped columns (if returnColsFlag is True).
    - radiomics (DataFrame): The filtered radiomics dataframe.
    """
    
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
    

def volumeFilter(radiomics, volThresh=0.1, outcome='OS', returnColsFlag=True):
    """
    Filters the columns of a radiomics dataframe based on the correlation with the 'original_shape_VoxelVolume' column.

    Parameters:
    - radiomics (DataFrame): The radiomics dataframe.
    - volThresh (float, optional): The correlation threshold. Columns with absolute correlation greater than volThresh will be dropped. Default is 0.1.
    - outcome (str, optional): The outcome variable. Default is 'OS'.
    - returnColsFlag (bool, optional): Flag indicating whether to return the dropped columns along with the filtered dataframe. Default is True.

    Returns:
    - cols_to_drop (Index): The dropped columns (if returnColsFlag is True).
    - radiomics (DataFrame): The filtered radiomics dataframe.
    """
    cor = radiomics.corr(method='spearman')['original_shape_VoxelVolume']
    cols_to_drop = cor[abs(cor) > volThresh].index

    if cols_to_drop.isin(['original_shape_VoxelVolume']).any():
        cols_to_drop = cols_to_drop.drop('original_shape_VoxelVolume')

    if cols_to_drop.isin(['T_' + outcome]).any():
        cols_to_drop = cols_to_drop.drop('T_' + outcome)

    if cols_to_drop.isin(['E_' + outcome]).any():
        cols_to_drop = cols_to_drop.drop('E_' + outcome)

    if returnColsFlag:
        return cols_to_drop, radiomics.drop(cols_to_drop, axis=1)
    else:
        return radiomics.drop(cols_to_drop, axis=1)

def featureReduction(radiomics,varThresh=10,volThresh=0.1,outcome='OS',returnColsFlag=False,scaleFlag=False):
    """
    Perform feature reduction on radiomics data.

    Parameters:
    - radiomics (DataFrame): The input radiomics data.
    - varThresh (int): The variance threshold for feature selection (default: 10).
    - volThresh (float): The volume threshold for feature selection (default: 0.1).
    - outcome (str): The outcome variable for filtering (default: 'OS').
    - returnColsFlag (bool): Flag indicating whether to return the selected columns (default: False).
    - scaleFlag (bool): Flag indicating whether to scale the features (default: False).

    Returns:
    - df_volReduced (DataFrame): The reduced radiomics data.

    """
    df_varReduced = varianceFilter(radiomics,varThresh,outcome,returnColsFlag)
    df_volReduced = volumeFilter(df_varReduced,volThresh,outcome,returnColsFlag)
    
    if scaleFlag:
        scaledFeatures = StandardScaler().fit_transform(df_volReduced.iloc[:,1:-2])
        df_volReduced.iloc[:,1:-2] = scaledFeatures

    return df_volReduced
