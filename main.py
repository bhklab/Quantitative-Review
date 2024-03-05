#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 29 2023
Last updated Feb 23 2024

@author: caryn-geady
"""



# %% IMPORTS
# Change working directory to be whatever directory this script is in
import os
os.chdir(os.path.dirname(__file__))
import numpy as np, pandas as pd

# custom functions
import Code.misc_splitting as ms
import Code.lesion_selection as ls
import Code.lesion_aggregation as la
import Code.feature_handling as fh
import Code.survival_analysis as sa
     

# %% DATA LOADING

# SARC021
sarc021_radiomics = pd.read_csv('Data/SARC021/SARC021_radiomics.csv')
sarc021_clinical = pd.read_csv('Data/SARC021/SARC021_clinical.csv')

# optional ? -- look only at those patients with 2+ lesions contoured
id_counts = sarc021_radiomics['USUBJID'].value_counts()
valid_ids = id_counts[id_counts >= 2].index
sarc021_radiomics = sarc021_radiomics[sarc021_radiomics['USUBJID'].isin(valid_ids)].reset_index()

# RADCURE
radcure_radiomics = pd.read_csv('Data/RADCURE/RADCURE_radiomics.csv')
radcure_clinical = pd.read_csv('Data/RADCURE/RADCURE_clinical.csv')

# CRLM 
crlm_radiomics = pd.read_csv('Data/TCIA-CRLM/CRLM_radiomics.csv')
crlm_clinical = pd.read_csv('Data/TCIA-CRLM/CRLM_clinical.csv')


del(id_counts,valid_ids)

# %% TESTING


aggName = 'largest'
inclMetsFlag = False

# RADCURE
# df_imaging = radcure_radiomics
# df_clinical = radcure_clinical

# TCIA-CRLM
df_imaging = crlm_radiomics
df_clinical = crlm_clinical

train,test = ms.randomSplit(df_imaging,df_clinical,0.8,False)
val = np.nan

# SARC021
# df_imaging = sarc021_radiomics
# df_clinical = sarc021_clinical
# train,test,val = ms.singleInstValidationSplit(df_imaging,df_clinical,0.8)

pipe_dict = {
                'train' : [train,True],
                'test'  : [test,False],
                'val'   : [val,False]
    }

func_dict = {
                'largest'  : [ls.selectLargestLesion, lambda x: fh.featureSelection(fh.featureReduction(x,numMetsFlag=inclMetsFlag,scaleFlag=True),numFeatures=10,numMetsFlag=inclMetsFlag)],
                'smallest' : [ls.selectSmallestLesion, lambda x: fh.featureSelection(fh.featureReduction(x,numMetsFlag=inclMetsFlag,scaleFlag=True),numFeatures=10,numMetsFlag=inclMetsFlag)],
                'primary'  : [ls.selectPrimaryTumor, lambda x: fh.featureSelection(fh.featureReduction(x,numMetsFlag=inclMetsFlag,scaleFlag=True),numFeatures=10,numMetsFlag=inclMetsFlag)],
                'lung'     : [ls.selectLargestLungLesion, lambda x: fh.featureSelection(fh.featureReduction(x,numMetsFlag=inclMetsFlag,scaleFlag=True),numFeatures=10,numMetsFlag=inclMetsFlag)],
                'UWA'      : [la.calcUnweightedAverage, lambda x: fh.featureSelection(fh.featureReduction(x,numMetsFlag=inclMetsFlag,scaleFlag=True),numFeatures=10,numMetsFlag=inclMetsFlag)],
                'VWA'      : [la.calcVolumeWeightedAverage, lambda x: fh.featureSelection(fh.featureReduction(x,numMetsFlag=inclMetsFlag,scaleFlag=True),numFeatures=10,numMetsFlag=inclMetsFlag)],
                'VWANLrg'  : [la.calcVolumeWeightedAverageNLargest, lambda x: fh.featureSelection(fh.featureReduction(x,numMetsFlag=inclMetsFlag,scaleFlag=True),numFeatures=10,numMetsFlag=inclMetsFlag)],
                'cosine'   : [la.calcCosineMetrics, lambda x: x],
                'concat'   : [la.concatenateNLargest, lambda x: fh.featureSelection(fh.featureReduction(x,varOnly=True))]
    
    }
    
# ----- TRAINING SET -----
# isolate the patients in the defined split (i.e., train/test/val)
df_imaging_train = df_imaging[df_imaging.USUBJID.isin(pipe_dict['train'][0])].reset_index()
df_clinical_train = df_clinical[df_clinical.USUBJID.isin(pipe_dict['train'][0])].reset_index()

trainingSet = func_dict[aggName][1](func_dict[aggName][0](df_imaging_train,df_clinical_train,numMetsFlag=inclMetsFlag).drop('USUBJID',axis=1))
#debugging = func_dict[aggName][0](df_imaging_train,df_clinical_train,numMetsFlag=inclMetsFlag).drop('USUBJID',axis=1)
# debugging = fh.featureReduction(func_dict[aggName][0](df_imaging_train,df_clinical_train,numMetsFlag=inclMetsFlag).drop('USUBJID',axis=1),numMetsFlag=inclMetsFlag,scaleFlag=True)

# 
# ----- TESTING SET -----
# isolate the patients in the defined split (i.e., train/test/val)
df_imaging_test = df_imaging[df_imaging.USUBJID.isin(pipe_dict['test'][0])].reset_index()
df_clinical_test = df_clinical[df_clinical.USUBJID.isin(pipe_dict['test'][0])].reset_index()

testingSet = func_dict[aggName][0](df_imaging_test,df_clinical_test,scaleFlag=True,numMetsFlag=inclMetsFlag).drop('USUBJID',axis=1)[trainingSet.columns]


best_params_CPH = sa.CPH_bootstrap(trainingSet,aggName,'OS',pipe_dict['train'][1])
sa.CPH_bootstrap(testingSet,aggName,'OS',pipe_dict['test'][1],param_grid=best_params_CPH)

best_params_LAS = sa.LASSO_COX_bootstrap(trainingSet,aggName,'OS',pipe_dict['train'][1])
sa.LASSO_COX_bootstrap(testingSet,aggName,'OS',pipe_dict['test'][1],param_grid=best_params_LAS)

# %%

# best_params_RSF = sa.RSF_bootstrap(trainingSet,aggName,'OS',pipe_dict['train'][1])
# sa.RSF_bootstrap(testingSet,aggName,'OS',pipe_dict['test'][1],param_grid=best_params_RSF)

# %
# ----- SARC021 SINGLE INSTITUTION VALIDATION -----
# isolate the patients in the defined split (i.e., train/test/val)
df_imaging_val = df_imaging[df_imaging.USUBJID.isin(pipe_dict['val'][0])].reset_index()
df_clinical_val = df_clinical[df_clinical.USUBJID.isin(pipe_dict['val'][0])].reset_index()

validationSet = func_dict[aggName][0](df_imaging_val,df_clinical_val,scaleFlag=True,numMetsFlag=inclMetsFlag).drop('USUBJID',axis=1)[trainingSet.columns]
# validationSet = calcCosineMetrics(df_imaging_val,df_clinical_val)
sa.CPH_bootstrap(validationSet,aggName,'OS',pipe_dict['val'][1],param_grid=best_params_CPH)
sa.LASSO_COX_bootstrap(validationSet,aggName,'OS',pipe_dict['val'][1],param_grid=best_params_LAS)


	

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



# %% 

# def featureSelection(df,numFeatures=10,numMetsFlag=False,volFlag=False,scaleFlag=False):
    
#     if numMetsFlag:
#         numFeatures -= 1
#         solutions = mrmr.mrmr_ensemble_survival(features=df.iloc[:,2:-2],
#                                                 targets=df.iloc[:,-2:],
#                                                 solution_length=numFeatures)[0]
#         df_Selected = df.copy()[solutions[0]+["NumMets","T_OS","E_OS"]]
        
#     else:
#         solutions = mrmr.mrmr_ensemble_survival(features=df.iloc[:,1:-2],
#                                             targets=df.iloc[:,-2:],
#                                             solution_length=numFeatures)[0]
#         df_Selected = df.copy()[solutions[0]+["T_OS","E_OS"]]
      
#     if volFlag and df.columns.isin(['original_shape_VoxelVolume']).any():
#         numFeatures -= 1
#         solutions = mrmr.mrmr_ensemble_survival(features=df.iloc[:,2:-2],
#                                                 targets=df.iloc[:,-2:],
#                                                 solution_length=numFeatures)[0]
#         df_Selected = df.copy()[solutions[0]+["original_shape_VoxelVolume","T_OS","E_OS"]]
        
#     else:
#         solutions = mrmr.mrmr_ensemble_survival(features=df.iloc[:,1:-2],
#                                             targets=df.iloc[:,-2:],
#                                             solution_length=numFeatures)[0]
#         df_Selected = df.copy()[solutions[0]+["T_OS","E_OS"]]
    
    
#     if scaleFlag:
#         scaledFeatures = StandardScaler().fit_transform(df_Selected.iloc[:,:-2])
#         df_Selected.iloc[:,:-2] = scaledFeatures
    
#     return df_Selected
