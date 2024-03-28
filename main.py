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
import matplotlib.pyplot as plt
import seaborn as sns

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

# %% ANALYSIS

dataName = 'radcure'
aggName = 'UWA'
inclMetsFlag = False
rfFlag = True
uniFlag = False
shuffleFlag = False
numfeatures = 10

print('----------')
print(dataName, ' - ', aggName)
print('----------')

data_dict = {
                'radcure' : [radcure_radiomics,radcure_clinical],
                'sarc021' : [sarc021_radiomics,sarc021_clinical],
                'crlm'    : [crlm_radiomics,crlm_clinical]
        }

df_imaging, df_clinical = data_dict[dataName][0],data_dict[dataName][1]

if shuffleFlag:
    df_clinical['T_OS'] = df_clinical['T_OS'].sample(frac=1).reset_index(drop=True)
    df_clinical['E_OS'] = df_clinical['E_OS'].sample(frac=1).reset_index(drop=True)

if uniFlag:
    # print univariate results
    sa.univariate_CPH(df_imaging,df_clinical,mod_choice='total')
    sa.univariate_CPH(df_imaging,df_clinical,mod_choice='max')

if dataName == 'sarc021':
    train,test,val = ms.singleInstValidationSplit(df_imaging,df_clinical,0.8)
else:
    train,test = ms.randomSplit(df_imaging,df_clinical,0.8,False)
    val = np.nan

pipe_dict = {
                'train' : [train,True],
                'test'  : [test,False],
                'val'   : [val,False]
    }

func_dict = {
                'largest'  : [ls.selectLargestLesion, lambda x: fh.featureSelection(fh.featureReduction(x,numMetsFlag=inclMetsFlag,scaleFlag=True),numFeatures=numfeatures,numMetsFlag=inclMetsFlag)],
                'smallest' : [ls.selectSmallestLesion, lambda x: fh.featureSelection(fh.featureReduction(x,numMetsFlag=inclMetsFlag,scaleFlag=True),numFeatures=numfeatures,numMetsFlag=inclMetsFlag)],
                'primary'  : [ls.selectPrimaryTumor, lambda x: fh.featureSelection(fh.featureReduction(x,numMetsFlag=inclMetsFlag,scaleFlag=True),numFeatures=numfeatures,numMetsFlag=inclMetsFlag)],
                'lung'     : [ls.selectLargestLungLesion, lambda x: fh.featureSelection(fh.featureReduction(x,numMetsFlag=inclMetsFlag,scaleFlag=True),numFeatures=numfeatures,numMetsFlag=inclMetsFlag)],
                'UWA'      : [la.calcUnweightedAverage, lambda x: fh.featureSelection(fh.featureReduction(x,numMetsFlag=inclMetsFlag,scaleFlag=True),numFeatures=numfeatures,numMetsFlag=inclMetsFlag)],
                'VWA'      : [la.calcVolumeWeightedAverage, lambda x: fh.featureSelection(fh.featureReduction(x,numMetsFlag=inclMetsFlag,scaleFlag=True),numFeatures=numfeatures,numMetsFlag=inclMetsFlag)],
                'VWANLrg'  : [la.calcVolumeWeightedAverageNLargest, lambda x: fh.featureSelection(fh.featureReduction(x,numMetsFlag=inclMetsFlag,scaleFlag=True),numFeatures=numfeatures,numMetsFlag=inclMetsFlag)],
                'cosine'   : [la.calcCosineMetrics, lambda x: x],
                'concat'   : [la.concatenateNLargest, lambda x: fh.featureSelection(fh.featureReduction(x,varOnly=True))]
    }
    
# ----- TRAINING -----
df_imaging_train = df_imaging[df_imaging.USUBJID.isin(pipe_dict['train'][0])].reset_index()
df_clinical_train = df_clinical[df_clinical.USUBJID.isin(pipe_dict['train'][0])].reset_index()

print('feature selection')
trainingSet = func_dict[aggName][1](func_dict[aggName][0](df_imaging_train,df_clinical_train,numMetsFlag=inclMetsFlag).drop('USUBJID',axis=1))
print('----------')
# ----- TESTING -----
df_imaging_test = df_imaging[df_imaging.USUBJID.isin(pipe_dict['test'][0])].reset_index()
df_clinical_test = df_clinical[df_clinical.USUBJID.isin(pipe_dict['test'][0])].reset_index()

testingSet = func_dict[aggName][0](df_imaging_test,df_clinical_test,scaleFlag=True,numMetsFlag=inclMetsFlag).drop('USUBJID',axis=1)[trainingSet.columns]

best_params_CPH, scores_CPH = sa.CPH_bootstrap(trainingSet,aggName,'OS',pipe_dict['train'][1])
sa.CPH_bootstrap(testingSet,aggName,'OS',pipe_dict['test'][1],param_grid=best_params_CPH)

best_params_LAS, scores_LAS = sa.LASSO_COX_bootstrap(trainingSet,aggName,'OS',pipe_dict['train'][1])
sa.LASSO_COX_bootstrap(testingSet,aggName,'OS',pipe_dict['test'][1],param_grid=best_params_LAS)

if rfFlag:
    best_params_RSF, scores_RSF = sa.RSF_bootstrap(trainingSet,aggName,'OS',pipe_dict['train'][1])
    sa.RSF_bootstrap(testingSet,aggName,'OS',pipe_dict['test'][1],param_grid=best_params_RSF)

if dataName == 'sarc021':
    # ----- VALIDATION -----
    print('----------')
    print('validation')
    df_imaging_val = df_imaging[df_imaging.USUBJID.isin(pipe_dict['val'][0])].reset_index()
    df_clinical_val = df_clinical[df_clinical.USUBJID.isin(pipe_dict['val'][0])].reset_index()

    validationSet = func_dict[aggName][0](df_imaging_val,df_clinical_val,scaleFlag=True,numMetsFlag=inclMetsFlag).drop('USUBJID',axis=1)[trainingSet.columns]

    sa.CPH_bootstrap(validationSet,aggName,'OS',pipe_dict['val'][1],param_grid=best_params_CPH)
    sa.LASSO_COX_bootstrap(validationSet,aggName,'OS',pipe_dict['val'][1],param_grid=best_params_LAS)
    
    if rfFlag:
        sa.RSF_bootstrap(validationSet,aggName,'OS',pipe_dict['val'][1],param_grid=best_params_RSF)

if np.logical_and(aggName == 'largest',inclMetsFlag):
    aggName = 'largest+'

# save results to file
ms.add_column_to_csv('Results/'+dataName+'_CPH.csv', aggName, scores_CPH)
ms.add_column_to_csv('Results/'+dataName+'_LAS.csv', aggName, scores_LAS)
ms.add_column_to_csv('Results/'+dataName+'_RSF.csv', aggName, scores_RSF)



# %% PLOTTING/SAVING DATA

dataName = 'radcure'
# univariate results for total volume of all ROIs and OS
uni_dict = {
            'radcure' : 0.632,
            'crlm'    : 0.585,
            'sarc021' : 0.607}

all_data = pd.read_csv('Results/'+dataName+'_LAS.csv')
#all_data = all_data[['largest','largest+','smallest','UWA','VWA','VWANLrg','primary','cosine','concat']]

plt.axvline(x=uni_dict[dataName],linestyle='--',color='black')
sns.violinplot(data=all_data,orient='h',palette='dark')
plt.xlabel('Concordance Index (C-Index)')
plt.ylabel('Method')
# plt.savefig('testing.png',dpi=200,bbox_inches='tight')
plt.show()



    #%% TESTING

# outcome = 'OS'
# numFeatures = 10
# df = reduced_crlm

# dup = df.copy()

# if 'E_'+outcome in dup.columns:
#     df_surv = dup.pop('E_'+outcome)
    
# if 'T_'+outcome in dup.columns:
#     df_surv = pd.concat((df_surv,dup.pop('T_'+outcome)),axis=1)

# x = dup.copy().iloc[:,:]
# # print('x shape: ', x.shape)

# if len(df_surv.shape)>1:
#     y = Surv.from_arrays(df_surv['E_OS'],df_surv['T_OS'])
#     # print(y)
# else:
#     y = df_surv.values
#     # print(y)

# selected_features = mrmr_classif(x,y,numFeatures,relevance='rf')
# # print('selected features: ',selected_features)

# df_Selected = pd.concat([df[selected_features],df_surv],axis=1)





# %%
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
