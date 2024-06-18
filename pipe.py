#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 29 2023
Last updated Jun 18 2024

@author: caryn-geady
"""

# IMPORTS
# Change working directory to be whatever directory this script is in
import os
os.chdir(os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import ks_2samp

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

# RADCURE
radcure_radiomics = pd.read_csv('Data/RADCURE/RADCURE_radiomics.csv')
radcure_clinical = pd.read_csv('Data/RADCURE/RADCURE_clinical.csv')
# radcure_clinicalIso = radcure_clinical[np.unique(radcure_radiomics.USUBJID)[0]]

# CRLM 
crlm_radiomics = pd.read_csv('Data/TCIA-CRLM/CRLM_radiomics.csv')
crlm_clinical = pd.read_csv('Data/TCIA-CRLM/CRLM_clinical.csv')


# %% ANALYSIS

# stable
uniFlag = False
numfeatures = 10

# iterable
dataName = ['crlm']
numLesions = [3]

for dat in dataName:
    for num in numLesions:

        # adjust methods
        aggMethods = ['largest','largest+','smallest','primary','lung','VWANLrg','concat','cosine','UWA','VWA']
        
        if num == 1:
            aggMethods.remove('cosine')
        if dat in ['crlm','radcure']:
            aggMethods.remove('lung')
        if dat in ['crlm','sarc021']:
            aggMethods.remove('primary')

        print('----------')
        print('Minimum Lesions: ',str(num))


        data_dict = {
                        'radcure' : [radcure_radiomics,radcure_clinical],
                        'sarc021' : [sarc021_radiomics,sarc021_clinical],
                        'crlm'    : [crlm_radiomics,crlm_clinical]
                    }
    
        # load the data
        df_imaging, df_clinical = data_dict[dat][0],data_dict[dat][1]
        train,test = ms.randomSplit(df_imaging,df_clinical,0.8,True,False)

        # isolate patient with at least minLesions
        id_counts = df_imaging['USUBJID'].value_counts()
        valid_ids = id_counts[id_counts >= num].index
        df_imaging = df_imaging[df_imaging['USUBJID'].isin(valid_ids)].reset_index()
        df_clinical = df_clinical[df_clinical['USUBJID'].isin(valid_ids)].reset_index()

        print('Number of patients in subgroup: ',str(len(df_clinical.USUBJID)))
        print('----------')
             
        # print univariate results
        if uniFlag:
            sa.univariate_CPH(df_imaging,df_clinical,mod_choice='total')
                    
        
        pipe_dict = {
                        'train' : [train,True],
                        'test'  : [test,False]
                    }

        func_dict = {
                        'largest'  : [ls.selectLargestLesion, lambda x: fh.featureSelection(fh.featureReduction(x,scaleFlag=True),numFeatures=numfeatures)],
                        'largest+' : [ls.selectLargestLesion, lambda x: fh.featureSelection(fh.featureReduction(x,numMetsFlag=True,scaleFlag=True),numFeatures=numfeatures,numMetsFlag=True)],
                        'smallest' : [ls.selectSmallestLesion, lambda x: fh.featureSelection(fh.featureReduction(x,scaleFlag=True),numFeatures=numfeatures)],
                        'primary'  : [ls.selectPrimaryTumor, lambda x: fh.featureSelection(fh.featureReduction(x,scaleFlag=True),numFeatures=numfeatures)],
                        'lung'     : [ls.selectLargestLungLesion, lambda x: fh.featureSelection(fh.featureReduction(x,scaleFlag=True),numFeatures=numfeatures)],
                        'UWA'      : [la.calcUnweightedAverage, lambda x: fh.featureSelection(fh.featureReduction(x,scaleFlag=True),numFeatures=numfeatures)],
                        'VWA'      : [la.calcVolumeWeightedAverage, lambda x: fh.featureSelection(fh.featureReduction(x,scaleFlag=True),numFeatures=numfeatures)],
                        'VWANLrg'  : [la.calcVolumeWeightedAverageNLargest, lambda x: fh.featureSelection(fh.featureReduction(x,scaleFlag=True),numFeatures=numfeatures)],
                        'cosine'   : [la.calcCosineMetrics, lambda x: x],
                        'concat'   : [la.concatenateNLargest, lambda x: fh.featureSelection(fh.featureReduction(x,scaleFlag=True),numFeatures=numfeatures)]
                    }
            
        # ----- TRAINING -----
        df_imaging_train = df_imaging[df_imaging.USUBJID.isin(pipe_dict['train'][0])].reset_index()
        df_clinical_train = df_clinical[df_clinical.USUBJID.isin(pipe_dict['train'][0])].reset_index()
        
        for aggName in aggMethods:
            
            print(dat, ' - ', aggName)
            print('feature selection')
            if aggName in ['concat','VWANLrg','cosine']:
                trainingSet = func_dict[aggName][1](func_dict[aggName][0](df_imaging_train,df_clinical_train,numLesions=num).drop('USUBJID',axis=1))
            elif aggName == 'largest+':
                trainingSet = func_dict[aggName][1](func_dict[aggName][0](df_imaging_train,df_clinical_train,numMetsFlag=True).drop('USUBJID',axis=1))
            else:
                trainingSet = func_dict[aggName][1](func_dict[aggName][0](df_imaging_train,df_clinical_train).drop('USUBJID',axis=1))
            
            print('Features Selected: ',trainingSet.columns)
            print('----------')
            # ----- TESTING -----
            df_imaging_test = df_imaging[df_imaging.USUBJID.isin(pipe_dict['test'][0])].reset_index()
            df_clinical_test = df_clinical[df_clinical.USUBJID.isin(pipe_dict['test'][0])].reset_index()
            
            if aggName in ['concat','VWANLrg','cosine']:
                testingSet = func_dict[aggName][0](df_imaging_test,df_clinical_test,numLesions=num,scaleFlag=True).drop('USUBJID',axis=1)[trainingSet.columns]
            elif aggName == 'largest+':
                testingSet = func_dict[aggName][0](df_imaging_test,df_clinical_test,scaleFlag=True,numMetsFlag=True).drop('USUBJID',axis=1)[trainingSet.columns]
            else:
                testingSet = func_dict[aggName][0](df_imaging_test,df_clinical_test,scaleFlag=True).drop('USUBJID',axis=1)[trainingSet.columns]

            print('Training Size: ',str(len(trainingSet)))
            print('Testing Size: ',str(len(testingSet)))
            print('----------')
            
            best_params_CPH, scores_CPH = sa.CPH_bootstrap(trainingSet,aggName,'OS',pipe_dict['train'][1])
            test_CPH = sa.CPH_bootstrap(testingSet,aggName,'OS',pipe_dict['test'][1],param_grid=best_params_CPH)
            
            print('----------')
            # save results to file
            # training
            ms.add_column_to_csv('Results/Spreadsheets/'+dat+'_min'+str(num)+'_CPH_training.csv', aggName, scores_CPH)
            # testing
            ms.add_column_to_csv('Results/Spreadsheets/'+dat+'_min'+str(num)+'_CPH_testing.csv', aggName, [test_CPH])
        


# %% PLOTTING/SAVING DATA


dataName = ['radcure','sarc021','crlm']
numLesions = [1,2,3]

# univariable results for total volume of all ROIs and OS
uni_dict = {
            'radcure' : [0.626,0.632,0.636],
            'crlm'    : [0.589,0.585,0.588],
            'sarc021' : [0.609,0.607,0.569]}

for dat in dataName:
    for num in numLesions:
        # load data
        all_data = pd.read_csv('Results/Spreadsheets/'+dat+'_min'+str(num)+'_CPH_training.csv')
        test_df = pd.read_csv('Results/Spreadsheets/'+dat+'_min'+str(num)+'_CPH_testing.csv')

        if dat != 'radcure':
            all_data['primary'] = np.nan 
            test_df['primary'] = np.nan
        if dat != 'sarc021':
            all_data['lung'] = np.nan 
            test_df['lung'] = np.nan
        if num == 1:
            all_data['cosine'] = np.nan
            test_df['cosine'] = np.nan

        all_data = all_data[['largest','largest+','smallest','primary','lung','VWANLrg','concat','cosine','UWA','VWA']]
        all_data.columns = ['Largest','Largest+','Smallest','Primary','Lung','VWA N-largest','Concatenation','Cosine Similarity','UWA','VWA']
        test_df = test_df[['largest','largest+','smallest','primary','lung','VWANLrg','concat','cosine','UWA','VWA']]
        test_df.columns = ['Largest','Largest+','Smallest','Primary','Lung','VWA N-largest','Concatenation','Cosine Similarity','UWA','VWA']

        # plotting params
        my_pal = ['#4daf4a','#4daf4a','#4daf4a','#4daf4a','#4daf4a','#ff7f00','#ff7f00','#ff7f00','#377eb8','#377eb8']
        plt.rcParams.update({'font.size': 18})
        plt.rcParams["font.family"] = "Avenir"
        
        plt.axvline(x=uni_dict[dat][num-1],linestyle='--',color='k')
        plt.axvline(x=0.5,linestyle='--',color='lightgray')
        ax = sns.violinplot(data=all_data,orient='h',palette=my_pal)
        sns.stripplot(data=test_df,orient='h',edgecolor='k', linewidth=1, palette=['white'] * 4,ax=ax)
        
        # Modify the legend
        legend_elements = [Line2D([0], [0], linestyle='--', color='lightgrey', label='Random'),
                           Line2D([0], [0], linestyle='--', color='k', label='Total Volume'),
                           Line2D([0], [0], marker='s', color='w', label='Lesion Selection', markeredgecolor='k',markerfacecolor='#4daf4a', markersize=10,),
                           Line2D([0], [0], marker='s', color='w', label='Information from Select Lesions', markeredgecolor='k',markerfacecolor='#ff7f00', markersize=10),
                           Line2D([0], [0], marker='s', color='w', label='Information from All Lesions', markeredgecolor='k',markerfacecolor='#377eb8', markersize=10),
                           Line2D([0], [0], marker='o', color='w', label='Testing Data', markeredgecolor='k',markerfacecolor='w', markersize=8)]
        
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=14)
        
        plt.xlabel('Concordance Index (C-Index)')
        plt.xlim([0.35,1])
        plt.ylabel('Method')
        plt.title(dat+' - '+str(num)+('+ lesions'))
        plt.savefig('Results/Figures/'+dat+'_min'+str(num)+'_CPH.png',dpi=300,bbox_inches='tight')
        plt.show()


# %% QUANTITATIVE STUFF FOR RESULTS / DISCUSSION

# data1 = 'sarc021'
# data2 = 'sarc021'
# num1 = 2
# num2 = 3

# file1 = data1+'_min'+str(num1)+'_CPH_training.csv'
# file2 = data2+'_min'+str(num2)+'_CPH_training.csv'

# df1 = pd.read_csv('Results/Spreadsheets/'+file1)
# df2 = pd.read_csv('Results/Spreadsheets/'+file2)

# pvals = []
# med1 = []
# med2 = []

# for c in df1.columns:
    
#     pvals.append(ks_2samp(df1[c],df2[c]).pvalue)
#     med1.append(df1[c].median())
#     med2.append(df2[c].median())

# pvals
# # %%
# np.min(np.array(med2)-np.array(med1))
# np.array(med2)-np.array(med1)
