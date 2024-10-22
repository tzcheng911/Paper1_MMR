#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:23:58 2023
Visualize and calculate EEG & MEG-measured MMR correlation in adults.

Correlation methods: pearson r, xcorr

@author: tzcheng
"""

import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr
from scipy import signal
import pandas as pd
import copy
import random
import mne

#%%######################################## load data MMR
# MMR
root_path='/home/tzcheng/Documents/GitHub/Paper1_MMR/data/'
times = np.linspace(-0.1,0.6,3501)

MEG_mmr = np.load(root_path + 'adults/adult_group_mmr2_pa_vector_morph.npy') # cMMR: adult_group_mmr2_vector_morph; iMMR: adult_group_mmr2_pa_vector_morph
EEG_mmr = np.load(root_path + 'adults/adult_group_mmr2_pa_eeg.npy') # cMMR: adult_group_mmr2_eeg; iMMR: adult_group_mmr2_pa_eeg

ts = 1000 # 100 ms
te = 1750 # 250 ms
cond = ['cMMR1','cMMR2','iMMR1','iMMR2']
ttcorr_vertices = [26455,25002,19843,23817] # representative vertex

#%%%% Waveform-to-Waveform pearson correlation between EEG & MEG
EEG = EEG_mmr
stc = MEG_mmr[:,:,ts:te].mean(axis=1)

r_all_s = []

for s in np.arange(0,len(stc),1):
    r,p = pearsonr(stc[s,:],EEG[s,ts:te])
    r_all_s.append(r)

print('mean abs corr between MEG_v & EEG:' + str(np.abs(r_all_s).mean()))
print('std abs corr between MEG_v & EEG:' + str(np.abs(r_all_s).std()))

#%%%% Waveform-to-Waveform xcorr between EEG & MEG
xcorr_all_s = []

for s in np.arange(0,len(MEG_mmr),1):
    a = (EEG[s,ts:te] - np.mean(EEG[s,ts:te]))/np.std(EEG[s,ts:te])
    c = (stc[s,:] - np.mean(stc[s,:]))/np.std(stc[s,:])
    
    ## the way matlab do xcorr normalization: the max is 1 if do a zero lag autocorrealtoin
    a = a / np.linalg.norm(a)
    c = c / np.linalg.norm(c)
    
    xcorr = signal.correlate(a,c)
    xcorr_all_s.append(xcorr)

lags = signal.correlation_lags(len(a),len(c))
lags_time = lags/5000

print('mean max abs xcorr between MEG & EEG:' + str(np.max(np.abs(xcorr_all_s),axis=1).mean()))
print('mean max lag of abs xcorr between MEG & EEG:' + str(lags_time[np.argmax(np.abs(xcorr_all_s),axis=1)].mean()))
print('std max abs xcorr between MEG & EEG:' + str(np.max(np.abs(xcorr_all_s),axis=1).std()))
print('std max lag of abs xcorr between MEG & EEG:' + str(lags_time[np.argmax(np.abs(xcorr_all_s),axis=1)].std()))

#%% Waveform-to-Waveform xcorr between EEG and each vertice
# Slow, consider load the pickled files from the /data folder
stc = MEG_mmr
EEG = EEG_mmr
xcorr_all_s = []
lag_all_s = []

for s in np.arange(0,len(MEG_mmr),1):
    print('Now starting sub' + str(s))
    for v in np.arange(0,np.shape(MEG_mmr)[1],1):
        a = (EEG[s,ts:te] - np.mean(EEG[s,ts:te]))/np.std(EEG[s,ts:te])
        c = (stc[s,v,ts:te] - np.mean(stc[s,v,ts:te]))/np.std(stc[s,v,ts:te])

        ## the way matlab do xcorr normalization: the max is 1 if do a zero lag autocorrealtoin
        a = a / np.linalg.norm(a)
        c = c / np.linalg.norm(c)
        
        xcorr = signal.correlate(a,c)
        xcorr_all_s.append([s,v,max(abs(xcorr))])
        lag_all_s.append([s,v,lags_time[np.argmax(abs(xcorr))]])
    
df_v = pd.DataFrame(columns = ["Subject", "Vertno","XCorr MEG & EEG"], data = xcorr_all_s)
df_v.to_pickle(root_path + 'df_xcorr_MEGEEG_iMMR2.pkl')
df_lag = pd.DataFrame(columns = ["Subject", "Vertno", "Lag XCorr MEG & EEG"], data = lag_all_s)
df_lag.to_pickle(root_path + 'df_xcorr_lag_MEGEEG_iMMR2.pkl')

#%%%% Sample-by-Sample pearson correlation between EEG & MEG
EEG = EEG_mmr
stc = MEG_mmr.mean(axis=1) 

r_all_t = []
r_all_t_v = np.zeros([np.shape(MEG_mmr)[1],np.shape(MEG_mmr)[2]])

for t in np.arange(0,len(times),1):
    r,p = pearsonr(stc[:,t],EEG[:,t])
    r_all_t.append(r)

## whole-brain ttcorr
for v in np.arange(0,np.shape(MEG_mmr)[1],1):
    print('Vertex ' + str(v))
    for t in np.arange(0,len(times),1):
        r,p = pearsonr(EEG[:,t],MEG_mmr[:,v,t])
        r_all_t_v[v,t] = r
r_all_t_v = np.asarray(r_all_t_v)
np.save(root_path + 'ttcorr_iMMR2_v.npy',r_all_t_v)

# permutation for the avg MEG
n_perm=1000
r_all_t_perm = np.zeros([n_perm,len(times)])
for i in range(n_perm):
    print('Iteration' + str(i))
    EEG_p = copy.deepcopy(EEG).transpose() # transpose to shuffle the first dimension
    stc_p = copy.deepcopy(stc).transpose()
    np.random.shuffle(EEG_p)
    np.random.shuffle(stc_p)
    
    for t in np.arange(0,len(times),1):
        r,p = pearsonr(stc_p[t,:],EEG_p[t,:])
        r_all_t_perm[i,t] = r
r_all_t_perm = np.asarray(r_all_t_perm)
np.save(root_path + 'ttcorr_iMMR2_perm1000.npy',r_all_t_perm)

## permutation for a peak vertex
subjects_dir = '/home/tzcheng/Documents/GitHub/Paper1_MMR/subjects/'
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
nv = ttcorr_vertices[3]
v_ind = np.where(src[0]['vertno'] == nv)
stc_v = np.squeeze(MEG_mmr[:,v_ind,:])
n_perm=1000
r_all_t_perm = np.zeros([n_perm,len(times)])
for i in range(n_perm):
    print('Iteration' + str(i))
    EEG_p = copy.deepcopy(EEG).transpose() # transpose to shuffle the first dimension
    stc_p = copy.deepcopy(stc_v).transpose()
    np.random.shuffle(EEG_p)
    np.random.shuffle(stc_p)
    
    for t in np.arange(0,len(times),1):
        r,p = pearsonr(stc_p[t,:],EEG_p[t,:])
        r_all_t_perm[i,t] = r
r_all_t_perm = np.asarray(r_all_t_perm)
np.save(root_path + 'ttcorr_iMMR2_perm1000_v' + str(nv) + '.npy',r_all_t_perm)
