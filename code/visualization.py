#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:31:16 2023

Used to visualize MMR paper Figure 3 - 5. 

@author: tzcheng
"""

import mne
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import pearsonr
from scipy import stats
import pandas as pd

def plot_err(group_stc,color,t):
    """Plot the mean with shaded area of std
    
    Keyword arguments:
    group_stc -- whatever time series across subjects
    color -- the line and shade color
    t -- the time series on the x-axis
    """
    group_avg=np.mean(group_stc,axis=0)
    err=np.std(group_stc,axis=0)/np.sqrt(group_stc.shape[0])
    up=group_avg+err
    lw=group_avg-err
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)

#%%####################################### Load the files for any of the four conditions: adults, infants, cMMR, iMMR
root_path='/home/tzcheng/Documents/GitHub/Paper1_MMR/data/'
subjects_dir = '/home/tzcheng/Documents/GitHub/Paper1_MMR/subjects/'
subject = 'fsaverage'
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
times = np.linspace(-0.1,0.6,3501)

cond = ['cMMR1','cMMR2','iMMR1','iMMR2']
wwcorr_vertices = [25102,19960,22434,26262] # representative vertex
ttcorr_vertices = [26455,25002,19843,23817] # representative vertex

MEG_mmr = np.load(root_path + 'adults/adult_group_mmr2_pa_vector_morph.npy') # cMMR: adults/adult_group_mmr2_vector_morph.npy; iMMR: adults/adult_group_mmr2_pa_vector_morph.npy
EEG_mmr = np.load(root_path + 'adults/adult_group_mmr2_pa_eeg.npy') # cMMR: adults/adult_group_mmr2_eeg.npy; iMMR: adults/adult_group_mmr2_pa_eeg.npy

#%%####################################### Figure3ab Waveform-to-Waveform Cross-Correlation 
# change xcorr and nv for corresponding condition
xcorr = pd.read_pickle(root_path + 'adults/df_xcorr_MEGEEG_iMMR2.pkl') # cMMR: df_xcorr_MEGEEG_cMMR2; iMMR: df_xcorr_MEGEEG_iMMR2
xcorr_mean = xcorr.groupby('Vertno').mean()

## Visualize whole-brain
v_hack = pd.concat([xcorr_mean["XCorr MEG & EEG"],xcorr_mean["XCorr MEG & EEG"]],axis=1)
stc = mne.read_source_estimate(subjects_dir + 'cbs_A101_mmr2_vector_morph-vl.stc')
stc.data = v_hack
stc.plot(src,clim=dict(kind="percent",lims=[90,95,99]),subject=subject, subjects_dir=subjects_dir)

## Visualize the EEG and MEG waveform of 1 hot spot
nv = wwcorr_vertices[3] # 0 to 3 
v_ind = np.where(src[0]['vertno'] == nv)

plt.figure()
plot_err(stats.zscore(EEG_mmr,axis=1),'k',times)
plot_err(stats.zscore(MEG_mmr[:,v_ind[0][0],:],axis=1),'r',times)
plt.legend(['EEG','','MEG',''])
plt.xlabel('Time (s)')
plt.title('' + str(nv))
plt.xlim([-0.05, 0.45])
plt.ylim([-1.5, 1.5])

#%%####################################### Figure3cd Sample-by-Sample Pearson Correlation 
# change ttcorr and nv for corresponding condition
ttcorr = np.load(root_path + 'adults/ttcorr_iMMR2_v.npy') # cMMR: ttcorr_cMMR2_v; iMMR: ttcorr_iMMR2_v
ttcorr_perm = np.load(root_path + 'adults/ttcorr_iMMR2_perm1000_v23817.npy') 
# cMMR: ttcorr_cMMR2_perm1000_v25002 (peak vertex), ttcorr_cMMR2_perm1000 (source-averaged)
# iMMR: ttcorr_iMMR2_perm1000_v23817 (peak vertex), ttcorr_iMMR2_perm1000 (source-averaged)

stc = mne.read_source_estimate(subjects_dir + 'cbs_A101_mmr2_vector_morph-vl.stc')
stc.data = ttcorr
stc.plot(src=src,subject=subject, clim=dict(kind="percent", pos_lims = (95,97.5,99.975)),subjects_dir=subjects_dir) # or set your own threshold

# [wholebrain MEG] Plot the sample-by-sample Pearson correlation for each time point
nv = ttcorr_vertices[3]
v_ind = np.where(src[0]['vertno'] == nv)

fig, ax = plt.subplots(1)
ax.plot(times, ttcorr[v_ind[0][0],:])
ax.axhline(0, color="k", linestyle="--")
ax.axvline(0, color="k")
plt.title('MMR Correlation')
plt.legend(['MEG v' + str(nv)])
plt.xlabel('Time (s)')
plt.ylabel('Pearson r')
plt.xlim([-0.1,0.45])
plt.ylim([-1,1])
plt.plot(times, np.percentile(ttcorr_perm,97.5, axis= 0))
plt.plot(times, np.percentile(ttcorr_perm,2.5, axis= 0))
print("largest corr: " + str(np.min(ttcorr[v_ind[0][0],:])))

# [wholebrain MEG] Plot the scatterplot at certain time points
nt = 1475 # or 900
plt.figure()
plt.scatter(MEG_mmr[:,v_ind,nt],EEG_mmr[:,nt])  # look up the corresponding sample points from times
plt.xlabel('MEG v' + str(nv))
plt.ylabel('EEG')
plt.title('t = 0.195 s') # change accordingly based on nt

#%%####################################### Figure4ab
MEG_mmr1 = np.load(root_path + 'adults/adult_group_mmr1_mba_vector_morph.npy') # cMMR: adults/adult_group_mmr1_vector_morph.npy; iMMR: adults/adult_group_mmr1_mba_vector_morph.npy
MEG_mmr2 = np.load(root_path + 'adults/adult_group_mmr2_pa_vector_morph.npy') # cMMR: adults/adult_group_mmr2_vector_morph.npy; iMMR: adults/adult_group_mmr2_pa_vector_morph.npy
plt.figure()
plot_err(MEG_mmr1.mean(axis=1),'orange',times)
plot_err(MEG_mmr2.mean(axis=1),'green',times)
plt.legend(['Nonnative source-averaged MMR','','Native source-averaged MMR',''])
plt.xlabel('Time (s)')
plt.xlim([-0.05, 0.45])

#%%####################################### Figure6ab
MEG_mmr1 = np.load(root_path + 'infants/baby_group_mmr1_mba_vector_morph.npy') # cMMR: infants/baby_group_mmr1_vector_morph.npy; iMMR: infants/baby_group_mmr1_mba_vector_morph.npy
MEG_mmr2 = np.load(root_path + 'infants/baby_group_mmr2_pa_vector_morph.npy') # cMMR: infants/baby_group_mmr2_vector_morph.npy; iMMR: infants/baby_group_mmr2_pa_vector_morph.npy
plt.figure()
plot_err(MEG_mmr1.mean(axis=1),'orange',times)
plot_err(MEG_mmr2.mean(axis=1),'green',times)
plt.legend(['Nonnative source-averaged MMR','','Native source-averaged MMR',''])
plt.xlabel('Time (s)')
plt.xlim([-0.05, 0.45])

#%%####################################### Figure5abcd adults and Figure7abcd infants Decoding analysis 
## Figure5ab cMMR: ba to mba vs. ba to pa
scores_observed = np.load(root_path + 'adults/adult_scores_conv_morph_kall.npy')
patterns = np.load(root_path +'adults/adult_patterns_conv_morph_kall.npy')
scores_permute = np.load(root_path +'adults/adult_vector_scores_100perm_kall_conv.npz')

## Figure5cd iMMR: first - last mba vs. first pa - last pa
scores_observed = np.load(root_path + 'adults/adult_scores_cont_morph_kall.npy')
patterns = np.load(root_path + 'adults/adult_patterns_cont_morph_kall.npy')
scores_permute = np.load(root_path + 'adults/adult_vector_scores_100perm_kall_cont.npz')

## Figure7ab cMMR: ba to mba vs. ba to pa
scores_observed = np.load(root_path + 'infants/baby_scores_conv_morph_kall.npy')
patterns = np.load(root_path +'infants/baby_patterns_conv_morph_kall.npy')
scores_permute = np.load(root_path +'infants/baby_vector_scores_100perm_kall_conv.npz')

## Figure7cd iMMR: first - last mba vs. first pa - last pa
scores_observed = np.load(root_path + 'infants/baby_scores_cont_morph_kall.npy')
patterns = np.load(root_path + 'infants/baby_patterns_cont_morph_kall.npy')
scores_permute = np.load(root_path + 'infants/baby_vector_scores_100perm_kall_cont.npz')

## Plot decoding accuracy across time
fig, ax = plt.subplots(1)
ax.plot(times[250:2750], scores_observed.mean(0), label="score")
ax.plot(scores_permute['peaks_time'],np.percentile(scores_permute['scores_perm_array'],95,axis=0),'g.')
ax.axhline(0.5, color="k", linestyle="--", label="chance")
ax.axhline(np.percentile(scores_observed.mean(0),q = 95), color="grey", linestyle="--", label="95 percentile")
ax.axvline(0, color="k")
plt.xlabel('Time (s)')
plt.xlim([-0.05,0.45])
plt.ylim([0,1])

## 1000 iterations 
# scores_permute = np.load(root_path +'adults/roc_auc_1000perm_kall.npz')
# ax.plot(scores_permute['peaks_time'][scores_permute['peaks_time']<0.45],np.percentile(scores_permute['scores_perm_array'][:,scores_permute['peaks_time']<0.45],95,axis=0),'g.')

## Plot patterns
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
stc = mne.read_source_estimate(subjects_dir + 'cbs_A101_mmr2_vector_morph-vl.stc')
stc_crop = stc.copy().crop(tmin= -0.05, tmax=0.45)
stc_crop.data = patterns
stc_crop.plot(src, clim=dict(kind="percent",pos_lims=[90,95,99]), subject='fsaverage', subjects_dir=subjects_dir)

#%%####################################### Supplementary Figure 1 Visualize the EEG time series
first_ba = np.load(root_path + 'adults/adult_group_dev_reverse_eeg.npy')
last_mba = np.load(root_path + 'adults/adult_group_std1_reverse_eeg.npy')
last_pa = np.load(root_path + 'adults/adult_group_std2_reverse_eeg.npy')
last_ba = np.load(root_path + 'adults/adult_group_std_eeg.npy')
first_mba = np.load(root_path + 'adults/adult_group_dev1_eeg.npy')
first_pa = np.load(root_path + 'adults/adult_group_dev2_eeg.npy')

## Conventional calculation
cMMR1 = first_mba - last_ba
cMMR2 = first_pa - last_ba

plt.figure()
plot_err(last_ba*1e6,'k',times)
plot_err(first_mba*1e6,'r',times)
plot_err(first_pa*1e6,'b',times)
plt.title('cMMR calculation')
plt.legend(['std last_ba','','dev1 first_mba','','dev2 first_pa',''])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.xlim([-0.05,0.45])
plt.ylim([-5,5])

plt.figure()
plot_err(cMMR1*1e6,'orange',times)
plot_err(cMMR2*1e6,'green',times)
plt.title('cMMR')
plt.legend(['Nonnative cMMR','','Native cMMR',''])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.xlim([-0.05,0.45])
plt.ylim([-5,5])

## Reversed calculation
rMMR1 = first_ba - last_mba
rMMR2 = first_ba - last_pa

plt.figure()
plot_err(last_ba*1e6,'k',times)
plot_err(first_mba*1e6,'r',times)
plot_err(first_pa*1e6,'b',times)
plt.title('rMMR calculation')
plt.legend(['std last_ba','','dev1 first_mba','','dev2 first_pa',''])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.xlim([-0.05,0.45])
plt.ylim([-5,5])

plt.figure()
plot_err(rMMR1*1e6,'orange',times)
plot_err(rMMR2*1e6,'green',times)
plt.title('rMMR')
plt.legend(['Nonnative rMMR','','Native rMMR',''])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.xlim([-0.05,0.45])
plt.ylim([-5,5])

## Identity calculation
iMMR1 = first_mba - last_mba
iMMR2 = first_pa - last_pa

plt.figure()
plot_err(last_mba*1e6,'m',times)
plot_err(first_mba*1e6,'r',times)
plot_err(last_pa*1e6,'c',times)
plot_err(first_pa*1e6,'b',times)
plt.title('iMMR calculation')
plt.legend(['control1 last_mba','','deviant 1 first_mba','','control2 last_pa','','deviant 2 first_pa',''])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.xlim([-0.05,0.45])
plt.ylim([-5,5])

plt.figure()
plot_err(iMMR1*1e6,'orange',times)
plot_err(iMMR2*1e6,'green',times)
plt.title('iMMR')
plt.legend(['Nonnative iMMR','','Native iMMR',''])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.xlim([-0.05,0.45])
plt.ylim([-5,5])

#%%####################################### Supplementary Figure 2 See the ttcorr and wwcorr plot up