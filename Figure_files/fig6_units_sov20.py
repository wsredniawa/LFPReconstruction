# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:02:31 2020

@author: Wladek
"""

import numpy as np 
import matplotlib.pyplot as py

from matplotlib.colors import LogNorm
from figure_properties import *
import matplotlib.patches as mpatches
from scipy.signal import welch, argrelmin, butter, filtfilt, spectrogram, detrend
from scipy.stats import wilcoxon, f_oneway, ttest_rel, sem, shapiro
import scipy.io
# from mlxtend.evaluate import permutation_test
py.close('all')
import matplotlib as mpl
# import sarna as srn
import random
from IPython import get_ipython
import h5py
get_ipython().run_line_magic('matplotlib', 'inline')
# mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['xtick.major.size'] = 1
mpl.rcParams['ytick.major.size'] = 1

def spike_heatmap(spikes, x=None, log=False):
    from matplotlib.pyplot import cm
    cmap = cm.Blues
    cmaplist = [cmap(i) for i in range(int(cmap.N/4), cmap.N)]
    cmaplist[0],cmaplist[-1] = (1, 1, 1, 1),(0, 0, 0, 1)
    
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    spMin, spMax = spikes.min(),spikes.max()
    spBins = np.linspace(spMin, spMax, int(round(2*spMax)))
    if spBins.shape[0] < 3: spBins = np.linspace(spMin, spMax, 3)
    nSamp = spikes.shape[1]
    if x is None: x = range(nSamp)
    imdata = np.zeros((len(spBins) - 1, nSamp))
    for col in range(nSamp):
        data = np.histogram(spikes[:, col], bins=spBins)[0]
        if log: imdata[:, col] = np.log(1 + data)
        else: imdata[:, col] = data
    ydiff = (spBins[1] - spBins[0])/2.
    extent = [x[0], x[-1], spMin-ydiff, spMax-ydiff]
    return imdata, extent, cmap

def load_spikes(name, path, lid='',polarity = 'neg'):
    global f_pos,groups,klasy
    
    f = h5py.File(path+'\\'+name+lid+'\\data_'+name+lid+'.h5', 'r')
    pos_times,waves = np.asarray(f[polarity]['times']),np.asarray(f[polarity]['spikes'])
    f_pos = h5py.File(path+'\\'+name+lid+'\\sort_'+polarity+'_simple\\sort_cat.h5', 'r')
    spikes, klasy = np.asarray(f_pos['index']), np.asarray(f_pos['classes'])
    sort_pos,sort_waves = pos_times[spikes],waves[spikes]
    groups = np.asarray(f_pos['groups'][:,1])
    print(groups)
    list_groups = np.unique(groups)
    units = []
    units_g = {k: [] for k in list_groups}
    units_w = {k: [] for k in list_groups}
    for i,cl in enumerate(f_pos['groups'][:,0]):
        sub = np.where(klasy==cl)[0]
        units.append([sort_pos[sub], f_pos['groups'][i][1], sort_waves[sub]])
    for un in units:
        units_g[un[1]] = units_g[un[1]] + list(un[0])
        units_w[un[1]] = units_w[un[1]] + list(un[2])
    return units_g,units_w

def extract_spikes(name, path, lid='',polarity = 'neg'):
    global f_pos,groups,klasy
    
    f = h5py.File(path+'\\'+name+lid+'\\data_'+name+lid+'.h5', 'r')
    pos_times,waves = np.asarray(f[polarity]['times']),np.asarray(f[polarity]['spikes'])
    f_pos = h5py.File(path+'\\'+name+lid+'\\sort_'+polarity+'_simple\\sort_cat.h5', 'r')
    spikes, klasy = np.asarray(f_pos['index']), np.asarray(f_pos['classes'])
    sort_pos,sort_waves = pos_times[spikes],waves[spikes]
    groups = np.asarray(f_pos['groups'][:,1])
    print(groups)
    list_groups = np.unique(groups)
    units = []
    units_g = {k: [] for k in list_groups}
    units_w = {k: [] for k in list_groups}
    for i,cl in enumerate(f_pos['groups'][:,0]):
        sub = np.where(klasy==cl)[0]
        units.append([sort_pos[sub], f_pos['groups'][i][1], sort_waves[sub]])
    for un in units:
        units_g[un[1]] = units_g[un[1]] + list(un[0])
        units_w[un[1]] = units_w[un[1]] + list(un[2])
    return units_g,units_w

def exp_design(po, pos, name):
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    img = py.imread(name)
    ax.imshow(img, aspect='auto', extent=[0,1,0,1])
    # py.xlim(0.1,0.9)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    py.xticks([]), py.yticks([])

def ex_lfp2(po, pos, name1,name2, channel=255,div=2, Fs=5000):
    global recon_th,mn_plid
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, -.04, po[1], letter= po[0])
    pots = scipy.io.loadmat(name1)['pots']
    mn_pots,std_pots = pots.mean(axis=1), sem(pots,axis=1)
    recon_th = scipy.io.loadmat(name1)['pots_est_th']
    mn_rth,std_rth = np.mean(recon_th, axis=0), sem(recon_th,axis=0)
    potslid = scipy.io.loadmat(name2)['pots']
    mn_plid,std_plid = np.mean(potslid, axis=1), sem(potslid,axis=1)
    recon_thlid = scipy.io.loadmat(name2)['pots_est_th']
    mn_rlid,std_rlid = np.mean(recon_thlid, axis=0), sem(recon_thlid,axis=0)
    # mn_pots[channel] -= np.mean(mn_pots[channel,:25])
    # mn_rth[channel] -= np.mean(mn_rth[channel,:25])
    # mn_plid[channel] -= np.mean(mn_plid[channel,:25])
    # mn_rlid[channel] -= np.mean(mn_rlid[channel,:25])
    mn_pots[:,30:60] = 0
    mn_rth[:,30:60] = 0
    mn_plid[:,30:60] = 0
    mn_rlid[:,30:60] = 0
    
    loc_time = np.linspace(-5,25, pots.shape[2])
    py.plot(loc_time, mn_pots[channel], color='grey',label='control')
    # py.fill_between(loc_time, mn_pots[channel]-std_pots[channel],mn_pots[channel]+std_pots[channel], 
                    # color='grey',alpha=.3)
    py.plot(loc_time, mn_rth[channel], color='black',label='recon. control', ls='--')
    # py.fill_between(loc_time, mn_rth[channel]-std_rth[channel],mn_rth[channel]+std_rth[channel], 
                    # color='black',alpha=.3)
    py.plot(loc_time, mn_plid[channel], color='red',label='ligno.')
    # py.fill_between(loc_time, mn_plid[channel]-std_plid[channel],mn_plid[channel]+std_plid[channel], 
                    # color='red',alpha=.3)
    py.plot(loc_time, mn_rlid[channel], color='darkred',label='recon. ligno.' ,ls='--')
    # py.fill_between(loc_time, mn_rlid[channel]-std_rlid[channel],mn_rlid[channel]+std_rlid[channel], 
                    # color='darkred',alpha=.3)
    ax.set_ylabel('Potential (mV)')
    ax.spines['right'].set_visible(False)
    # py.legend(ncol=2,loc=4, frameon = False, fontsize = 14)
    py.xlim(-5,25),py.xticks([]),#py.ylim(-5,5)  
    
def ex_lfp_trials(po, pos, name1,name2, channel=255,div=2, Fs=5000, trials = 30):
    global recon_th
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, -.04, po[1], letter= po[0])
    for i,trial in enumerate(trials):
        pots = scipy.io.loadmat(name1)['pots']
        mn_pots,std_pots = pots[:,trial], sem(pots,axis=1)
        recon_th = scipy.io.loadmat(name1)['pots_est_th']
        mn_rth,std_rth = recon_th[trial], sem(recon_th,axis=0)
        potslid = scipy.io.loadmat(name2)['pots']
        mn_plid,std_plid = potslid[:,trial], sem(potslid,axis=1)
        recon_thlid = scipy.io.loadmat(name2)['pots_est_th']
        mn_rlid,std_rlid = recon_thlid[trial], sem(recon_thlid,axis=0)
        mn_pots[:,40:70] = 0
        mn_rth[:,40:70] = 0
        mn_plid[:,40:70] = 0
        mn_rlid[:,40:70] = 0
        loc_time = np.linspace(-5,25, pots.shape[2])
        if i==0: 
            py.plot(loc_time, mn_pots[channel], color='grey',label='control')
            py.plot(loc_time, mn_rth[channel], color='black',label='recon. control', ls='--')
            py.plot(loc_time, mn_plid[channel], color='red',label='ligno.')
            py.plot(loc_time, mn_rlid[channel], color='darkred',label='recon. ligno.' ,ls='--')
        else:
            py.plot(loc_time, mn_pots[channel], color='grey')
            py.plot(loc_time, mn_rth[channel], color='black', ls='--')
            py.plot(loc_time, mn_plid[channel], color='red')
            py.plot(loc_time, mn_rlid[channel], color='darkred',ls='--')
            
    ax.set_ylabel('Potential (mV)'),ax.set_xlabel('Time (s)')
    ax.spines['right'].set_visible(False)
    py.legend(ncol=4,loc=2, frameon = False, fontsize = 13)
    py.xlim(-5,25)#,py.xticks([]),
    # py.ylim(-8,5)  
    size=20
    py.xlim(-5,25),py.xlabel('Time (ms)', fontsize=size)
    py.xticks(fontsize = size), py.yticks(fontsize = size)
    
def lfp_profile(po, pos, name, title, vmax=200, scale=40, size=20):
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    lfp = scipy.io.loadmat(name)['pots']
    ele_pos = scipy.io.loadmat('../data/sov19.mat')['ele_pos'].T
    lfp = lfp.mean(axis=1)
    py.title(title, fontsize=size)
    time= np.linspace(-5,25,lfp.shape[1])
    for i in range(lfp.shape[0]-1,0,-1):
        py.plot(time, lfp[i]/scale+ele_pos[i,2]-11, color='black', linewidth=.5)
        if i==channel:
            py.plot(time, lfp[i]/scale+ele_pos[i,2]-11, color='black', linewidth=3)   
    py.axvline(0, ls='--', color='grey'), py.ylabel('Electrode number')
    py.ylim(-7.5,1), py.ylabel('Depth (mm)', fontsize=size)
    py.xlabel('Time (ms)', fontsize=size)
    ax.spines['right'].set_visible(False)
    py.xticks(fontsize = size), py.yticks(fontsize = size)
    py.axvline(0, ls='--', lw = 2, color='grey')
    if po[0]=='B':
        py.yticks([]),py.ylabel('')
        py.plot([25.8,25.8],[0,-1], 'k',lw=3)
        py.text(26.4,-0.2, '1 mV', size=15)
    
def comparison(po, pos, name1,name2, channel=255,div=2, Fs=5000, n_iter=1000):
    global recon_th,stat,pvals,all_indexes,worek
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, -.04, po[1], letter= po[0])
    pots = scipy.io.loadmat(name1)['pots']
    # mn_pots,std_pots = pots.mean(axis=1), sem(pots,axis=1)
    rth = scipy.io.loadmat(name1)['pots_est_th']
    # mn_rth,std_rth = np.mean(recon_th, axis=0), sem(recon_th,axis=0)
    plid = scipy.io.loadmat(name2)['pots']
    # mn_plid,std_plid = np.mean(potslid, axis=1), sem(potslid,axis=1)
    rlid = scipy.io.loadmat(name2)['pots_est_th']
    p_values=np.zeros((3, pots.shape[-1]))
    loc_time = np.linspace(-5,25, pots.shape[-1])
    pvals_resamp = np.zeros((3,n_iter,pots.shape[-1]))
    worek = np.concatenate((pots[channel], rth[:,channel]),axis=0)
    worek_lid = np.concatenate((plid[channel], rlid[:,channel]),axis=0)
    worek_r = np.concatenate((rth[:,channel], rlid[:,channel]),axis=0)
    omean = abs(np.mean(pots[channel],axis=0)-np.mean(rth[:,channel],axis=0))
    omean_lid = abs(np.mean(plid[channel],axis=0)-np.mean(rlid[:,channel],axis=0))
    omean_r = abs(np.mean(rth[:,channel],axis=0)-np.mean(rlid[:,channel],axis=0))
    for iteracja in range(n_iter):
        print('\r'+str(iteracja),end='', flush=True)
        all_indexes = np.arange(len(worek),dtype=np.int16)
        r_ind = np.array(random.sample(list(all_indexes),pots.shape[1]))
        zbior1,zbior2 = worek[r_ind],worek[np.delete(all_indexes,r_ind)]
        zbior1l,zbior2l = worek_lid[r_ind],worek_lid[np.delete(all_indexes,r_ind)]
        zbior1r,zbior2r = worek_r[r_ind],worek_r[np.delete(all_indexes,r_ind)]
        pvals_resamp[0,iteracja] = abs(np.mean(zbior1,axis=0)-np.mean(zbior2,axis=0))
        pvals_resamp[1,iteracja] = abs(np.mean(zbior1l,axis=0)-np.mean(zbior2l,axis=0))
        pvals_resamp[2,iteracja] = abs(np.mean(zbior1r,axis=0)-np.mean(zbior2r,axis=0))
        # for ii in range(raw1.shape[0]):
            # stat, pvals_resamp[iteracja,ii] = f_oneway(zbior1[:,ii],zbior2[:,ii])
    for i in range(p_values.shape[1]):
        p_values[0,i] = sum(pvals_resamp[0,:,i]>omean[i])/n_iter
        p_values[1,i] = sum(pvals_resamp[1,:,i]>omean_lid[i])/n_iter
        p_values[2,i] = sum(pvals_resamp[2,:,i]>omean_r[i])/n_iter
        # p_values[0,i] = permutation_test(pots[channel,:,i], rth[:,channel,i], 
                                           # paired=False, method="approximate", seed=0, num_rounds=1000)
        # p_values[1,i] = permutation_test(plid[channel,:,i], rlid[:,channel,i], 
                                           # paired=False, method="approximate", seed=0, num_rounds=1000)
    # stat,clus,p_vals = srn.cluster.permutation_cluster_test_array([pots[channel,:,:], rth[:,channel,:]], adjacency=None,
    #                                                                 n_stat_permutations=1000,progress=False)
    # pvals = np.zeros(len(stat))+1
    # for n_row, pv in enumerate(p_vals):
    #     pvals[clus[n_row]]=pv
    # py.plot(loc_time, np.log10(pvals+1e-4), color= 'k', label='control pval.')
    p_values[:,40:70]=1
    py.plot(loc_time, np.log10(p_values[0]+1e-4), color= 'k', label='control vs recon. control')
    py.plot(loc_time, np.log10(p_values[1]+1e-4), color= 'r', label='ligno. vs recon. control')
    py.plot(loc_time, np.log10(p_values[2]+1e-4), color= 'grey', label='ligno. vs control')
    py.ylabel('log(p-value)'),py.legend(frameon = True, loc=3)
    # py.yticks(fontsize=10)
    py.axhline(np.log10(0.05/100), ls='--',color='grey')
    ax.spines['right'].set_visible(False)
    size=20
    py.xlim(-5,25),py.xlabel('Time (ms)', fontsize=size)
    py.xticks(fontsize = size), py.yticks(fontsize = size)
    
    
def spike_plot(po, pos, name1, name2, path, u1, u2, unit_average=False):
    global units_g,units_g2,wavetrials, loc_times1, loc_times2
    
    loc_times1 = scipy.io.loadmat('../data/sov20.mat')['trial_times'][0]*1000
    loc_times2 = scipy.io.loadmat('../data/sov20lid.mat')['trial_times'][0]*1000
    
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, -.04, po[1], letter= po[0])
    lid=''
    start,stop = -.005*1000, .025*1000
    bins = int((stop - start))
    units_g = scipy.io.loadmat('../data/fig6_spike_file.mat')['units_g'][0]
    # units_w = scipy.io.loadmat('../data/fig6_spike_file.mat')['units_w']
    unit1 = np.asarray(units_g)
    # loc_times=np.asarray(units_g[u1100])
    times = np.linspace(start+1.5,stop+1.5,bins)
    all_units = []
    unit = np.asarray(unit1)
    vec = np.zeros((len(loc_times1), bins))
    for ii,tm in enumerate(loc_times1):
        inds = ((tm+start)<unit)*((tm+stop)>unit)
        ind_list = list(np.where(inds==1)[0])
        hist,biny= np.histogram(unit[ind_list]-tm, range=(start,stop), bins=bins)
        vec[ii] = hist
        all_units.append(unit[ind_list]-tm)
        
    licz = 0    
    for nn, uu in enumerate(all_units):
        # if nn not in trials_list: 
        if nn==30: 
            ax.plot(uu-1,np.zeros(len(uu))+licz,'|',color='grey', ms=12, markeredgewidth=2, label='control')
        else:
            ax.plot(uu-1,np.zeros(len(uu))+licz,'|',color='grey', ms=12, markeredgewidth=2)
        licz+=1
    ax.set_xlim(-5,25)
    # ax.plot(times,vec.mean(axis=0)*mnoz, 'grey')
    # ax.fill_between(times, vec.mean(axis=0)*mnoz-sem(vec,axis=0)*mnoz,
                    # vec.mean(axis=0)*mnoz+sem(vec,axis=0)*mnoz,color='grey',alpha=.1)
    lid='lid'
    # units_g2,units_w2 = load_spikes(name2, path, lid)
    units_g2 = scipy.io.loadmat('../data/fig6_spike_filelid.mat')['units_g2'][0]
    all_units2=[]
    unit = np.asarray(units_g2)
    # loc_times=np.asarray(units_g2[u2100])
    vec2 = np.zeros((len(loc_times2), bins))
    for ii,tm in enumerate(loc_times2):
        inds = ((tm+start)<unit)*((tm+stop)>unit)
        ind_list = list(np.where(inds==1)[0])
        hist,biny= np.histogram(unit[ind_list]-tm, range=(start,stop), bins=bins)
        vec2[ii] = hist
        all_units2.append(unit[ind_list]-tm)
    
    licz2=0
    for nnn, uu in enumerate(all_units2):
        # if nn not in trials_list:  
        if nnn==30:
            ax.plot(uu-1,np.zeros(len(uu))+licz2+licz,'|',color='r', ms=12, markeredgewidth=2, label='ligno.')
        else:
            ax.plot(uu-1,np.zeros(len(uu))+licz2+licz,'|',color='r', ms=12, markeredgewidth=2)
        licz2+=1
        # if nnn==30: 
            # py.axhline(licz2+licz-1, ls='--', color='k')
    ax.set_xlim(-5,25)
    py.legend(loc=1)
    py.axvspan(-1, 2, color='grey')
    size=20
    py.xlim(-5,25),py.xlabel('Time (ms)', fontsize=size)
    py.xticks(fontsize = size), py.yticks(fontsize = size)
    from matplotlib.patches import Rectangle
    # currentAxis = py.gca()
    # currentAxis.add_patch(Rectangle((-1, 0), 3, 200, facecolor="grey"))
    # ax.plot(times,vec2.mean(axis=0)*mnoz, 'r')
    # ax.fill_between(times, vec2.mean(axis=0)*mnoz-sem(vec2,axis=0)*mnoz,
                    # vec2.mean(axis=0)*mnoz+sem(vec2,axis=0)*mnoz,color='r',alpha=.1)
    if unit_average:
        from mpl_toolkits.axes_grid.inset_locator import inset_axes
        inax1 = inset_axes(ax,width="20%", height=.7, bbox_to_anchor=(0,-.5,1,1),loc=2,
                           bbox_transform=ax.transAxes)
        wavetrials = np.asarray(units_w[u1])
        imdata,extent,cmap=spike_heatmap(wavetrials)
        inax1.imshow(imdata,cmap=cmap,
                     interpolation='spline16', aspect='auto', origin='lower',
                     extent=extent)
        inax2 = inset_axes(ax,width="20%", height=.7, bbox_to_anchor=(0,-.1,1,1),loc=2,
                           bbox_transform=ax.transAxes)
        wavetrials = np.asarray(units_w2[u2])
        imdata,extent,cmap=spike_heatmap(wavetrials)
        inax2.imshow(imdata,cmap=cmap,interpolation='spline16', aspect='auto', origin='lower',
                  extent=extent)
        inax2.set_xticks([]),inax1.set_xticks([])
        inax2.set_yticks([]),inax1.set_yticks([])
        inax2.set_ylim(-10,5),inax1.set_ylim(-10,5)
    # ax.set_ylim(0,200)
    ax.set_ylabel('Trials/spikes density'),ax.set_xlabel('Time (ms)')
    
loadir='../data/'
loadir2='../data/'
channel=130
fig = py.figure(figsize=(30,12), dpi=100)
gs = fig.add_gridspec(10, 20)


rat='20'

lfp_profile(('A',1.03), (0,10,0,3), '../data/an_multi_sov20.mat', 'control')
lfp_profile(('B',1.03), (0,10,3,6), '../data/an_multi_sov20lid.mat', 'lignocaine')
exp_design(('C',1.03), (0,10,7,10), '../utils/fig6_hist_sov20.png')

ex_lfp_trials(('D',1.05), (0,3,11,20), '../data/an_multi_sov20.mat',
        '../data/an_multi_sov20lid.mat', channel=channel, trials=list(np.arange(30,31)))
comparison(('E',1.05), (4,6,11,20), '../data/an_multi_sov20.mat',
           '../data/an_multi_sov20lid.mat', channel=channel)
spike_plot(('F',1.05), (7,10,11,20), 'CSC130clean','CSC130clean',
            r'C:\Users\Wladek\Desktop\combinato-main', 1, 1)
py.savefig('fig6_new_spikes_20')
#%%
# mua1 = scipy.io.loadmat('../data/CSC130clean.mat')['data'][0]
# mua2 = scipy.io.loadmat('../data/CSC130cleanlid.mat')['data'][0]
# Fs=scipy.io.loadmat('../data/CSC130.mat')['sr'][0]
# loc_times1 = scipy.io.loadmat('../data/sov20.mat')['trial_times'][0]
# loc_times2 = scipy.io.loadmat('../data/sov20lid.mat')['trial_times'][0]
# py.figure()
# time = np.linspace(0,len(mua2)/Fs, len(mua2))
# b,a = butter(2,500/(Fs/2), btype='high')
# py.plot(time, filtfilt(b,a,mua2))
# for i in range(100):
#     if i in [29,30,31,32]:
#         py.axvline(loc_times2[i], color='red')
# units = np.asarray(units_g2[1])
# py.plot(units/1000, np.zeros(len(units_g2[1])), '|')
#%%
# loadir= r'C:\Users\Wladek\Dysk Google\Autism_model'
# path = r'C:\Users\Wladek\Desktop\combinato-main'

# for i in range(100):
#     tm = int(loc_times1[i]*Fs)
#     mua1[tm:tm+30] = 0
#     tm = int(loc_times2[i]*Fs)
#     mua2[tm:tm+30] = 0
# # lid=''
# name2 = 'CSC130cleanlid'
# scipy.io.savemat(name2+'.mat', dict(data=np.asarray(mua2,dtype=np.float32),sr=Fs))
# subprocess.run(['python', 'css-extract', '--matfile', loadir+'\\'+name2+'.mat'], cwd=path)
# subprocess.run(['python', 'css-mask-artifacts', '--datafile', name2+'\data_'+name2+'.h5'],  cwd=path)
# subprocess.run(['python', 'css-simple-clustering', '--neg','--datafile', name2+'\data_'+name2+'.h5'], cwd=path)
# name2 = 'CSC130clean'
# scipy.io.savemat(name2+'.mat', dict(data=np.asarray(mua1,dtype=np.float32),sr=Fs))
# subprocess.run(['python', 'css-extract', '--matfile', loadir+'\\'+name2+'.mat'], cwd=path)
# subprocess.run(['python', 'css-mask-artifacts', '--datafile', name2+'\data_'+name2+'.h5'],  cwd=path)
# subprocess.run(['python', 'css-simple-clustering', '--neg','--datafile', name2+'\data_'+name2+'.h5'], cwd=path)
# subprocess.run(['python', 'css-gui'], cwd=r'C:\Users\Wladek\Desktop\combinato-main')
# py.close()