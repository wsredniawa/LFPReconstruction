# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:02:31 2020

@author: Wladek
"""

import numpy as np 
import matplotlib.pyplot as py
import matplotlib as mpl
from matplotlib.colors import LogNorm
from figure_properties import *
import matplotlib.patches as mpatches
from scipy.signal import welch, argrelmin, butter, filtfilt, spectrogram, detrend
import scipy.io
import matplotlib.ticker
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

py.close('all')
# mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['xtick.major.size'] = 1
mpl.rcParams['ytick.major.size'] = 1
def exp_design(po, pos, name):
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    img = py.imread(name)
    ax.imshow(img)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    py.xticks([]), py.yticks([])

def cor_map_stat(po, pos, name=''):
    global df, th, cor_score_th
    tps, lp = list(np.arange(28)), 1
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    loadir='../data/'
    cor_score_th = np.array([scipy.io.loadmat(loadir+'an_sov16.mat')['cor'], 
                             scipy.io.loadmat(loadir+'an_sov17.mat')['cor'],
                             scipy.io.loadmat(loadir+'an_sov18.mat')['cor'], 
                             scipy.io.loadmat(loadir+'sov01.mat')['cor'],
                             scipy.io.loadmat(loadir+'sov02.mat')['cor'],
                             scipy.io.loadmat(loadir+'sov11.mat')['cor'],
                             scipy.io.loadmat(loadir+'sov06.mat')['cor'],
                             scipy.io.loadmat(loadir+'sov09.mat')['cor'],
                             scipy.io.loadmat(loadir+'sov19.mat')['cor'],
                             scipy.io.loadmat(loadir+'sov20.mat')['cor'],
                             scipy.io.loadmat(loadir+'sov21.mat')['cor']])
    n_rats = cor_score_th.shape[0]
    th = np.zeros((n_rats,len(tps)+2,lp))
    for n,i in enumerate(tps):
        th[:, n] = (cor_score_th[0][114, i:i+lp], 
                    cor_score_th[1][158, i:i+lp],
                    cor_score_th[2][152, i:i+lp],
                    cor_score_th[3][27, i:i+lp],
                    cor_score_th[4][26, i:i+lp],
                    cor_score_th[5][34, i:i+lp],
                    cor_score_th[6][14, i:i+lp],
                    cor_score_th[7][25, i:i+lp],
                    cor_score_th[8][123, i:i+lp],
                    cor_score_th[9][133, i:i+lp],
                    cor_score_th[10][129, i:i+lp]) 
        
    time_scale=np.linspace(-5,24,30)
    mn_th, std_th = th.mean(axis=0)[:,0], th.std(axis=0)[:,0]/(n_rats**(.5))
    sov_labels = [16,17,18,1,2,11,6,9,19,20,21]
    ex_list = [17,2,3]
    # for i in range(0,11,1):
        # if sov_labels[i] in ex_list: continue
        # py.plot(time_scale, th[i], lw=2, label='sov '+str(sov_labels[i]))
    py.plot(time_scale, mn_th, '-o')
    py.fill_between(time_scale, mn_th-std_th, mn_th+std_th, alpha=.3, color='blue')
    ax.set_ylabel('Correlation'),ax.set_xlabel('Time (ms)')
    py.grid()
    py.title('Correlation Cortical-Thalamic channels')
    py.axvspan(9, 13, alpha=.2, color='grey')
    py.axvspan(-3,-1, alpha=.3, color='indianred')
    # py.text(-3.5,0.8,'Stim Art.')
    # py.arrow(-3,.6,0,1.9,clip_on = False, head_width=.5, length_includes_head=True)
    # py.legend(loc=4,  bbox_to_anchor=(1.3, 0)) 
    py.ylim(-1.1,1.1), py.xlim(-5,22)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    
def ex_lfp_NP(po, pos, name):
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    mpl.rcParams['axes.spines.right'] = True
    # pots = np.load(loadir+name)[::-1]
    pots = scipy.io.loadmat('../data/an_sov19.mat')['pots']
    # pots=pots[:,475+5:625+5]
    py.title('Example evoked potential')
    ch_th,ch_crtx=124,320
    loc_time = np.linspace(-5,25,pots.shape[1])
    py.plot(loc_time, pots[ch_crtx]-pots[ch_crtx,:25].mean(), color='navy', label='Cortical EP', lw=3)
    ax.tick_params(axis='y', labelcolor='navy')
    ax.set_ylim(-2.5,2.5),ax.set_ylabel('Amplitude (mV)')
    ax.legend(loc=1,frameon=False)
    ax2 = ax.twinx()
    ax2.plot(loc_time, pots[ch_th]-pots[ch_th,:25].mean(),'--', color='red', label='Thalamic EP', lw=3)
    ax2.set_ylim(-.25,.25)
    ax2.tick_params(axis='y', labelcolor='red')
    ax.text(3.2,-1.3, 'o',fontsize=25)
    ax.text(8.4,-1.8, 'oo',fontsize=25)
    ax.set_xlabel('Time (ms)')
    # ax.set_xlim(-10,50)
    # py.axvspan(-4, -2, alpha=.3, color='grey')
    py.axvspan(-2, 1, alpha=.3, color='grey')
    ax.arrow(1,-2,-2.5,0, width=.1, head_width=.4)
    # ax2.scatter([-4,-3], [0,-.15],color='black')
    # py.plot([-3,-1],[-.15,-.15], color='black')
    # py.plot([-4,-2],[-.15,-.15], color='black')
    py.axvline(0, ls='--', lw = 2, color='grey')
    ax2.set_xlim(-5,22)
    ax2.legend(loc=1,frameon=False, bbox_to_anchor=(1.01,.9))
    mpl.rcParams['axes.spines.right'] = False
    # est = mpatches.Patch(color='orange', label='Thalamic EP')
    # pot = mpatches.Patch(color='navy', label='Cortical EP')
    # py.legend(handles=[est, pot], ncol=1,loc=1, frameon = False, fontsize = 10) 


def lfp_profile(po, pos, name, title, vmax=200):
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    # ele_pos_pre, wave_pre = np.load('ele_pos_19.npy'), np.load('05pots_19.npy')
    lfp = scipy.io.loadmat(name)['pots']*2
    ele_pos = scipy.io.loadmat('../data/sov19.mat')['ele_pos'].T
    # wave = np.delete(wave_pre, 287, axis=0)
    # ele_pos = np.delete(ele_pos_pre,287, axis=1)
    py.title("Profile of averaged "+title+"EPs (Neuropixel)")
    time= np.linspace(-5,25,lfp.shape[1])
    ch_th,ch_crtx=124,320
    for i in range(lfp.shape[0]-1,0,-1):
        print(i)
        py.plot(time, lfp[i]/2+ele_pos[i,2]-11, color='black', linewidth=.5)
        if i==ch_th:
            py.plot(time, lfp[i]/2+ele_pos[i,2]-11,'--', color='red', linewidth=4)   
        if i==ch_crtx:
            py.plot(time, lfp[i]/2+ele_pos[i,2]-11, color='navy', linewidth=4)
    # py.yticks([0,100,200,300,400,500,600,700],[7,6,5,4,3,2,1,0])
    # cont_lfp1 = abs(lfp[::-1])>20
    py.axvline(0, ls='--', color='grey')
    py.ylabel('Electrode number')
    py.ylim(-7.5,1)
    py.plot([25.8,25.8],[0,-1], 'k',lw=3)
    py.text(26.4,-0.2, '1 mV', size=15)
    # lfp[:,:62]=0
    # py.colorbar(orientation='horizontal', pad=.1, ticks=[-vmax,0,vmax])
    # py.contour(x,y,cont_lfp1*abs(lfp[::-1]), levels=[0,30], cmap="Oranges", linestyles='dashed')
    # 1lfp[150:]=0
    # ont_lfp2 = abs(lfp[::-1])>3
    # py.contour(x,y,cont_lfp2*abs(lfp[::-1]), levels=[0,1], cmap="Oranges", linestyles='dashed')
    py.xlabel('Time (ms)'),py.ylabel('Depth (mm)')
    ax.spines['right'].set_visible(False)
    py.axvline(0, ls='--', lw = 2, color='grey')
    
loadir='../data/'
fig = py.figure(figsize=(20,10))
gs = fig.add_gridspec(20, 23)
exp_design(('A',1.03), (0,20,0,7), 'fig1_rat.png')
# exp_design(('B',1.07), (12,20,0,6), 'hist.png')
# ex_lfp(('D',1.05), (0,9,14,20), 'rat18_EP.npy')

lfp_profile(('B',1.03), (0,20,8,14), '../data/an_sov19.mat', '')
# correlation_map(('D',1.1), (1,20,14,18), 'cor_score_280_19.npy')
# lfp_profile(('E',1.1), (12,20,8,13), 'sov6events_t.npy', 'thalamic', .1)
# ex_lfp_NP(('E',1.05), (0,9,20,25), 'rat13_EP.npy')
ex_lfp_NP(('C',1.05), (0,9,16,23), '05pots_19.npy')
cor_map_stat(('D',1.02), (12,20,16,23), name='')
# exp_design(('F',1), (10,20,20,26), 'pipline.png')


py.savefig('fig1_new')
# py.close('all')
# py.tight_layout() 