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

py.close('all')

# mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['xtick.major.size'] = 1
mpl.rcParams['ytick.major.size'] = 1

def exp_design(po, pos, name):
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, -.1, po[1], letter= po[0])
    img = py.imread('../utils/'+name)
    
    ax.imshow(img, aspect='auto',extent=[0,1,0,1])
    py.xlim(.1,.9), py.ylim(0,1)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    py.xticks([]), py.yticks([])

def pots_profile(po, pos, part1, part2, title='', vmax=1, typ='th', rec_color='deepskyblue'):
    global pots1, pots2, ele_pos2
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, -.1, po[1], letter= po[0])
    ele_pos = scipy.io.loadmat('../data/an_sov17.mat')['ele_pos'].T
    pots1 = scipy.io.loadmat('../data/an_sov17.mat')['pots_'+part1]
    pots2 = scipy.io.loadmat('../data/an_sov17.mat')['pots_est'+part2]
    py.title(title)
    ele_pos_list,ele_pos_list2 = [],[]
    for c in range(8):
        for d in range(3): ele_pos_list.append([0, c*.2, d*.2])
        for d in range(8): ele_pos_list.append([0, c*.2, d*.2])
    ele_pos_pre1 = np.asarray(ele_pos_list)
    # ele_pos_pre[:,1]+=3.5
    time = np.linspace(-0.025,0.125,pots1.shape[-1])
    if typ=='th': 
        ch_num, color = pots1.shape[0], 'orange'
    else: 
        ch_num, color = th_start, 'deepskyblue'
        pots1, pots2 = pots1[:th_start], pots2[:th_start]
        pots2[2] = 0 
        # pots1, pots2 =np.delete(pots1, 2, axis=0), np.delete(pots2, 2, axis=0)
        # ele_pos2 = np.delete(ele_pos[:th_start], 2, axis=0)
    for n in range(ch_num):
        if n==0: label1,label2= 'measured', 'reconstructed' 
        else: label1, label2=None, None
        if typ=='th':
            py.plot(time+ele_pos[n+th_start,1], pots1[n]/1e3-ele_pos[n+th_start,2], color='brown', label=label1)
            py.plot(time+ele_pos[n+th_start,1], pots2[n+th_start]/1e3-ele_pos[n+th_start,2], color=rec_color, ls='--', label=label2)
            py.plot([ele_pos[n+th_start,1], ele_pos[n+th_start,1]], [-3,-6.4], ls='--', lw=.1, )
            py.ylim(-8,-4)
            if 'A' in po[0]: py.ylabel('Depth (mm)')
            else: py.yticks([])
            if '3' in po[0]: py.xlabel('Width (mm)')
    if typ=='crtx':
        for n in range(th_start):
            if n<24: cor=ele_pos[21+n%3,2]-ele_pos[n,2]-1
            if n>=24: cor=ele_pos[80+n%8,2]-ele_pos[n,2]-1
            if n==0: label1,label2= 'measured', 'reconstructed' 
            else: label1, label2=None, None
            py.plot(time/2-ele_pos[n,0], pots1[n]/1e4-ele_pos[n,2]-cor, color='navy', label=label1)
            py.plot(time/2-ele_pos[n,0], pots2[n]/1e4-ele_pos[n,2]-cor, color=rec_color, ls='--',label=label2)
            # py.text(-ele_pos[n,0], -ele_pos[n,2], str(n), fontsize=10)
            if 'A' in po[0]: 
                py.ylabel('Distance (mm)')
            else:
                py.yticks([])
        # py.xticks([])
        py.ylim(-2.3,0)
    if po[0]=='C2':
        s=.08
        py.plot([2.3-s,2.325-s], [-2,-2], color='k')
        py.plot([2.3-s,2.3-s], [-2,-1.9], color='k')
        py.text(2.3-s,-2.15, '10 ms',fontsize=10)
        py.text(2.3-s,-1.89, '2 mV',fontsize=10)
    if po[0]=='C3':
        s=-1.45
        py.plot([2.3-s,2.35-s], [-7.8,-7.8], color='k')
        py.plot([2.3-s,2.3-s], [-7.8,-7.7], color='k')
        py.text(2.3-s,-7.95, '10 ms',fontsize=10)
        py.text(2.3-s,-7.69, '0.1 mV',fontsize=10)
    py.legend(ncol=2, frameon = False, fontsize = 10, loc=3)     
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    

th_start=88
loadir='../data/'
fig = py.figure(figsize=(15,14), dpi=400)
gs = fig.add_gridspec(26, 26)
exp_design(('A1',1.07), (0,6,0,7), 'full_space.png')
pots_profile(('A2',1.06), (7,14,0,8),'crtx', '', title='Estimated from all sources to cortex', typ='crtx')
pots_profile(('A3',1.04), (16,26,0,8),'th', '', title='Estimated from all sources to thalamus', rec_color='orange')
exp_design(('B1',1.06), (0,6,9,16), 'crtx_space.png')
pots_profile(('B2',1.06), (7,14,9,17),'crtx', '_crtx', title='Estimated from cortex to cortex', typ='crtx')
pots_profile(('B3',1.04), (16,26,9,17),'th', '_crtx', title='Estimated from cortex to thalamus', rec_color='deepskyblue')
exp_design(('C1',1.06), (0,6,18,26), 'th_space.png')
pots_profile(('C2',1.06), (7,14,18,26),'crtx', '_th', title='Estimated from thalamus to cortex', typ='crtx',rec_color='orange')
pots_profile(('C3',1.04), (16,26,18,26),'th', '_th', title='Estimated from thalamus to thalamus', rec_color='orange')
# exp_design(('B',1.07), (0,10,5,12), 'hist_proj.png')
# exp_design(('',1.07), (0,8,17,25), 'hist_section.png')
# ex_lfp(('C',1.05), (0,9,14,20), 'rat18_EP.npy')
# csd_profile(('C',1.07), (0,11,14,19), 'csd_section18.npz', key='pot', vmax=.5, title='LFP profile')
# csd_profile(('D',1.08), (12,20,0,7), 'csd_section18.npz', title='CSD reconstruction')
# VC_profile(('F',1.08), (12,20,18,25), 'vc_section18.npz', key='pot', vmax=.5, title='Volume conducted LFP from the cortex')
# lfp_profile(('E',1.1), (12,20,8,13), 'sov6events_t.npy', 'thalamic', .1)
# exp_design(('F',1), (12,20,14,20), 'pipline.png')
# py.tight_layout()
py.savefig('fig3_new_test')
# py.close()
 