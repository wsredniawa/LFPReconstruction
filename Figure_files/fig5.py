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
import matplotlib.ticker as ticker

py.close('all')
# mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['xtick.major.size'] = 1
mpl.rcParams['ytick.major.size'] = 1

def exp_design2(po, pos, name, text,par):
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, -.05, po[1], letter= po[0])
    img = py.imread('./fig5_files/'+name)
    
    py.imshow(img[:,::1], extent=[0,1,0,1], alpha=1)
    
    # if po[0]=='A':
        # py.xlim(0.2,0.7), py.ylim(0.2,.9)
    # else:
        # py.xlim(0.2,0.7), py.ylim(0.1,.9)
    # ax.text(par,-5.5, text, fontsize=30)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    py.xticks([]), py.yticks([])

def exp_design(po, pos, name, text,par,i):
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, -.05, po[1], letter= po[0])
    img = py.imread('fig_5_sov'+str(i)+'_histologia.png')
    ele_pos = scipy.io.loadmat('./mats/an_sov'+str(i)+'.mat')['ele_pos']
    plane = scipy.io.loadmat('./mats/an_sov'+str(i)+'.mat')['est_plane']
    ymin,ymax = plane[1].min(), plane[1].max()
    zmin,zmax = plane[2].min(), plane[2].max()
    ax.imshow(img, aspect='auto', extent=[ymin, ymax, zmax, zmin])
    if po[0]=='A':
        py.scatter(ele_pos[1,:], ele_pos[2,:], s=1, color='k')
        py.plot([ymin,4,4,ymin,ymin],[zmax,zmax,3.5,3.5,zmax], 'grey',lw=3)
    if po[0]=='B':
        py.scatter(ele_pos[1,th_start:], ele_pos[2,th_start:], s=.2, color='k')
        py.plot([ymin,4,4,ymin,ymin],[zmax,zmax,3.5,3.5,zmax], 'grey',lw=3)
    if po[0]=='D':
        py.scatter(ele_pos[1,th_start:], ele_pos[2,th_start:], s=.2, color='k')
        py.plot([ymin,6,6,ymin,ymin],[zmax,zmax,ele_pos[2].min()-.5,ele_pos[2].min()-.5,zmax], 'grey',lw=3)
    if po[0]=='C':
        py.scatter(ele_pos[1,:], ele_pos[2,:], s=.2, color='k')
        py.plot([ymin,6,6,ymin,ymin],[zmax,zmax,ele_pos[2].min()-.5,ele_pos[2].min()-.5,zmax], 'grey',lw=3)
        # py.xlim(0.2,0.7), py.ylim(0.2,.9)
    # else:
    py.xlim(ymin,ymax), py.ylim(zmax,ele_pos[2].min()-.5)
    # ax.text(par,-5.5, text, fontsize=30)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    py.xticks([]), py.yticks([])


def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def lfp_profile2(po, pos, name, title, tp=100, vmax=1e3,rat='16'):
    global pots,ele_pos,inds, cont_lfp1
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, -.24, po[1], letter= po[0])
    csd2 = scipy.io.loadmat('./mats/an_sov'+rat+'.mat')[name]/1e3
    ele_pos = scipy.io.loadmat('./mats/an_sov'+rat+'.mat')['ele_pos']
    plane = scipy.io.loadmat('./mats/an_sov'+rat+'.mat')['est_plane']
    py.title(title, pad=15, fontsize=15)
    extra,res = .5,38
    x_ = np.linspace(ele_pos[0].min()-extra, ele_pos[0].max()+extra,20)
    y_ = np.linspace(ele_pos[1].min()-extra, ele_pos[1].max()+extra,20)
    z_ = np.linspace(ele_pos[2].min()-extra, ele_pos[2].max()+extra,res)
    img = py.imread('fig_5_sov'+str(i)+'_histologia.png')
    if po[0]=='A1':
        py.ylabel('5 ms after stimulus', fontsize=25, labelpad=25)
    if po[0]=='A2':
        py.ylabel('10 ms after stimulus', fontsize=25, labelpad=25)
    if po[0]=='A3':
        py.ylabel('20 ms after stimulus', fontsize=25, labelpad=25)
    ymin,ymax = plane[1].min(), plane[1].max()
    zmin,zmax = plane[2].min(), plane[2].max()
    
    X,Y = np.meshgrid(np.linspace(ymin, ymax, 20), np.linspace(zmax, zmin, 38))
    cmap='bwr'
    if 'pot' in name: cmap = 'PRGn'
    plot = py.contourf(X,Y,csd2[:,::-1,tp].reshape((20,38)).T,levels=np.linspace(-vmax,vmax,101),
                     cmap=cmap, alpha=.6)
    cbar=py.colorbar(plot, pad=0, ticks=[-vmax, 0, vmax])
    cbar.formatter.set_scientific(False)
    cbar.ax.tick_params(size=0)
    cbar.update_ticks()
    py.scatter(ele_pos[1,th_start:], ele_pos[2,th_start:], s=.2, color='k')
    py.imshow(img[:,::1], extent=[ymin, ymax, zmax, zmin], alpha=1)
    # py.yticks([1,2,3,4,5,6,7],[7,6,5,4,3,2,1])
    if 'pot' in name:
        pots = csd2[:,::-1,tp].reshape((20,38)).T
        cont_lfp1 = pots<-0.1
        py.contour(X,Y,cont_lfp1*pots, levels=np.linspace(-0.5,0.5,31), 
                   cmap="PRGn", linestyles='dashed', linewidth=.1)
    else:
        pots = csd2[:,::-1,tp].reshape((20,38)).T
        cont_lfp1 = abs(pots)>20
        py.contour(X,Y,cont_lfp1*pots, levels=np.linspace(-80,80,51), 
                   cmap="bwr", linestyles='dashed', linewidth=.1)
        # py.imshow(csd_to_plot, extent=[-2.4,-.8,6.9,.1],aspect='auto',
          # origin='lower', vmax=vmax, vmin=-vmax, cmap='bwr', alpha=1)
    # ax.invert_yaxis()
    # cbar.formatter.set_powerlimits((10, 10))    
    if '3' in po[0]:
        py.xlabel('M<->L axis (mm)',fontsize=15)
    else:
        py.xticks([])
   
    # if po[0]=='B':py.ylabel('D<->V axis')
    ax.spines['right'].set_visible(False)
    # py.axvline(0, ls='--', lw = 2, color='grey')

loadir='./fig4_files/'
loadir2='./fig6_files/'
tp=100
tp2=150
tp3=250
hght = 1.03
for i in ['17']:
    fig = py.figure(figsize=(24,24), dpi=150)
    gs = fig.add_gridspec(36,40)
    rat,th_start,vmax=i, 88, 3.5
    vmax2=1.5
    if rat=='18': 
        th_start=96
        vmax=2
    if rat=='16':
        th_start=86
        vmax=2
    exp_design(('A',1.07), (0,6,2,6), 'th_crtx_th_src.png', 
               '',-.3,i)
    exp_design(('B',1.07), (0,6,10,14),'th_th_src.png', 
                '', -.1,i)
    exp_design(('C',1.07), (0,6,19,23),'th_crtx_full_src.png', 
                '', -.1,i)
    exp_design(('D',1.07), (0,6,27,31),'th_full_src.png', 
                '', -.1,i)
    # exp_design(('E',1.07), (0,6,34,42),'th_crtx_full_src.png', 
                # '', par=-.1)
    
    
    lfp_profile2(('A1',hght), (7,16,0,8), 'csd_B', 'Cortical and thalamic',vmax=15, tp=tp,rat=i)
    lfp_profile2(('B1',hght), (7,16,8,16), 'csd_C', 'Only thalamic', vmax=vmax, tp=tp, rat=i)
    
    lfp_profile2(('A2',hght), (17,26,0,8), 'csd_B', '',vmax=15, tp=tp2,rat=i)
    lfp_profile2(('B2',hght), (17,26,8,16), 'csd_C', '', vmax=vmax, tp=tp2, rat=i)
    
    lfp_profile2(('A3',hght), (27,36,0,8), 'csd_B', '',vmax=15, tp=tp3, rat=i)
    lfp_profile2(('B3',hght), (27,36,8,16), 'csd_C', '', vmax=vmax, tp=tp3, rat=i)
    
    
    lfp_profile2(('D1',hght), (7,16,25,33), 'csd_F', 'Only thalamic',vmax=vmax2, tp=tp, rat=i)
    # lfp_profile2(('E1',hght), (7,16,33,41), 'csd_Fpot', 'Cortical and thalamic LFP',vmax=1, tp=tp,rat=i)
    lfp_profile2(('C1',hght), (7,16,17,25), 'csd_E', 'Cortical thalamic',vmax=vmax2, tp=tp, rat=i)
    
    lfp_profile2(('D2',hght), (17,26,25,33), 'csd_F', '',vmax=vmax2, tp=tp2, rat=i)
    # lfp_profile2(('E2',hght), (17,26,33,41), 'csd_Fpot', '',vmax=.1, tp=tp2, rat=i)
    lfp_profile2(('C2',hght), (17,26,17,25), 'csd_E', '',vmax=vmax2, tp=tp2, rat=i)
    
    lfp_profile2(('D3',hght), (27,36,25,33), 'csd_F', '',vmax=vmax2, tp=tp3, rat=i)
    # lfp_profile2(('E3',hght), (27,36,33,41), 'csd_Fpot', '',vmax=.3, tp=tp3, rat=i)
    lfp_profile2(('C3',hght), (27,36,17,25), 'csd_E', '',vmax=vmax2, tp=tp3, rat=i)
    
    py.savefig('fig5_new'+rat)
    py.close('all')
 