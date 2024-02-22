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
    img = py.imread('../data/fig_5_sov'+str(i)+'_histologia.png')
    ele_pos = scipy.io.loadmat('../data/an_sov'+str(i)+'.mat')['ele_pos']
    plane = scipy.io.loadmat('../data/an_sov'+str(i)+'.mat')['est_plane']
    ymin,ymax = plane[1].min(), plane[1].max()
    zmin,zmax = plane[2].min(), plane[2].max()
    ax.imshow(img, aspect='auto', extent=[ymin, ymax, zmax, zmin])
    if po[0]=='A':
        py.scatter(ele_pos[1,:], ele_pos[2,:], s=1, color='k')
        py.plot([ymin,4,4,ymin,ymin],[zmax,zmax,3.5,3.5,zmax], 'grey',lw=3)
    if po[0]=='B':
        py.scatter(ele_pos[1,th_start:], ele_pos[2,th_start:], s=.2, color='k')
        py.plot([ymin,4,4,ymin,ymin],[zmax,zmax,3.5,3.5,zmax], 'grey',lw=3)
    if po[0]=='C':
        py.scatter(ele_pos[1,th_start:], ele_pos[2,th_start:], s=.2, color='k')
        py.plot([ymin,6,6,ymin,ymin],[zmax,zmax,ele_pos[2].min()-.5,ele_pos[2].min()-.5,zmax], 'grey',lw=3)
    if po[0]=='D':
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

def lfp_profile_stat(po, pos, name, title, tp=100, vmax=1e3,rat='16'):
    global pots,ele_pos,inds, cont_lfp1
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, -.24, po[1], letter= po[0])
    csd2 = scipy.io.loadmat('../data/an_sov'+rat+'.mat')[name]/1e3
    csd3 = scipy.io.loadmat('../data/an_sov'+rat+'.mat')['csd_E']/1e3
    ele_pos = scipy.io.loadmat('../data/an_sov'+rat+'.mat')['ele_pos']
    plane = scipy.io.loadmat('../data/an_sov'+rat+'.mat')['est_plane']
    py.title(title, pad=15, fontsize=15)
    # b,a = butter(2, [2/(Fs/2), 300/(Fs/2)], btype='bandpass')
    # pots = filtfilt(b,a,pots)
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
    plot = py.contourf(X,Y,csd2[:,::1,tp].reshape((38,20))-csd3[:,::1,tp].reshape((38,20)),
                       levels=np.linspace(-vmax,vmax,101),
                     cmap=cmap, alpha=.3)
    cbar=py.colorbar(plot, pad=0, ticks=[-vmax, 0, vmax])
    cbar.formatter.set_scientific(False)
    cbar.ax.tick_params(size=0)
    cbar.update_ticks()
    py.scatter(ele_pos[1,th_start:], ele_pos[2,th_start:], s=.2, color='k')
    py.imshow(img[:,::1], extent=[ymin, ymax, zmax, zmin], alpha=1)
    # py.yticks([1,2,3,4,5,6,7],[7,6,5,4,3,2,1])
    if 'pot' in name:
        pots = csd2[:,::1,tp].reshape((38,20))
        cont_lfp1 = pots<-0.1
        py.contour(X,Y,cont_lfp1*pots, levels=np.linspace(-0.5,0.5,31), 
                   cmap="PRGn", linestyles='dashed', linewidth=.1)
    else:
        pots = csd2[:,::1,tp].reshape((38,20))
        cont_lfp1 = abs(pots)>20
        py.contour(X,Y,cont_lfp1*pots, levels=np.linspace(-80,80,51), 
                   cmap="bwr", linestyles='dashed', linewidth=.1)
        # py.imshow(csd_to_plot, extent=[-2.4,-.8,6.9,.1],aspect='auto',
          # origin='lower', vmax=vmax, vmin=-vmax, cmap='bwr', alpha=1)
    # ax.invert_yaxis()
    # cbar.formatter.set_powerlimits((10, 10))    
    if '3' in po[0]:
        py.xlabel('M<->L axis (mm)')
    else:
        py.xticks([])
   
    # if po[0]=='B':py.ylabel('D<->V axis')
    ax.spines['right'].set_visible(False)
    # py.axvline(0, ls='--', lw = 2, color='grey')
    
def compute_score(true_csd_r, est_csd_r):
    epsilon = np.linalg.norm(est_csd_r)/np.max(abs(est_csd_r))
    err_r = abs(est_csd_r/(np.linalg.norm(est_csd_r)) -
                true_csd_r/(np.linalg.norm(true_csd_r)))
    err_r *= epsilon
    point_error = err_r.reshape(true_csd_r.shape)
    return point_error    
    
def compute_stat(csd2,csd3,n_iter=1000,tp=100):
    global worek,r_ind,zbior1
    import random
    pvals_resamp = np.zeros((n_iter,20,38))
    worek = np.concatenate((csd2[..., tp], csd3[..., tp]),axis=0)
    omean = compute_score(np.mean(csd2[..., tp],axis=0),np.mean(csd3[..., tp],axis=0))
    for iteracja in range(n_iter):
        print('\r'+str(iteracja),end='', flush=True)
        all_indexes = np.arange(len(worek),dtype=np.int16)
        r_ind = np.array(random.sample(list(all_indexes), csd2.shape[0]))
        zbior1,zbior2 = worek[r_ind],worek[np.delete(all_indexes,r_ind)]
        pvals_resamp[iteracja] = compute_score(np.mean(zbior1,axis=0),np.mean(zbior2,axis=0))
    return sum(pvals_resamp>omean)/n_iter
    
def lfp_profile2(po, pos, name, title, tp=100, vmax=1e3,rat='16', ifpval=1):
    global pots,ele_pos,inds, cont_lfp1, pval
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, -.24, po[1], letter= po[0])
    csd2 = scipy.io.loadmat('../data/an_sov'+rat+'.mat')[name]/1e3
    csd3 = scipy.io.loadmat('../data/an_sov'+rat+'.mat')['csd_all_multi']/1e3
    ele_pos = scipy.io.loadmat('../data/an_sov'+rat+'.mat')['ele_pos']
    plane = scipy.io.loadmat('../data/an_sov'+rat+'.mat')['est_plane']
    py.title(title, pad=15, fontsize=15)
    img = py.imread('../data/fig_5_sov'+str(i)+'_histologia.png')
    if po[0]=='A1':
        py.ylabel('5 ms after stimulus', fontsize=25, labelpad=25)
    if po[0]=='A2':
        py.ylabel('10 ms after stimulus', fontsize=25, labelpad=25)
    if po[0]=='A3':
        py.ylabel('20 ms after stimulus', fontsize=25, labelpad=25)
    ymin,ymax = plane[1].min(), plane[1].max()
    zmin,zmax = plane[2].min(), plane[2].max()
    
    X,Y = np.meshgrid(np.linspace(ymin, ymax, 20), np.linspace(zmax, zmin, 38))
    if ifpval==0:
        csd3m = np.mean(csd3[...,tp],axis=0)
        csd2m = np.mean(csd2[...,tp],axis=0)
        point_error = compute_score(csd2m,csd3m)
        vmax=.5
        plot = py.contourf(X,Y,point_error[:,::-1].T,
                            levels=np.linspace(-vmax,vmax,101),
                            cmap='bwr', alpha=.3)
        
        cbar=py.colorbar(plot, pad=0, ticks=[-vmax, 0, vmax])
        cbar.formatter.set_scientific(False)
        cbar.ax.tick_params(size=0)
        cbar.update_ticks()
    
    if ifpval:
        pval = compute_stat(csd3,csd2,tp=tp)
    
        plot = py.contourf(X, Y, pval[:,::-1].T, levels=np.linspace(0,0.1,100),
                            cmap='Greys_r', alpha=.5)
        cbar=py.colorbar(plot, pad=0, ticks=[0,0.025,0.05,0.075])
        cbar.formatter.set_scientific(False)
        cbar.ax.tick_params(size=0)
        cbar.update_ticks()
    py.scatter(ele_pos[1,th_start:], ele_pos[2,th_start:], s=.2, color='k')
    py.imshow(img[:,::1], extent=[ymin, ymax, zmax, zmin], alpha=1)
    
    if '3' in po[0]:
        py.xlabel('M<->L axis (mm)')
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
files_list = ['17']
# files_list = ['18']
for i in files_list:
    fig = py.figure(figsize=(24,24), dpi=150)
    gs = fig.add_gridspec(36,40)
    rat,th_start,vmax=i, 88, 3.5
    vmax2=2.5
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
    exp_design(('C',1.07), (0,6,19,23),'th_full_src.png', 
                '', -.1,i)
    # exp_design(('D',1.07), (0,6,27,31),'th_crtx_full_src.png', 
                # '', -.1,i)
    # exp_design(('E',1.07), (0,6,34,42),'th_crtx_full_src.png', 
                # '', par=-.1)
    
    lfp_profile2(('A1',hght), (7,16,0,8), 'csd_est_B', 'Cortical and thalamic',vmax=20, tp=tp,rat=i)
    lfp_profile2(('B1',hght), (7,16,8,16), 'csd_est_C', 'Only thalamic', vmax=vmax, tp=tp, rat=i)
    
    lfp_profile2(('A2',hght), (17,26,0,8), 'csd_est_B', '',vmax=20, tp=tp2,rat=i)
    lfp_profile2(('B2',hght), (17,26,8,16), 'csd_est_C', '', vmax=vmax, tp=tp2, rat=i)
    
    lfp_profile2(('A3',hght), (27,36,0,8), 'csd_est_B', '',vmax=20, tp=tp3, rat=i)
    lfp_profile2(('B3',hght), (27,36,8,16), 'csd_est_C', '', vmax=vmax, tp=tp3, rat=i)
    
    
    # lfp_profile2(('D1',hght), (7,16,25,33), 'csd_th_multi', 'Cortical and thalamic',vmax=vmax2, tp=tp, rat=i)
    # lfp_profile2(('E1',hght), (7,16,33,41), 'csd_th_multi', 'Cortical and thalamic LFP',vmax=1, tp=tp,rat=i)
    lfp_profile2(('C1',hght), (7,16,17,25), 'csd_th_multi', 'Only thalamic',vmax=vmax2, tp=tp, rat=i)
    
    # lfp_profile2(('D2',hght), (17,26,25,33), 'csd_th_multi', '',vmax=vmax2, tp=tp2, rat=i)
    # lfp_profile2(('E2',hght), (17,26,33,41), 'csd_th_multi', '',vmax=.1, tp=tp2, rat=i)
    lfp_profile2(('C2',hght), (17,26,17,25), 'csd_th_multi', '',vmax=vmax2, tp=tp2, rat=i)
    
    # lfp_profile2(('D3',hght), (27,36,25,33), 'csd_th_multi', '',vmax=vmax2, tp=tp3, rat=i)
    # lfp_profile2(('E3',hght), (27,36,33,41), 'csd_th_multi', '',vmax=.3, tp=tp3, rat=i)
    lfp_profile2(('C3',hght), (27,36,17,25), 'csd_th_multi', '',vmax=vmax2, tp=tp3, rat=i)
    
    py.savefig('fig5_new_comp'+rat)
    # py.close('all')
 