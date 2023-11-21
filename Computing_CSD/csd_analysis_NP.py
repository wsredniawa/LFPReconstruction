# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 22:22:41 2021

@author: Wladek
"""

import numpy as np
import pylab as py 
from scipy.signal import filtfilt, butter, detrend, argrelmax, argrelmin
from scipy.stats import pearsonr
from kcsd import oKCSD3D
from mayavi import mlab
from basicfunc import model_data, kaji_score
import sov_funcs as sf
import scipy.io
from scipy.spatial import distance

def rotate(X, theta, axis='x'):
  '''Rotate multidimensional array `X` `theta` degrees around axis `axis`'''
  c, s = np.cos(theta), np.sin(theta)
  if axis == 'x': 
      return np.dot(X,np.array([[1.,  0,  0],[0 ,  c, -s],[0 ,  s,  c] ]))
  elif axis == 'y': 
      return np.dot(X,np.array([[c,  0,  -s],[0,  1,   0],[s,  0,   c]]))
  elif axis == 'z': 
      return np.dot(X,np.array([[c, -s,  0 ],[s,  c,  0 ],[0,  0,  1.]]))
 
def check_sources():
    mlab.figure(bgcolor=(0.2, 0.2, 0.2), size=(1000, 800))
    mlab.points3d(est_xyz[0], est_xyz[1], est_xyz[2], 
                    scale_mode='none', scale_factor=0.01, color = (.5, .5, .5))

    mlab.points3d(est_xyz[0,inds], est_xyz[1,inds], est_xyz[2,inds], 
                   scale_mode='none', scale_factor=0.01, color = (.1, .5, .5))
    mlab.points3d(est_xyz[0, inds_cent], est_xyz[1, inds_cent], est_xyz[2, inds_cent], 
                  scale_mode='none', scale_factor=0.01, color = (.5, .1, .5))
    # mlab.points3d(est_xyz[0, ele_src_dist], est_xyz[1, ele_src_dist], est_xyz[2,ele_src_dist], color=(0, 0, 1), scale_mode='none', scale_factor=.2)
    mlab.points3d(ele_pos[0], ele_pos[1], ele_pos[2], color=(1, 0, 0), scale_mode='none', scale_factor=.01)

slownik = {'sov19.mat':(260, 123),
           'sov20.mat':(267, 133),
           'sov20lid.mat':(267, 133),
           'sov21.mat':(264, 129),
           'sov21lid.mat':(264, 129)}
#%%
Fs = 10000
file = 'sov19.mat'
above_cortex=slownik[file][0]
th_channel=slownik[file][1]
mat_file = scipy.io.loadmat('../data/'+file)
# est_env = np.load('est_env.npy')

crtx, th, ele_pos = mat_file['crtx'], mat_file['th'], mat_file['ele_pos'][:,::-1]
pots=np.concatenate((th,crtx))
for i in range(pots.shape[0]): pots[i] = (pots[i] - pots[i,:40].mean())#*1e3/512

num_points = 200
lines = 25
est_xyz = np.zeros((3,num_points,lines))
cent_array =  np.array([np.linspace(ele_pos[0,0], ele_pos[0,-1], num_points),
                        np.linspace(ele_pos[1,0], ele_pos[1,-1], num_points),
                        np.linspace(ele_pos[2,0]+1, ele_pos[2,-1]+.1, num_points)])
n=0
for row in np.arange(-2,3,1):
    for col in np.arange(-2,3,1):
        side_ar = cent_array.copy()
        side_ar[0]+=row/20
        side_ar[1]+=col/20
        est_xyz[:,:,n] = side_ar
        n+=1
        
# extra,res = 1,24
extra,res = .1,200
x_ = np.linspace(ele_pos[0].min()-extra, ele_pos[0].max()+extra,res//20)
y_ = np.linspace(ele_pos[1].min()-extra, ele_pos[1].max()+extra,res//20)
z_ = np.linspace(ele_pos[2].min()-extra, ele_pos[2].max()+extra,res)
csd_space = np.meshgrid(x_, y_, z_)
est_xyz = np.vstack(map(np.ravel, csd_space))
#%%
k=oKCSD3D(ele_pos.T, pots, own_src=est_xyz,own_est=ele_pos,src_type='gauss', 
          R_init=.4, lambd=1e-5)
# k.L_curve()
csd = k.values('CSD')
#%%
indx = est_xyz[2]<6.5
indxt = est_xyz[2]>6.5
inds, indst,inds_cent = np.where(indx==1)[0], np.where(indxt==1)[0], list(np.arange(12,est_xyz.shape[1],lines))
# pots_est_crtx = model_data(ele_pos, csd[inds], est_xyz[:,inds], sigma=1, r0=.4,  typ='gauss')/len(inds)
# pots_est_th = model_data(ele_pos, csd[indst], est_xyz[:,indst], sigma=1, r0=.4,  typ='gauss')/len(indst)
# pots_est = model_data(ele_pos, csd, est_xyz[:,:], sigma=1, r0=.4, typ='gauss')/est_xyz.shape[1]
beta_new = np.dot(np.linalg.inv(k.kernel), k.pots)
k_pot_crtx = np.dot(k.b_interp_pot[:,inds], k.b_pot[inds])/k.n_src
k_pot_th = np.dot(k.b_interp_pot[:,indst], k.b_pot[indst])/k.n_src
pots_est_th = np.dot(k_pot_th, beta_new)
pots_est_crtx = np.dot(k_pot_crtx, beta_new)
#%%
# check_sources()
frags=30
leng=30
leap=10
ch_stay=320
cor_score_crtx, cor_score_th, cor_score = np.zeros((3,ele_pos.shape[1],frags))
for part in range(frags-3):
    for ch in range(ele_pos.shape[1]):
        cor_score_crtx[ch,part] = pearsonr(pots[ch,leap*part:leap*part+leng], pots_est_crtx[ch, leap*part:leap*part+leng])[0]
        cor_score_th[ch,part] = pearsonr(pots[ch,leap*part:leap*part+leng], pots_est_th[ch, leap*part:leap*part+leng])[0]
    # cor_score_crtx[:,part] = kaji_score(pots[:,leap*part:leap*part+leng], pots_est_crtx[:, leap*part:leap*part+leng])
    # cor_score_th[:,part] = kaji_score(pots[:,leap*part:leap*part+leng], pots_est_th[:, leap*part:leap*part+leng])
#%%
py.figure()
py.subplot(131)
py.title('CSD along electrodes')
py.imshow(csd, cmap='bwr', origin='lower',
          extent=[-5,25, 200, 0], vmin=-1e2, vmax=1e2, aspect='auto')
py.axhline(1, color='black', ls='--')
py.axhline(177, color='black', ls='--')

py.subplot(132)
py.title('LFP measured')
py.imshow(pots, cmap='PRGn', origin='lower', vmin=-1e1, vmax=1e1, aspect='auto')
py.subplot(133)
# beta_new = np.dot(np.linalg.inv(k.kernel), k.pots)
# k_pot = np.dot(k.b_interp_pot[:,inds], k.b_pot[inds])/5000
# pots_est_csd = np.dot(k_pot, beta_new)
# py.title('LFP reconstructed  from all sources')
# py.imshow(pots-pots_est_csd, cmap='PRGn', origin='lower', vmin=-1e1, vmax=1e1, aspect='auto')
# py.plot(pots_est_csd[343])
# py.plot(pots_est[343])
# py.subplot(222)
py.imshow(cor_score_th, vmax=1, vmin=-1, cmap='PiYG', origin='lower', 
           aspect='auto', extent=[-5,25, 0, ele_pos.shape[1]])
py.colorbar()
# py.subplot(224)
# py.imshow(cor_score_th, vmax=1, vmin=-1, cmap='PiYG', origin='lower',
          # aspect='auto', extent=[-5,25, 0, ele_pos.shape[1]])
#%%
if '19' in file:
    k2=oKCSD3D(ele_pos[:,above_cortex:].T, pots[above_cortex:],own_src=est_xyz[:,inds],own_est=ele_pos,
              src_type='gauss', R_init=.4, lambd=1e-5)
    # k.L_curve()
    csd_VC = k2.values('CSD')
    pot_VC = k2.values('POT')
    py.figure()
    py.subplot(131)
    py.imshow(csd_VC, cmap='bwr', extent=[-5,25, ele_pos.shape[1], 0], vmin=-1e3, vmax=1e3, aspect='auto', origin='lower')
    # py.axhline(135)
    py.subplot(132)
    ele_dist=distance.cdist(ele_pos.T,ele_pos.T, 'euclidean')[-th_channel]
    for i in range(ele_pos.shape[1]): csd_VC[i]=(csd[i]/ele_dist[130]+1e-10)
    py.imshow(csd_VC, cmap='PRGn', extent=[-5,25, ele_pos.shape[1], 0], vmin=-1e3, vmax=1e3, aspect='auto',origin='lower')
    # py.axhline(135)
    py.subplot(133)
    py.imshow(pot_VC, cmap='PRGn', extent=[-5,25, ele_pos.shape[1], 0], vmin=-1e1, vmax=1e1, aspect='auto', origin='lower')
    scipy.io.savemat('../data/an_'+file,
                     dict(cs_crtx=cor_score_crtx, cs_th=cor_score_th,
                     pots_est_crtx=pots_est_crtx, pots_est_th= pots_est_th, pots=pots,
                     csd=csd, pot_VC=pot_VC, csd2 = k2.values('CSD')))
else:
    scipy.io.savemat('../data/an_'+file,
                     dict(cs_crtx=cor_score_crtx, cs_th=cor_score_th,
                          pots_est_crtx=pots_est_crtx, pots_est_th= pots_est_th, pots=pots,
                          csd=csd))