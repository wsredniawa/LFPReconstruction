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
from basicfunc import model_data
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
def check_sources(est_xyz, ele_pos, inds, indst):
    mlab.figure(bgcolor=(0.2, 0.2, 0.2), size=(1000, 800))
    # mlab.points3d(est_xyz[0], est_xyz[1], est_xyz[2], scale_mode='none', scale_factor=0.03, color = (1, 1, 0))
    mlab.points3d(est_xyz[0,indx], est_xyz[1,indx], est_xyz[2,indx], scale_mode='none', scale_factor=0.05, color = (.1, .5, .9))
    mlab.points3d(est_xyz[0,indst], est_xyz[1,indst], est_xyz[2,indst], scale_mode='none', scale_factor=0.05, color = (.5, .1, .5))
    # mlab.points3d(est_xyz[0, ele_src_dist], est_xyz[1, ele_src_dist], est_xyz[2,ele_src_dist], color=(0, 0, 1), scale_mode='none', scale_factor=.2)
    mlab.points3d(ele_pos[0], ele_pos[1], ele_pos[2], color=(1, 0, 0), scale_mode='none', scale_factor=.2)

#%%
Fs = 10000
b,a = butter(2,[1/(Fs/2), 1000/(Fs/2)], btype='bandpass')
file = 'sov11.mat'
mat_file = scipy.io.loadmat('../Data/'+file)
# est_env = np.load('est_env.npy')

crtx, th, ele_pos = mat_file['crtx'], mat_file['th'], mat_file['ele_pos']
pots=np.concatenate((crtx,th))

for i in range(pots.shape[0]): pots[i] = pots[i] - pots[i,:40].mean()

# brain_data, skip, div =sf.brain_env(), 1, 1/(0.05*4)
# px, pyy, pz = np.where(brain_data[10:-10, 49:61, 100:125]>100)
if '01' in file:
    corx, cory, corz = -3.5, 2, -2.2
if '02' in file:
    corx, cory, corz = -3.5, 2, -2.2
if '03' in file:
    corx, cory, corz = -4.6, 1.9, -2.2
if '06' in file:
    corx, cory, corz = -3.5, 1.9, -20
if '09' in file:
    corx, cory, corz = -3.5, 1, -3.4
if '11' in file:
    corx, cory, corz = -3.5, 1, -3.4
# else:
    # corx, cory, corz = -2, 4, -2
extra,res = 1,24
x_ = np.linspace(ele_pos[0].min()-extra, ele_pos[0].max()+extra,res)
y_ = np.linspace(ele_pos[1].min()-extra, ele_pos[1].max()+extra,res)
z_ = np.linspace(ele_pos[2].min()-extra, ele_pos[2].max()+extra,res)
csd_space = np.meshgrid(x_, y_, z_)
est_xyz = np.vstack(map(np.ravel, csd_space))
# est_xyz = np.array([xpos,ypos,zpos])
# ele_pos = rotate(ele_pos.T, np.pi/8, 'x').T
# ele_pos = ele_pos[:,:]
indx = est_xyz[2]<4.5
indxt = est_xyz[2]>4.5
if '06' in file:
    indx = est_xyz[2]<4
    indxt = est_xyz[2]>4

inds, indst = np.where(indx==1)[0], np.where(indxt==1)[0]
ele_line = np.array([np.linspace(ele_pos[0,0], ele_pos[0,-1], 50),
                     np.linspace(ele_pos[1,0], ele_pos[1,-1], 50),
                     np.linspace(ele_pos[2,0], ele_pos[2,-1], 50)])
# ele_src=np.argmin(distance.cdist(ele_line.T,est_xyz.T, 'euclidean'), axis=1)
# if '03' in file or '06' in file:
#     ele_src=np.argmin(distance.cdist(ele_pos.T,est_xyz.T, 'euclidean'), axis=1)
# ele_src_dist=[]
# [ele_src_dist.append(x) for x in list(ele_src) if x not in ele_src_dist]

check_sources(est_xyz, ele_pos, inds, indst)
#%%
k=oKCSD3D(ele_pos.T, pots, own_src=est_xyz,own_est=ele_pos, src_type='gauss', R_init=.4, lambd=1e-5)
# k.L_curve()
csd = k.values('CSD')
#%%
beta_new = np.dot(np.linalg.inv(k.kernel), k.pots)
k_pot_crtx = np.dot(k.b_interp_pot[:,inds], k.b_pot[inds])/k.n_src
k_pot_th = np.dot(k.b_interp_pot[:,indst], k.b_pot[indst])/k.n_src
pots_est_th = np.dot(k_pot_th, beta_new)
pots_est_crtx = np.dot(k_pot_crtx, beta_new)

frags=30
leng=30
leap=10
cor_score_crtx, cor_score_th = np.zeros((2,ele_pos.shape[1],frags))
for part in range(frags):
    for ch in range(ele_pos.shape[1]):
        cor_score_crtx[ch,part] = pearsonr(pots[ch,leap*part:leap*part+leng], pots_est_crtx[ch, leap*part:leap*part+leng])[0]
        cor_score_th[ch,part] = pearsonr(pots[ch,leap*part:leap*part+leng], pots_est_th[ch, leap*part:leap*part+leng])[0]
#%%
py.figure()
vmax=1
py.subplot(121)
py.imshow(csd, cmap='bwr', extent=[-5,25, ele_pos.shape[1], 0], vmin=-vmax, vmax=vmax, aspect='auto')
py.subplot(222)
py.imshow(cor_score_crtx, vmax=1, vmin=-1, cmap='PiYG', aspect='auto', extent=[-5,25, ele_pos.shape[1],0])
py.subplot(224)
py.imshow(cor_score_th, vmax=1, vmin=-1, cmap='PiYG', aspect='auto', extent=[-5,25, ele_pos.shape[1],0])
scipy.io.savemat('./mats/an_'+file,
                 dict(cs_crtx=cor_score_crtx, cs_th=cor_score_th,
                      csd=csd, pots=pots, pots_est_th = pots_est_th,pots_est_crtx=pots_est_crtx))
#%%
# k=oKCSD3D(ele_pos[:,:-above_cortex].T, pots[:-above_cortex],own_src=est_xyz[:,inds],own_est=est_xyz,
#           src_type='step', R_init=.1, lambd=1e-5)
# csd_VC = k.values('CSD')
# pot_VC = k.values('POT')
# py.figure()
# py.subplot(131)
# py.imshow(csd_VC[ele_src_dist], cmap='bwr', extent=[-5,25, ele_pos.shape[1], 0], vmin=-1e4, vmax=1e4, aspect='auto')
# # py.axhline(135)
# py.subplot(132)
# ele_dist=distance.cdist(ele_pos.T,ele_pos.T, 'euclidean')[-th_channel]
# for i in range(ele_pos.shape[1]): csd_VC[ele_src_dist][i]/=(ele_dist[i]+1e-5)
# py.imshow(csd_VC[ele_src_dist], cmap='PRGn', extent=[-5,25, ele_pos.shape[1], 0], vmin=-vmax, vmax=1e4, aspect='auto')
# # py.axhline(135)
# py.subplot(133)
# py.imshow(pot_VC[ele_src_dist], cmap='PRGn', extent=[-5,25, ele_pos.shape[1], 0], vmin=-1e1, vmax=1e1, aspect='auto')

