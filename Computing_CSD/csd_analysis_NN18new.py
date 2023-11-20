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
import itertools
import scipy.io
from scipy.spatial import distance
import pandas as pd
from scipy.interpolate import griddata
import nibabel as nib

# data = nib.load('CBWJ13_P80_indexed_volume.nii.gz').get_fdata(dtype=np.float16)[::8,::8,::8]
ele_pos_list = []
for c in range(8):
    for d in range(8): ele_pos_list.append([0, c*.2, d*.2])
ele_pos_pre = np.asarray(ele_pos_list) 
ele_pos_list = []
ele_xdist = np.linspace(0,1.3,8,endpoint=1)
for c in range(8):
    for d in range(8): ele_pos_list.append([d*0.0125, c*ele_xdist[1], d*.2])
ele_pos_pre_th = np.asarray(ele_pos_list) 

def rotate(X, theta, axis='x'):
  '''Rotate multidimensional array `X` `theta` degrees around axis `axis`'''
  c, s = np.cos(theta), np.sin(theta)
  if axis == 'x': 
      return np.dot(X,np.array([[1.,  0,  0],[0 ,  c, -s],[0 ,  s,  c] ]))
  elif axis == 'y': 
      return np.dot(X,np.array([[c,  0,  -s],[0,  1,   0],[s,  0,   c]]))
  elif axis == 'z': 
      return np.dot(X,np.array([[c, -s,  0 ],[s,  c,  0 ],[0,  0,  1.]]))

def draw_signal_npy(pots, ele_pos, part,color='black'):
    time = np.linspace(0,0.1,pots.shape[-1])
    for n in range(pots.shape[0]):
        if part=='th':
            py.plot(time+ele_pos[n,1], pots[n]-ele_pos[n,2], color=color)
            py.text(ele_pos[n,1], -ele_pos[n,2], str(n+th_start), fontsize=12)
            # try:
                # py.plot(time+ele_pos[n,1], csd[ele_src_dist[n]]/1e4-ele_pos[n,2], color='blue')
            # except:pass
        else: 
            py.plot(time+ele_pos[n,1], pots[n]/8-ele_pos[n,2], color=color)
            py.text(ele_pos[n,1], -ele_pos[n,2], str(n), fontsize=12)
            # py.xlim(.2,-1.1)
         
def organize_eles(ele_pos, angle):
    ele_pos_crtx, ele_pos_th = np.zeros((2,ele_pos.shape[0],3))
    for n in range(64):
        new_x = 0
        new_y = ele_pos[n,1]
        new_z = -ele_pos[n,2]
        ele_pos_crtx[n] = new_x, new_y, new_z
    return ele_pos_crtx
            
def organize_pos(ele_pos,ele_pos2, plot=False):
    global a0
    a01 = np.array([-0.7, 4.45, 1.1])
    a0 = a01
    ele_pos = organize_eles(ele_pos, 0)
    ele_pos2 = organize_eles(ele_pos2, 0)
    ele_pos_rot1 = rotate(ele_pos, -np.pi/6, 'z')
    ele_pos_rot2 = rotate(ele_pos_rot1, -np.pi/8, 'x')
    a12 = np.array([-0.7, 4.05, 2.01])-a0
    ele_pos_crtx1 = ele_pos_rot2 - np.repeat(ele_pos_rot2[7],64).reshape((3,64)).T
    ele_pos_crtx2 = ele_pos_crtx1 + np.repeat(a12,64).reshape((3,64)).T
    a8t = np.array([-4, 2.3, 7.6]) - a0
    a8t2 = np.array([-4, 2.3, 6.6]) - a0
    a8t3 = np.array([-4, 2.3, 5.6]) - a0
    ele_pos_t1 = ele_pos2 + np.repeat(a8t,64).reshape((3,64)).T
    ele_pos_t2 = ele_pos2 + np.repeat(a8t2,64).reshape((3,64)).T
    ele_pos_t3 = ele_pos2 + np.repeat(a8t3,64).reshape((3,64)).T
    return a0, np.concatenate((ele_pos_crtx1, ele_pos_crtx2, ele_pos_t3, ele_pos_t2, ele_pos_t1), axis=0)
#%%
Fs = 10000
est_env = np.load('est_env.npy')
# files = ['sov17_02wasy_.mat','sov17_03wasy_.mat','sov17_04wasy_.mat','sov17_05wasy_.mat']
files = ['sov18_02wasy_.mat','sov18_03wasy_.mat','sov18_04wasy_.mat']
th_start=96
lista,listam=[],[]
b,a = butter(2., [1/(Fs/2), 2000/(Fs/2)], btype='bandpass')
for i,file in enumerate(files):
    mat_file = scipy.io.loadmat('./mats/'+file)
    crtx, th, ele_pos = filtfilt(b,a,mat_file['crtx']), filtfilt(b,a,mat_file['th']), mat_file['ele_pos']
    crtxm, thm = filtfilt(b,a,mat_file['crtx_multi']), filtfilt(b,a,mat_file['th_multi']),
    lista.append(th)
    listam.append(thm)
    if i>0:
        lista.append(crtx)
        listam.append(crtxm)

a0, ele_pos = organize_pos(ele_pos_pre, ele_pos_pre_th, plot=0)  
pots = np.concatenate((lista[2], lista[4], lista[0], lista[1], lista[3]), axis=0)
potsm = np.concatenate((listam[2][:,:80], listam[4][:,:80], listam[0][:,:80], listam[1][:,:80], listam[3][:,:80]), axis=0)
ind_list=[]
for i in range(40): 
    if i<8: ind_list.append(list(i*8+np.array([0,1,2,7])))
    elif i>=16 and i<24: ind_list.append(list(i*8+np.array([0])))
    elif i>=24 and i<32: ind_list.append(list(i*8+np.array([0,6,7])))
    elif i>=32 and i<40: ind_list.append(list(i*8+np.array([6,7])))
ind_list = list(itertools.chain.from_iterable(ind_list))
pots, ele_pos = np.delete(pots, ind_list,axis=0), np.delete(ele_pos, ind_list , axis=0).T
potsm = np.delete(potsm, ind_list,axis=0)
# ele_pos=ele_pos+np.repeat(a0,240).reshape((3,240))

for i in range(pots.shape[0]): 
    pots[i] = pots[i] - pots[i,:40].mean()

brain_data, skip, div =sf.brain_env(), 1, 1/(0.05*4)
px, pyy, pz = np.where(brain_data[10:-10, 50:70, 100:120]>100)

corx, cory, corz = -3.5, -2.6, 10
est_xyz = np.array([pyy[::skip]/div,pz[::skip]/div,px[::skip]/div])
est_xyz = rotate(est_xyz.T, np.pi, 'x').T
est_xyz = rotate(est_xyz.T, -np.pi/2, 'z').T

est_env = rotate(est_env.T, np.pi, 'x').T
est_env = rotate(est_env.T, -np.pi/2, 'z').T
est_xyz[0]+=corx
est_xyz[1]+=cory
est_xyz[2]+=corz
indx = (est_xyz[2]<ele_pos[2,:th_start].max()+1)*(est_xyz[0]>ele_pos[0,:th_start].min()-1)*(est_xyz[1]>ele_pos[1,:th_start].min()-1)*\
(est_xyz[2]>ele_pos[2,:th_start].min()-1)*(est_xyz[0]<ele_pos[0,:th_start].max()+1)*(est_xyz[1]<ele_pos[1,:th_start].max()+1)
indxt = (est_xyz[2]<ele_pos[2,th_start:].max()+1)*(est_xyz[0]>ele_pos[0,th_start:].min()-1)*(est_xyz[1]>ele_pos[1,th_start:].min()-1)*\
(est_xyz[2]>ele_pos[2,th_start:].min()-1)*(est_xyz[0]<ele_pos[0,th_start:].max()+1)*(est_xyz[1]<ele_pos[1,th_start:].max()+1)
th_cut = (est_xyz[0]>ele_pos[0,-8])*(est_xyz[0]<(ele_pos[0,-8]+0.2))*(est_xyz[2]<7.2)
inds,indst,ind_cut = np.where(indx==1)[0], np.where(indxt==1)[0], np.where(th_cut==1)[0]
eps = list(np.argsort(ele_pos[2]))
ele_src=np.argmin(distance.cdist(ele_pos[:,eps].T,est_xyz.T, 'euclidean'), axis=1)
# check_sources()
# est_env[0]+=corx+5, 
# est_env[1]+=cory, 
# est_env[2]+=1
# X,Y,Z = np.mgrid[est_env[0].min():est_env[0].max():55j, 
#                  est_env[1].min():est_env[1].max():35j, 
#                  est_env[2].min():est_env[2].max():35j]
# R = est_env.T
est_plane = est_xyz[:,ind_cut]
est_plane[2]+=1
# F = griddata(R, np.ones(est_env.shape[1]), (X,Y,Z))
# mlab.figure(bgcolor=(1, 1, 1), size=(1000, 800))
# mlab.points3d(est_xyz[0], est_xyz[1], est_xyz[2], scale_mode='none', scale_factor=0.06, color = (.1, .2, .5))
# mlab.points3d(est_xyz[0,ind_cut], est_xyz[1,ind_cut], est_xyz[2,ind_cut], scale_mode='none', scale_factor=0.1, color = (.1, .9, .1))
# mlab.points3d(est_xyz[0,indst], est_xyz[1,indst], est_xyz[2,indst], scale_mode='none',scale_factor=0.08, color = (.1, .2, .5))
# mlab.points3d(est_xyz[0,inds], est_xyz[1,inds], est_xyz[2,inds], scale_mode='none',scale_factor=0.08, color = (.1, .2, .5))
# mlab.points3d(ele_pos[0], ele_pos[1], ele_pos[2], color=(0, 0, 0), scale_mode='none', scale_factor=.2)
#%%
py.figure()
draw_signal_npy(pots[:th_start]/1e3, ele_pos[:, :th_start].T, 'crtx',color='black')
draw_signal_npy(pots[th_start:]/1e3, ele_pos[:, th_start:].T, 'th',color='black')
#%%
print('computing csd')
k=oKCSD3D(ele_pos.T, pots, own_src=est_xyz,own_est=ele_pos,src_type='gauss', R_init=.4, lambd=1e-5)
k_th=oKCSD3D(ele_pos.T, pots, own_src=est_xyz[:, indst], own_est=ele_pos, src_type='gauss', R_init=.4, lambd=1e-5)
k_crtx=oKCSD3D(ele_pos.T, pots, own_src=est_xyz[:, inds], own_est=ele_pos, src_type='gauss', R_init=.4, lambd=1e-5)
# k.L_curve(lambdas=np.linspace(1e-12,1e-4), Rs = [.1], n_jobs=4)
csd = k.values('CSD')
print('computing forward model')
beta_new = np.dot(np.linalg.inv(k.kernel), k.pots)
k_pot_crtx = np.dot(k.b_interp_pot[:,inds], k.b_pot[inds])/k.n_src
k_pot_th = np.dot(k.b_interp_pot[:,indst], k.b_pot[indst])/k.n_src
k_pot = np.dot(k.b_interp_pot, k.b_pot)/k.n_src
pots_est_th = np.dot(k_pot_th, beta_new)
pots_est_crtx = np.dot(k_pot_crtx, beta_new)
pots_est = np.dot(k_pot, beta_new)
#%%
frags=30
leng=30
leap=10
ch_stay=78
cor_score_crtx, cor_score_th, cor = np.zeros((3,ele_pos.shape[1],frags))
for part in range(frags):
    for ch in range(ele_pos.shape[1]):
        cor_score_crtx[ch,part] = pearsonr(pots[ch,leap*part:leap*part+leng], pots_est_crtx[ch, leap*part:leap*part+leng])[0]
        cor_score_th[ch,part] = pearsonr(pots[ch,leap*part:leap*part+leng], pots_est_th[ch, leap*part:leap*part+leng])[0]
        cor[ch,part] = pearsonr(pots[ch_stay,leap*part:leap*part+leng], pots[ch, leap*part:leap*part+leng])[0]
#%%
py.figure()
py.subplot(131)
py.title('full channels cover')
py.imshow(csd, cmap='bwr', extent=[-5,25, ele_pos.shape[1], 0], vmin=-1e3, vmax=1e3, aspect='auto')
py.axhline(th_start)
py.subplot(132)
py.title('crtx_sources')
py.imshow(k_crtx.values('CSD'), cmap='bwr', extent=[-5,25, ele_pos.shape[1], 0], vmin=-1e3, vmax=1e3, aspect='auto')
# py.imshow(cor_score_crtx[eps], vmax=1, vmin=-1, cmap='PiYG', aspect='auto', extent=[-5,25, 0, ele_pos.shape[1]])
py.subplot(133)
py.title('th sources')
py.imshow(k_th.values('CSD'), cmap='bwr', extent=[-5,25, ele_pos.shape[1], 0], vmin=-1e3, vmax=1e3, aspect='auto')
# py.imshow(cor_score_th[eps], vmax=1, vmin=-1, cmap='PiYG', aspect='auto', extent=[-5,25, 0, ele_pos.shape[1]])
# py.figure()
# py.imshow(cor[eps], vmin=-1, vmax=1, extent=[-5,25,ele_pos.shape[1],1],aspect='auto', cmap='PiYG')
#%%
k_B=oKCSD3D(ele_pos.T, pots, own_src=est_xyz[:,indst], own_est=est_plane, src_type='gauss', R_init=.4, lambd=1e-5)
k_C=oKCSD3D(ele_pos[:,th_start:].T, pots[th_start:], own_src=est_xyz[:,indst], own_est=est_plane, src_type='gauss', R_init=.4, lambd=1e-5)
k_E=oKCSD3D(ele_pos.T, pots, own_src=est_xyz, own_est=est_plane, src_type='gauss', R_init=.4, lambd=1e-5)
k_F=oKCSD3D(ele_pos[:,th_start:].T, pots[th_start:], own_src=est_xyz, own_est=est_plane, src_type='gauss', R_init=.4, lambd=1e-5)
#%%
R=.4
csd_est_all = np.zeros((potsm.shape[1], 38,20,300))
csd_est_th = np.zeros((potsm.shape[1], 38,20,300))
csd_est_B = np.zeros((potsm.shape[1], 38,20,300))
csd_est_C = np.zeros((potsm.shape[1], 38,20,300))
for i in range(potsm.shape[1]):
    k=oKCSD3D(ele_pos.T, potsm[:,i], own_src=est_xyz, own_est=est_plane, src_type='gauss', R_init=R, lambd=1e-5)
    k_th=oKCSD3D(ele_pos[:,th_start:].T, potsm[th_start:,i], own_src=est_xyz, own_est=est_plane, src_type='gauss', R_init=R, lambd=1e-5)
    k_est_B=oKCSD3D(ele_pos.T, potsm[:,i], own_src=est_xyz[:,indst], own_est=est_plane, src_type='gauss', R_init=R, lambd=1e-5)
    k_est_C=oKCSD3D(ele_pos[:,th_start:].T, potsm[th_start:,i], own_src=est_xyz[:,indst], own_est=est_plane, src_type='gauss', R_init=R, lambd=1e-5)
    csd_est_all[i] =  k.values('CSD').reshape((38,20,300))
    csd_est_th[i] = k_th.values('CSD').reshape((38,20,300))
    csd_est_B[i] = k_est_B.values('CSD').reshape((38,20,300))
    csd_est_C[i] = k_est_C.values('CSD').reshape((38,20,300))
#%%
csd2 = k_E.values('CSD')
fig = py.figure()
vmax=1e3
img = py.imread('histologia_17_2.png')
py.imshow(img, extent=[-2.6, 1.2, 7, -0.4])
X,Y = np.meshgrid(np.linspace(-2.6, 1.2, 20), np.linspace(-0.4, 7,38))
cx = py.contourf(X,Y,csd2[:,0].reshape((38,20)),levels=np.linspace(-vmax,vmax,101),
                 cmap='bwr', alpha=.5)
py.scatter(ele_pos[1,th_start:], ele_pos[2,th_start:], color='k')

fps, nSeconds = 5,30
tx = py.text(4,7.5, '0 '+'ms', fontsize=20)
time=np.linspace(-5,25,300)
# anim = animation.FuncAnimation(fig, animate_func,frames = nSeconds*fps,interval=1000/fps)
# anim.save(file[:5]+'.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
scipy.io.savemat('./mats/an_'+file[:5]+'.mat',
                 dict(cs_crtx=cor_score_crtx, cs_th=cor_score_th, pots_th=pots[th_start:], pots_crtx=pots[:th_start],
                      pots=pots,
                      ele_pos=ele_pos+np.repeat(a0,240).reshape((3,240)), 
                      pots_est_crtx=pots_est_crtx, pots_est=pots_est, pots_est_th = pots_est_th,
                      csd=csd, cor=cor, est_plane=est_plane+np.repeat(a0,est_plane.shape[1]).reshape((3,est_plane.shape[1])), 
                      csd_B=k_B.values('CSD').reshape((38,20,300)),
                      csd_C=k_C.values('CSD').reshape((38,20,300)),
                      csd_E=k_E.values('CSD').reshape((38,20,300)),
                      csd_F=k_F.values('CSD').reshape((38,20,300)),
                      csd_Fpot=k_E.values('POT').reshape((38,20,300)),
                      csd_all_multi = csd_est_all,
                      csd_th_multi = csd_est_th,
                      csd_est_B=csd_est_B,
                      csd_est_C=csd_est_C))