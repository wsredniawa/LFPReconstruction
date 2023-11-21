# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 11:52:36 2020

@author: Wladek
"""
import numpy as np
import pylab as py
from mayavi import mlab
from scipy.integrate import simps
from joblib.parallel import Parallel, delayed

def remove_points(mtrx, xpos, ypos, zpos, tp, thr):
    idx_to_remove = np.where(abs(mtrx[:,tp]) < thr)
    xpos = np.delete(xpos, idx_to_remove)
    ypos = np.delete(ypos, idx_to_remove)
    zpos = np.delete(zpos, idx_to_remove)
    mtrx = np.delete(mtrx[:,tp], idx_to_remove)
    return mtrx, xpos, ypos, zpos

def plot_1d_proj(est_csd,est_pot,roz2,loadir,k, st=0,sts=200):
    py.figure()
    csd_1d_projection = []
    pot_1d_projection = []
    for i in np.arange(64): 
        csd_1d_projection.append(np.mean(est_csd[i*145:(i+1)*145], axis=0))
        pot_1d_projection.append(np.mean(est_pot[i*145:(i+1)*145], axis=0))
    csd_1d_projection = np.asarray(csd_1d_projection)
    pot_1d_projection = np.asarray(pot_1d_projection)
    py.suptitle(loadir[-5:-1]+', Lambda: '+ str(k.lambd)+' R: '+str(k.R))
    py.subplot(121)
    roz = csd_1d_projection[:,st:sts].max()/roz2
    py.imshow(csd_1d_projection[::-1,st:sts], aspect='auto', vmax=roz/4, vmin=-roz/4, 
              cmap='bwr', origin='bottom', extent=[st/10,sts/10,0,8])
    py.colorbar(orientation='horizontal')
    py.contour(csd_1d_projection[::-1,st:sts],[-roz/2],colors=['black'], extent=[st/10,sts/10,0,8])
    py.subplot(122)
    py.title('LFP model')
    py.imshow(pot_1d_projection[::-1,st:sts], aspect='auto', vmax=roz/10, vmin=-roz/10, 
              cmap='PRGn', origin='bottom', extent=[st/10,sts/10,0,8])
    py.colorbar(orientation='horizontal')
    
def plot_pots(b1_name, trials_b, trials_t, Fs):
    py.figure()
    py.subplot(121)
    py.suptitle(b1_name)
    py.title('BCX')
    time = np.linspace(-250,750,Fs)
    for i in range(trials_b.shape[0]):
        py.plot(time, trials_b[i].mean(axis=0)-i*1, color='blue')
    py.xlim(-10, 40)
    py.grid()
    py.subplot(122)
    py.title('Thalamus')
    for i in range(trials_t.shape[0]):
        py.plot(time, trials_t[i].mean(axis=0)-i*.2, color='blue')
    py.xlim(-10, 40)
    py.grid()
    
def check_3d(ele_pos, est_xyz, env_est, save=True):
    mlab.figure(bgcolor=(0.2, 0.2, 0.2), size=(1000, 800))
    mlab.points3d(ele_pos[0], ele_pos[1], ele_pos[2], color = (0, 1, 0), 
                  scale_mode='none', scale_factor=.2)
    mlab.points3d(env_est[0], env_est[1], env_est[2], scale_mode='none', scale_factor=0.03, color = (1, 1, 0.5))
    mlab.points3d(est_xyz[0],  est_xyz[1],  est_xyz[2], scale_mode='none', scale_factor=.05)
    mlab.view(elevation = 70, azimuth=50, distance = 80)
    
def animate_data(wave, est_csd, est_xyz, est_env, ele_pos, stds, chs=[10,30]):
    roz = np.max(abs(est_csd))/10
    # colors = ['blue', 'navy', 'indianred', 'maroon', 'green', 'darkgreen']
    @mlab.animate(delay=100)
    def anim():
        tp = 0
        while True:
            est_csdi, xposii, yposii, zposii = remove_points(est_csd, est_xyz[0], est_xyz[1], est_xyz[2], tp, stds*np.std(est_csd[:,tp]))
            pt.mlab_source.reset(x=xposii, y=yposii, z=zposii, scalars=est_csdi)
            pt.module_manager.scalar_lut_manager.reverse_lut = True
            lines.mlab_source.reset(y=[tp/50], z=[wave_20[tp]-7])
            tp+=1
            if tp==200: tp=0
            yield
            
    dur = 200
    wave_20 = -wave[chs[1]]/wave[chs[1]].min()
    wave_10 = -wave[chs[0]]/wave[chs[0]].min()
    x_line, y_line = np.linspace(0,4,dur), np.zeros(dur)+10
    mlab.figure(bgcolor=(0.2, 0.2, 0.2), fgcolor=(0.,0.,0.),size=(1000, 800))
    mlab.points3d(est_env[0,::2], est_env[1,::2], est_env[2,::2], scale_mode='none', scale_factor=0.03, color = (1, 1, 0.5))
    mlab.points3d(ele_pos[0], ele_pos[1], ele_pos[2], color = (0, 1, 0), scale_mode='none', scale_factor=.2) 
    mlab.plot3d(y_line, x_line, wave_20-7, color=(1,1,1),tube_radius=.1)
    mlab.plot3d(y_line, x_line, wave_10-5, color=(1,1,1),tube_radius=.1)
    lines = mlab.points3d([10], [0], [wave_20[0]], color=(.9,0,.16), scale_factor=.4)
    est_csdi, xposii, yposii, zposii = remove_points(est_csd, est_xyz[0], est_xyz[1], est_xyz[2], 0, stds*np.std(est_csd[:,0]))
    pt = mlab.points3d(xposii, yposii, zposii, est_csdi, colormap = 'RdBu', 
                       vmin = -roz, vmax = roz, scale_mode='none', scale_factor=.3)
    anim()
    
def draw_cylinder(mv_x, mv_y, mv_z,angle, height=3.2):
    est_xyz = []
    point_list = []
    i=0
    for z in np.arange(height,0,-.1):
        for r in np.arange(.1,1,.1):
            for x in np.arange(-2, 2.1,.1):
                if abs(x)>r: continue
                y = r**2-x**2
                new_x = x
                new_y = y*np.cos(angle) - z*np.sin(angle)
                new_z = y*np.sin(angle) + z*np.cos(angle)
                est_xyz.append((new_x+mv_y,new_y+mv_x,new_z+mv_z))
                point_list.append((new_x,new_y,new_z))
                new_x = x
                new_y = -y*np.cos(angle) - z*np.sin(angle)
                new_z = -y*np.sin(angle) + z*np.cos(angle)
                if (new_x,new_y,new_z) in point_list: 
                    # print('Repeat2', new_y,new_x,new_z)
                    continue
                point_list.append((new_x,new_y,new_z))
                est_xyz.append((new_x+mv_y,new_y+mv_x,new_z+mv_z))
                i+=2
    return np.asarray(est_xyz).T, np.asarray(point_list).T

def draw_sphere(mv_x, mv_y, mv_z,angle):
    est_xyz = []
    point_list = []
    i=0
    for z in np.arange(1.1,-1.2,-.1):
        for r in np.arange(.1,np.sqrt(1.2**2-z**2),.1):
            for x in np.arange(-1.6,1.6,.1):
                if abs(x)>r: continue
                y = r**2-x**2
                new_x = x
                new_y = y*np.cos(angle) - z*np.sin(angle)
                new_z = y*np.sin(angle) + z*np.cos(angle)
                est_xyz.append((new_y+mv_x,new_x+mv_y,new_z+mv_z))
                point_list.append((new_y,new_x,new_z))
                new_x = x
                new_y = -y*np.cos(angle) - z*np.sin(angle)
                new_z = -y*np.sin(angle) + z*np.cos(angle)
                if (new_y,new_x,new_z) in point_list: 
                    # print('Repeat2', new_y,new_x,new_z)
                    continue
                point_list.append((new_y,new_x,new_z))
                est_xyz.append((new_y+mv_x,new_x+mv_y,new_z+mv_z))
                i+=2
    print(i)
    return np.asarray(est_xyz).T

def brain_env():
    # from skimage import filters
    modeldir = '/Users/Wladek/Dysk Google/Tom_data/'
    skacz = 4
    brain_data = np.load(modeldir+'volume_Brain.npy')[::skacz,::skacz, ::skacz]
    brain_data = np.rot90(brain_data, 1, axes=(1,2))
     #35
    return brain_data

def integrate_2D(x, y, xlim, ylim, csd, h, xlin, ylin, X, Y):
    Ny = xlin.shape[0]
    m = np.sqrt((x - X)**2 + (y - Y)**2)     # construct 2-D integrand
    m[m < 0.0000001] = 0.0000001             # I increased acuracy
    y = np.arcsinh(2*h / m) * csd            # corrected
    I = np.zeros(Ny)                         # do a 1-D integral over every row
    for i in range(Ny):
        I[i] = simps(y[i, :], ylin)          # I changed the integral
    F = simps(I, xlin)                       # then an integral over the result 
    return F 

def integrate_3D(x, y, z, xlim, ylim, zlim, csd, xlin, ylin, zlin, X, Y, Z):
    Nz = zlin.shape[0]
    Ny = ylin.shape[0]
    m = np.sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2)
    m[m < 0.0000001] = 0.0000001
    z = csd / m
    Iy = np.zeros(Ny)
    # print(z.shape, Ny, Nz, zlin.shape)
    for j in range(Ny):
        Iz = np.zeros(Nz)                        
        for i in range(Nz):
            Iz[i] = simps(z[:,j,i], xlin)
        Iy[j] = simps(Iz, ylin)
    F = simps(Iy, zlin)
    return F 

def calculate_potential_3D(true_csd, ele_xx, ele_yy, ele_zz, csd_x, csd_y, csd_z):
    xlin = csd_x[:,0,0]
    ylin = csd_y[0,:,0]
    zlin = csd_z[0,0,:]
    xlims = [xlin[0], xlin[-1]]
    ylims = [ylin[0], ylin[-1]]
    zlims = [zlin[0], zlin[-1]]
    sigma = 1.0
    # pots = np.zeros(len(ele_xx))
    # for ii in range(len(ele_xx)):
    #     pots[ii] = integrate_3D(ele_xx[ii], ele_yy[ii], ele_zz[ii],
    #                             xlims, ylims, zlims, true_csd, 
    #                             xlin, ylin, zlin, 
    #                             csd_x, csd_y, csd_z)
    #     print('Electrode:', ii)
    pots = Parallel(n_jobs=4)(delayed(integrate_3D)
                              (ele_xx[ii], ele_yy[ii], ele_zz[ii],
                                xlims, ylims,zlims, true_csd, 
                                xlin, ylin, zlin, csd_x, csd_y, csd_z)
                              for ii in range(len(ele_xx)))
    pots=np.asarray(pots)
    pots /= 4*np.pi*sigma
    return pots




def calculate_potential_2D(true_csd, ele_xx, ele_yy, csd_x, csd_y):
    xlin = csd_x[:,0]
    ylin = csd_y[0,:]
    xlims = [xlin[0], xlin[-1]]
    ylims = [ylin[0], ylin[-1]]
    sigma = 1
    h = 50.
    pots = np.zeros(len(ele_xx))
    # for ii in range(len(ele_xx)):
    #     pots[ii] = integrate_2D(ele_xx[ii], ele_yy[ii], 
    #                             xlims, ylims, true_csd, h, 
    #                             xlin, ylin, csd_x, csd_y)
    pots = Parallel(n_jobs=4)(delayed(integrate_2D)
                              (ele_xx[ii], ele_yy[ii], 
                               xlims, ylims, true_csd, h, 
                               xlin, ylin, csd_x, csd_y)
                              for ii in range(len(ele_xx)))
    pots=np.asarray(pots)
    pots /= 2*np.pi*sigma
    return pots