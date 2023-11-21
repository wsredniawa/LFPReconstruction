# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 17:26:00 2017

@author: Wladek
"""
import matplotlib.cm as cm
import numpy as np
import pylab as py
import os
from sklearn.decomposition import FastICA, PCA, NMF
from scipy.signal import correlate, welch, spectrogram, argrelextrema, butter,filtfilt, hilbert
from scipy import special
import pandas as pd


def kaji_score(pot1, pot2):
    norm_pot1, norm_pot2 = pot1/np.mean(pot1**2)**(1/2), pot2/np.mean(pot2**2)**(1/2)
    return np.diagonal(np.inner(norm_pot1, norm_pot2))
    
def model_data(ele_pos, charges, dip_cor,sigma=1,r0=.1, typ='step'):
    wave = np.zeros((ele_pos.shape[1],charges.shape[1]))
    for i in range(ele_pos.shape[1]):
        # print('ele_num: ', i)
        for n in range(charges.shape[0]):
            if typ=='step':
                wave[i] += (sigma/3*r0**3)*charges[n]/(np.sum((ele_pos[:,i]-dip_cor[:,n])**2))**(1/2)
            elif typ=='gauss':
                d = np.linspace(0,20,2)
                stdev = r0/3
                Z = np.sum(np.exp(-(d**2)/(2*stdev**2))/(np.sqrt(2*np.pi)*stdev)**3)
                wave[i] += (sigma/(4*np.pi))*charges[n]/(np.sum((ele_pos[:,i]-dip_cor[:,n])**2))**(1/2)
    return wave

def remove_points(mtrx, xpos, ypos, zpos, tp, thr):
    idx_to_remove = np.where(abs(mtrx[:,tp]) < thr)
    xpos = np.delete(xpos, idx_to_remove)
    ypos = np.delete(ypos, idx_to_remove)
    zpos = np.delete(zpos, idx_to_remove)
    mtrx = np.delete(mtrx[:,tp], idx_to_remove)
    return mtrx, xpos, ypos, zpos

def low_split(syg, wzor, thresh, dlugosc, thr = 1, skok = 1):
    if thr>0:
        wzor_bin = wzor> thresh
    else:
        wzor_bin = wzor< thresh
    n = int(dlugosc)
    lista_pocz =[]
    while n < (len(wzor) - 2*dlugosc): 
        if wzor_bin[n] == 1: # znajdowanie 1         
            npocz = n
            n+=int(dlugosc/10)
            while n<len(wzor_bin) and wzor_bin[n]== 1:
                n +=1
            if thr>0:
                npocz = np.argmax(wzor[npocz:n])+npocz
            else:
                npocz = np.argmin(wzor[npocz:n])+npocz
            lista_pocz.append(npocz)
            n += int(dlugosc*skok)
        else:
            n += 1
    return lista_pocz

def csd_plot(est_pot, est_csd, filename, Fs, xp, ss, sts, pics = 2, exp = 0):
    fig = py.figure(figsize=(10, 10), dpi=100)
    ind = np.linspace(0.5,30.5,16)+int(exp*10/2)
    if pics == 2:
        kcsdpic2 = py.subplot(pics,1,1)
        py.title('LFP '+ filename[:-4], fontsize = 10)
        py.ylabel('channel', rotation=90, fontsize = 12)
        show_mtrx = np.zeros((32+int(exp*10), np.shape(est_pot)[1]))
        if exp!=0: show_mtrx[int(exp*10/2):-int(exp*10/2)] =  est_pot
        else:  show_mtrx = est_pot
        pic1 = py.imshow(show_mtrx[::-1], extent=[ss, sts, 32+exp*10, 0],interpolation = 'none',
                         aspect = 'auto', vmin=-np.max(abs(show_mtrx)), 
                         vmax=np.max(abs(show_mtrx)), cmap='PRGn')
        py.xticks([])
        cbar = py.colorbar(pic1)
        cbar.ax.set_ylabel('amplitude [mV]',rotation=90,fontsize = 12)
        py.grid(linestyle='--', linewidth=0.5)
        layer_list = np.linspace(1,32,32)
        ymarks = np.round(layer_list, decimals = 0)
        kcsdpic2.set_yticks(ind)
        xtickNames = kcsdpic2.set_yticklabels(ymarks[::-2])
        py.setp(xtickNames, rotation=0, fontsize= 6)
        py.xlim(ss, sts)
    kcsdpic = py.subplot(pics,1,pics)
    py.title('CSD', fontsize = 10)
    pic2 = py.imshow(est_csd[::-1], extent=[ss, sts, 32+exp*10, 0], interpolation = 'none',
                     aspect = 'auto', cmap=cm.get_cmap('bwr'),
                     vmin=-np.max(abs(est_csd)), vmax=np.max(abs(est_csd)))
    cbar = py.colorbar(pic2)
    kcsdpic.set_yticks(ind)
    py.grid(linestyle='--', linewidth=0.5)
    xtickNames = kcsdpic.set_yticklabels(xp[::2], fontsize = 8)
    cbar.ax.set_ylabel('CSD [mA/mm]', rotation=90, fontsize = 10)
    py.xlabel('Time [ms]',fontsize = 12)
    py.xlim(ss, sts)
    py.tight_layout()
    return fig

def runningMean(x, N):
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
         y[ctr] = np.sum(x[ctr:(ctr+N)])
    return y/N

def decomp(mix_met,component = 0, method = 2 , comps = 3):
    if method == 0:
        nmf = NMF(n_components=8,init='nndsvd')#, random_state=0)
    elif method == 1:
        nmf = FastICA(n_components=comps, random_state = 0)
        nmf_1 = nmf.fit_transform(mix_met)
        A = nmf.mixing_
        A1 = A[:,0].reshape(len(A[:,0]),1)
        A2 = A[:,1].reshape(len(A[:,0]),1)
        if comps >2: A3 = A[:,2].reshape(len(A[:,0]),1)
    else: 
        nmf = PCA(n_components=comps)
        nmf.fit(mix_met)
        A = nmf.components_
        A1 = A[0,:].reshape(len(A[0,:]),1)
        A2= A[1,:].reshape(len(A[1,:]),1)
        A3 = A[2,:].reshape(len(A[2,:]),1)
    nmf_1 = nmf.fit_transform(mix_met)
    sygnal = nmf_1[:,component]
    frst_comp = np.dot(nmf_1[:,0].reshape(len(nmf_1),1), A1.T)
    scnd_comp = np.dot(nmf_1[:,1].reshape(len(nmf_1),1), A2.T)
    if comps>2: thrd_comp = np.dot(nmf_1[:,2].reshape(len(nmf_1),1), A3.T)
    if comps>2 and len(nmf_1[:,0])<len(A1): return sygnal, np.array([frst_comp,scnd_comp,thrd_comp]), np.array([nmf_1[:,0], nmf_1[:,1], nmf_1[:,2]]), np.array([A1, A2, A3])
    elif comps>2 and len(nmf_1[:,0]) >len(A1): return sygnal, np.array([frst_comp,scnd_comp,thrd_comp]), np.array([A1, A2, A3]), np.array([nmf_1[:,0], nmf_1[:,1], nmf_1[:,2]])
    elif comps ==2 and len(nmf_1[:,0]) >len(A1): return sygnal, np.array([frst_comp,scnd_comp]), np.array([A1, A2]), np.array([nmf_1[:,0], nmf_1[:,1]])
    else: return sygnal, np.array([frst_comp,scnd_comp]), np.array([nmf_1[:,0], nmf_1[:,1]]), np.array([A1, A2])

def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def power_phase(hfo_sweeps, hfo_sweeps2, ch_cut, name, df, save = 0, addname = '', Fss = 1000, sigslow=0):
    pol = int(hfo_sweeps.shape[2]/2)
    ph_cut = 100
    frag = int(Fss/ph_cut)
    f_inst = np.zeros((hfo_sweeps.shape[0], hfo_sweeps.shape[1]))
    f_inst2 = np.zeros(hfo_sweeps.shape[0])
    freq_power = np.zeros((hfo_sweeps.shape[0], hfo_sweeps.shape[1]))
    freq_power2 = np.zeros(hfo_sweeps.shape[0])
    dt2 = np.linspace(-sigslow.shape[1]/Fss, sigslow.shape[1]/Fss, 2*sigslow.shape[1]-1)
    dt = np.linspace(-2*frag/Fss, 2*frag/Fss, 4*frag-1)
    for ch in range(hfo_sweeps.shape[0]):
        for i in range(hfo_sweeps.shape[1]):
            freq, sp = welch(hfo_sweeps[ch, i], Fss)
            freq, sp_orig = welch(hfo_sweeps2[ch_cut, i], Fss, nperseg = ph_cut)
            xcorr = correlate(hfo_sweeps[ch_cut, i, pol-frag:pol+frag], hfo_sweeps[ch, i, pol-frag:pol+frag])
            rts = dt[xcorr.argmax()]
            period = 1/100
            f_inst[ch,i] = (360*(((0.5 + rts/period) % 1.0) - 0.5))
            if f_inst[ch, i]<-60: f_inst[ch, i] += 360
            else: f_inst[ch, i] = abs(f_inst[ch, i])
            freq_power[ch,i] = sp[np.argmax(sp)]
        xcorr = correlate(sigslow[ch_cut], sigslow[ch])
        rts2 = dt2[xcorr.argmax()]
        period2 = 1/1
        f_inst2[ch] = (360*(((0.5 + rts2/period2) % 1.0) - 0.5))
        if f_inst2[ch]<-60: f_inst2[ch] += 360
        else: f_inst2[ch] = abs(f_inst2[ch])
#        print(f_inst2[ch])
        freq2, sp2 = welch(sigslow[ch], Fss, nperseg = Fss*4)
        freq_power2[ch] = sp2[np.argmax(sp2[1:10])+1]
    fig = py.figure(figsize=(6, 6), dpi=240)
    py.subplot(121)
    ind = np.linspace(1,32,32)
    py.xlabel('angle [degrees]',fontsize = 15)
    mean_anglea = abs(np.mean(f_inst, axis =1) - np.mean(f_inst[ch_cut]))
    std_anglea = np.std(f_inst,axis=1)#/np.sqrt(hfo_sweeps.shape[1])
    py.plot(mean_anglea, ind ,marker = 'o',linewidth = 2,label = 'HFO', color= 'blue')
    py.plot(f_inst2, ind ,marker = 'o',linewidth = 2,label = 'delta', color = 'green')
    py.legend()
    py.fill_betweenx(ind, mean_anglea - std_anglea,mean_anglea +std_anglea,facecolor='blue', alpha=0.5)
#    py.xlim(-np.pi,np.pi)
    py.grid()
    py.subplot(122)
    py.grid()
#    py.xlabel('spectrum power',fontsize = 15)
    py.xlabel('maks power freq',fontsize = 15)
    mean_power = np.mean(freq_power, axis= 1)
    mean_power = mean_power/np.max(mean_power)
    std_power = np.std(freq_power, axis= 1)/np.sqrt(hfo_sweeps.shape[1])
#    mean_power = np.mean(maks_freq, axis= 1)
#    std_power = np.std(maks_freq, axis= 1)/np.sqrt(hfo_sweeps.shape[1])
    py.plot(mean_power, ind ,marker = 'o',linewidth = 2,label = 'HFO')
    py.fill_betweenx(ind ,mean_power - std_power,mean_power +std_power,color='blue', alpha=0.5)
    
    mean_power2 = freq_power2
    mean_power2 = mean_power2/np.max(mean_power2)
    py.plot(mean_power2, ind ,marker = 'o',linewidth = 2,label = 'delta', color = 'green')
    py.scatter([mean_power[ch_cut]], [ch_cut+1], color = 'red', s=60)
    py.legend()
    try: py.yticks(ind, df[str(int(name))].tolist()[4:36])
    except: py.yticks(ind, df[int(name)].tolist()[4:36])
    if save: 
        os.chdir('\\Users\\Wladek\\Dysk Google\\Figures for HFO in olfactory bulb\\pub2\\')
        fig.savefig(str(name) + addname + '_phase2.png')
        df_loc = pd.read_excel('phase_and_corr.xlsx')
        df_loc[str(name) + '_ph'] = mean_anglea
        df_loc[str(name) + 'delta_ph'] = f_inst2
        df_loc[str(name) + '_pw'] = mean_power
        df_loc[str(name) + '_delta_pw'] = mean_power2
        df_loc.to_excel('phase_and_corr.xlsx', sheet_name='sheet1', index=False)
    return freq, mean_anglea, f_inst2, mean_power, mean_power2

def gam_and_hfo(HFO, freq_lim, ch_list, Fs, vmax = 1e-4, ss=0, sts=50):
    py.figure()
    py.subplot(211)
    freq, time_spec, spec_mtrx = spectrogram(HFO[ch_list[0]], Fs, nperseg=10*1024)
    ind_g1 = np.where(freq < 90)[-1][-1]
    ind_g2 = np.where(freq > 60)[-1][0]
    ind_h1 = np.where(freq < 180)[-1][-1]
    ind_h2 = np.where(freq > 100)[-1][0]
    sum_spec = np.sum(spec_mtrx[ind_g2:ind_g1], axis = 0)
    py.plot(time_spec,sum_spec, label = 'gamma')
    sum_spec = np.sum(spec_mtrx[ind_h2:ind_h1], axis = 0)
    py.plot(time_spec,sum_spec, label = 'HFO')
    py.legend()
    py.xlim(ss, sts)
    py.axhline()
    py.ylim(0, vmax)
    py.subplot(212)
    freq, time_spec, spec_mtrx = spectrogram(HFO[ch_list[1]], Fs, nperseg=Fs)
    sum_spec = np.sum(spec_mtrx[ind_g2:ind_g1], axis = 0)
    py.plot(time_spec,sum_spec, label = 'gamma' + ' 60 - 120 Hz')
    sum_spec = np.sum(spec_mtrx[ind_h2:ind_h1], axis = 0)
    py.plot(time_spec,sum_spec, label = 'HFO'  + ' 120 - 180 Hz')
    py.legend()
    py.ylim(0, vmax)
    py.xlim(ss, sts)
    return freq

import scipy.stats as ststats
def randsample(x,ile):
    ind = ststats.randint.rvs(0,len(x),size = ile)
    y = x[ind]
    return y

def stats(W, L):
    proc = np.arange(0,100,10)
    orig_mean = abs(W.mean(axis=0) - L.mean(axis=0))
    worek = np.concatenate((L, W))
    Nboots = 1000
    A=np.zeros((Nboots, L.shape[1]))
    for i in range(Nboots):
        if i*100/Nboots in list(proc): print('percent done:', i*100/Nboots)
        grupa1 = randsample(worek, W.shape[0])
        grupa2 = randsample(worek, L.shape[0])
        A[i]= abs(np.mean(grupa1, axis=0) - np.mean(grupa2, axis=0))>=(orig_mean)
    p_mtrx = A.sum(axis=0)/Nboots
    return p_mtrx

def func(r,q1,r1,q2,r2):
    return q1/(abs(r1-r)+1e-10)+q2/(abs(r2-r)+1e-10)
if __name__ == "__main__":
    q1,r1=-.1,1
    q2,r2=.02,4
    r=np.linspace(-100,100,1000)+1e-5
    py.plot(r,func(r,q1,r1,q2,r2))
    py.plot([r2, r2], [0,q2], lw=5, color='red')
    py.plot([r1, r1], [0,q1], lw=5, color='blue')
    py.axhline(0, ls='--', color='darkgreen')
    
    