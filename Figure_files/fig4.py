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
from scipy.stats import sem,wilcoxon,shapiro, permutation_test, ttest_1samp
from mlxtend.evaluate import permutation_test
import scipy.io
py.close('all')

mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['xtick.major.size'] = 1
mpl.rcParams['ytick.major.size'] = 1

def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def exp_design(po, pos, name):
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    img = py.imread(loadir2+name)
    ax.imshow(img, aspect='auto', extent=[0,1,0,1])
    # py.xlim(0.1,0.9)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    py.xticks([]), py.yticks([])

def lfp_map_stat(po, pos):
    global lfp, lfp_th_est, collect
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    loadir='./mats/an_'
    typ,typ_ar = 'pots', 'pots'
    mnoz_NN=0.002
    mnoz=1
    mnoz2=0.02
    lfp = np.array([scipy.io.loadmat(loadir+'sov16.mat')[typ_ar]*mnoz_NN, 
                    scipy.io.loadmat(loadir+'sov17.mat')[typ_ar]*mnoz_NN,
                    scipy.io.loadmat(loadir+'sov18.mat')[typ_ar]*mnoz_NN, 
                    scipy.io.loadmat(loadir+'sov01.mat')[typ]*mnoz,
                    scipy.io.loadmat(loadir+'sov02.mat')[typ]*mnoz,
                    scipy.io.loadmat(loadir+'sov11.mat')[typ]*mnoz,
                    scipy.io.loadmat(loadir+'sov06.mat')[typ]*mnoz,
                    scipy.io.loadmat(loadir+'sov09.mat')[typ]*mnoz,
                    scipy.io.loadmat(loadir+'sov19.mat')[typ]*mnoz2,
                    scipy.io.loadmat(loadir+'sov20.mat')[typ]*mnoz2,
                    scipy.io.loadmat(loadir+'sov21.mat')[typ]*mnoz2])
    typ,typ_ar = 'pots_est_th', 'pots_est_th'
    lfp_th_est = np.array([scipy.io.loadmat(loadir+'sov16.mat')[typ_ar]*mnoz_NN, 
                           scipy.io.loadmat(loadir+'sov17.mat')[typ_ar]*mnoz_NN,
                           scipy.io.loadmat(loadir+'sov18.mat')[typ_ar]*mnoz_NN, 
                           scipy.io.loadmat(loadir+'sov01.mat')[typ]*mnoz,
                           scipy.io.loadmat(loadir+'sov02.mat')[typ]*mnoz,
                           scipy.io.loadmat(loadir+'sov11.mat')[typ]*mnoz,
                           scipy.io.loadmat(loadir+'sov06.mat')[typ]*mnoz,
                           scipy.io.loadmat(loadir+'sov09.mat')[typ]*mnoz,
                           scipy.io.loadmat(loadir+'sov19.mat')[typ]*mnoz2,
                           scipy.io.loadmat(loadir+'sov20.mat')[typ]*mnoz2,
                           scipy.io.loadmat(loadir+'sov21.mat')[typ]*mnoz2])
    typ,typ_ar = 'pots_est_crtx', 'pots_est_crtx'
    lfp_crtx_est = np.array([scipy.io.loadmat(loadir+'sov16.mat')[typ_ar]*mnoz_NN, 
                           scipy.io.loadmat(loadir+'sov17.mat')[typ_ar]*mnoz_NN,
                           scipy.io.loadmat(loadir+'sov18.mat')[typ_ar]*mnoz_NN, 
                           scipy.io.loadmat(loadir+'sov01.mat')[typ]*mnoz,
                           scipy.io.loadmat(loadir+'sov02.mat')[typ]*mnoz,
                           scipy.io.loadmat(loadir+'sov11.mat')[typ]*mnoz,
                           scipy.io.loadmat(loadir+'sov06.mat')[typ]*mnoz,
                           scipy.io.loadmat(loadir+'sov09.mat')[typ]*mnoz,
                           scipy.io.loadmat(loadir+'sov19.mat')[typ]*mnoz2,
                           scipy.io.loadmat(loadir+'sov20.mat')[typ]*mnoz2,
                           scipy.io.loadmat(loadir+'sov21.mat')[typ]*mnoz2])
    
    timepoint=150
    collect_ex1, collect_ex2 = [], []
    collect_ex1c, collect_ex2c = [], []
    for i in range(11): 
        ex1 = lfp[i][th_channel_list[i],timepoint]#/min_scale
        ex2 = lfp_th_est[i][th_channel_list[i],timepoint]#/min_scale
        ex1c = lfp[i][crtx_list[i],timepoint]#/min_scale
        ex2c = lfp_crtx_est[i][crtx_list[i],timepoint]#/min_scale
        collect_ex1.append(ex1),collect_ex2.append(ex2)
        collect_ex1c.append(ex1c),collect_ex2c.append(ex2c)
        py.plot([-.25,1.25], [ex1,ex2],'--o', color='k', lw=2)
        # py.text(1,ex2,str(sov_labels[i]))
    # print(shapiro(collect_ex1),
           # shapiro(collect_ex2))
    # test = permutation_test(np.array([collect_ex1,collect_ex2]),statistic)
    # pvalue = test.pvalue
    pvalue = permutation_test(collect_ex1,collect_ex2, paired=True, seed=0, num_rounds=10000)
    print('figureB stat:', pvalue)
    py.text(.2, 0.07, '**', fontsize=30)
    py.xticks([-.25,1.25], ['EP', 'Recon.'])
    ax.set_ylabel('Potential (mV)')
    py.xlim(-1,2)
    ax2 = fig.add_subplot(gs[pos[0]:pos[1], pos[2]+4:pos[3]+4])
    set_axis(ax2, 0, po[1], letter= 'C')
    
    dif_ex1ex2 = -(np.array(collect_ex1)-np.array(collect_ex2))#/(np.array(collect_ex1))
    dif_ex1ex2c = -(np.array(collect_ex1c)-np.array(collect_ex2c))#/(np.array(collect_ex1c))
    
    # ax2.bar([2.5], dif_ex1ex2.mean(), yerr = sem(dif_ex1ex2), width=1,facecolor='orange',
           # edgecolor='red',alpha=.5,hatch=r"//",error_kw=dict(lw=5, capsize=5, capthick=2))
    ax2.boxplot(dif_ex1ex2, positions=[2.5])
    # ax2.bar([3.5], dif_ex1ex2c.mean(), yerr = sem(dif_ex1ex2c), width=1,facecolor='navy',
           # edgecolor='skyblue',alpha=.5,hatch=r"//",error_kw=dict(lw=5, capsize=5, capthick=2))
    ax2.boxplot(dif_ex1ex2c, positions=[3.5])
    for i in range(11):
        py.plot([2.3+i*0.04], [dif_ex1ex2[i]],'o', color='red', lw=2)
        py.plot([3.3+i*0.04], [dif_ex1ex2c[i]],'o', color='blue', lw=2)
        # py.text(3.3+i*0.04, dif_ex1ex2c[i], str(sov_labels[i]))
    py.xticks([2.5,3.5],['Thal.', 'Cort.'])
    # py.text(2.9, 0.12, '*', fontsize=30)
    py.text(2.4, -0.06, '*', fontsize=30)
    ax2.set_ylabel('Corrected Potential (mV)')
    print(shapiro(dif_ex1ex2c),
          shapiro(dif_ex1ex2))
    # test = permutation_test([dif_ex1ex2,dif_ex1ex2c],statistic)
    pvalue = permutation_test(dif_ex1ex2,dif_ex1ex2c, paired=True, seed=0, num_rounds=10000)
    print('figureC stat:', pvalue)
    print('figureC stat th:', ttest_1samp(dif_ex1ex2, 0))
    print('figureC stat crtx:', ttest_1samp(dif_ex1ex2c, 0))
    # py.xlim(-.5,4)
    

def cor_map_stat(po, pos, typ='cs_th',typ2='cs_crtx', name='', title=''):
    global df, th, mn_crtx, mn_ths, mn_th
    tps, lp = list(np.arange(28)), 1
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    loadir='./mats/an_'
    cor_score_th = np.array([scipy.io.loadmat(loadir+'sov16.mat')[typ], 
                             scipy.io.loadmat(loadir+'sov17.mat')[typ],
                             scipy.io.loadmat(loadir+'sov18.mat')[typ], 
                             scipy.io.loadmat(loadir+'sov01.mat')[typ],
                             scipy.io.loadmat(loadir+'sov02.mat')[typ],
                             scipy.io.loadmat(loadir+'sov11.mat')[typ],
                             scipy.io.loadmat(loadir+'sov06.mat')[typ],
                             scipy.io.loadmat(loadir+'sov09.mat')[typ],
                             scipy.io.loadmat(loadir+'sov19.mat')[typ],
                             scipy.io.loadmat(loadir+'sov20.mat')[typ],
                             scipy.io.loadmat(loadir+'sov21.mat')[typ]])
    cor_score_crtx = np.array([scipy.io.loadmat(loadir+'sov16.mat')[typ2], 
                             scipy.io.loadmat(loadir+'sov17.mat')[typ2],
                             scipy.io.loadmat(loadir+'sov18.mat')[typ2], 
                             scipy.io.loadmat(loadir+'sov01.mat')[typ2],
                             scipy.io.loadmat(loadir+'sov02.mat')[typ2],
                             scipy.io.loadmat(loadir+'sov11.mat')[typ2],
                             scipy.io.loadmat(loadir+'sov06.mat')[typ2],
                             scipy.io.loadmat(loadir+'sov09.mat')[typ2],
                             scipy.io.loadmat(loadir+'sov19.mat')[typ2],
                             scipy.io.loadmat(loadir+'sov20.mat')[typ2],
                             scipy.io.loadmat(loadir+'sov21.mat')[typ2]])
    n_rats = cor_score_th.shape[0]
    th_th, crtx_th = np.zeros((n_rats,len(tps)+2,lp)), np.zeros((n_rats,len(tps)+2,lp))
    th_crtx, crtx_crtx = np.zeros((n_rats,len(tps)+2,lp)), np.zeros((n_rats,len(tps)+2,lp))
    for n,i in enumerate(tps):
        if n not in [2,3,4]:
            crtx_th[:, n] = [cor_score_th[ii][ch, i:i+lp] for ii,ch in enumerate(crtx_list)]
            
            th_th[:, n] = [cor_score_th[ii][ch, i:i+lp] for ii,ch in enumerate(th_channel_list)]
    
            crtx_crtx[:, n] = [cor_score_crtx[ii][ch, i:i+lp] for ii,ch in enumerate(crtx_list)]
            
            th_crtx[:, n] = [cor_score_crtx[ii][ch, i:i+lp] for ii,ch in enumerate(th_channel_list)] 
    
    time_scale=np.linspace(-5,24,30)
    if 'Cortical' in title:
        mn_crtx, std_crtx = crtx_crtx.mean(axis=0)[:,0],crtx_crtx.std(axis=0)[:,0]/(n_rats**(.5))
        py.plot(time_scale, mn_crtx,'-o', color='navy', lw=2, label='Cortical sources')
        py.fill_between(time_scale, mn_crtx-std_crtx, mn_crtx+std_crtx, alpha=.3, color='blue')
        mn_th, std_th = crtx_th.mean(axis=0)[:,0], crtx_th.std(axis=0)[:,0]/(n_rats**(.5))
        py.plot(time_scale, mn_th,'-^', color='brown', lw=2,label='Thalamic sources')
        py.fill_between(time_scale, mn_th-std_th, mn_th+std_th, alpha=.3, color='brown')
    if 'Thalamic' in title:
        mn_crtx, std_crtx = th_crtx.mean(axis=0)[:,0],th_crtx.std(axis=0)[:,0]/(n_rats**(.5))
        py.plot(time_scale, mn_crtx,'-o', color='navy', lw=2, label='Cortical sources')
        py.fill_between(time_scale, mn_crtx-std_crtx, mn_crtx+std_crtx, alpha=.3, color='blue')
        mn_th, std_th = th_th.mean(axis=0)[:,0], th_th.std(axis=0)[:,0]/(n_rats**(.5))
        py.plot(time_scale, mn_th,'-^', color='brown', lw=2,label='Thalamic sources')
        mn_ths = mn_th[6:25]
        print('mn and sem th:', mn_ths.mean(),sem(mn_ths),
              mn_ths.max(), mn_ths.min())
        print('mn and sem crtx:', mn_crtx.max(), std_crtx[np.argmax(mn_crtx)], np.argmax(mn_crtx)+1)
        print('mn and sem thmax:', mn_ths.max(), std_th[np.argmax(mn_ths)], np.argmax(mn_ths)+1)
        print('mn and sem thmax:', mn_ths.min(), std_th[np.argmin(mn_ths)], np.argmin(mn_ths)+1)
        # print(std_th[np.argmin(mn_th[:25])],mn_th[np.argmin(mn_th[:25])])
        py.fill_between(time_scale, mn_th-std_th, mn_th+std_th, alpha=.3, color='brown')
        # sov_labels = [16,17,18,1,2,3,6,9,19,20,21]
        # for i in range(0,11,1): 
        #     py.plot(time_scale, th_th[i], lw=2, label = sov_labels[i])
        #     # py.plot(time_scale, crtx[i], lw=2, color='navy')
    py.axvspan(-5, -2, alpha=.3, color='grey')
    ax.arrow(-2,-.3,-2.4,0, width=.05, head_width=.3, color='k')
    ax.set_ylabel('Correlation')
    if 'Cort' not in title:
        ax.set_xlabel('Time (ms)')
    py.grid()
    py.legend(loc=4) 
    py.title(title)
    py.ylim(-1.1,1.1), py.xlim(-5,22)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    return th_th
    
def ex_lfp(po, pos, rat, div=1, div2=1, ch_th=1, ch_crtx=1):
    global recon_th
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    import scipy.io
    pots = scipy.io.loadmat('./mats/an_sov'+rat+'.mat')['pots']
    recon_crtx = scipy.io.loadmat('./mats/an_sov'+rat+'.mat')['pots_est_crtx']
    recon_th = scipy.io.loadmat('./mats/an_sov'+rat+'.mat')['pots_est_th']
    Fs = 10000
    loc_time = np.linspace(-5,24.9, pots.shape[1])
    pots1 = (pots[ch_crtx])#-pots[ch_crtx,:25].mean())
    # min_scale = abs(pots1[100:].min()/2)
    if rat in ['19,20,21']:
        min_scale=50
    if rat in ['16','17','18']:
        min_scale=500
    else:
        min_scale=1
    pots1/=min_scale
    rec_cortex = (recon_crtx[ch_crtx])/min_scale
    pots2 = (pots[ch_th]-pots[ch_th,:25].mean())/min_scale
    py.plot(loc_time, pots1, color='navy',label='Cortical EP',lw=3)
    py.plot(loc_time, rec_cortex, color='skyblue', ls='--',label='Cortical recon.',lw=3)
    ax.tick_params(axis='y', labelcolor='navy')
    py.legend(loc=4)
    ax2 = ax.twinx()
    color = 'orange'
    # pots2 = pots[ch_th]-pots[ch_th,:25].mean()
    ax2.plot(loc_time, pots2, color=color,label='Thalamic EP',lw=3)
    ax2.plot(loc_time, (recon_th[ch_th]-pots[ch_th,:25].mean())/min_scale, color='red', 
             ls='--',label='Thalamic recon.',lw=3)
    ax2.set_ylim(-.25,.25), ax.set_ylim(-2.5,2.5)
    ax2.tick_params(axis='y', labelcolor='orange')
    # ax2.text(3.4,-.25, '*')
    # ax2.text(9.2,-.5, '**')
    ax.set_xlabel('Time (ms)'),ax.set_ylabel('Potential (mV)')
    ax2.set_xlim(-5,23)
    py.axvline(0, ls='--', lw = 2, color='grey')
    py.axvline(loc_time[150], ls='--', lw = 2, color='k')
    ax2.legend(loc=1)
    # th_pot = mpatches.Patch(color=color, label='Thalamic EP')
    # crtx_pot = mpatches.Patch(color='navy', label='Cortical EP')
    # th_recon = mpatches.Patch(color='red', label='Thalamic recon.')
    # crtx_recon = mpatches.Patch(color='skyblue', label='Cortical recon.')
    # py.legend(handles=[th_pot, crtx_pot, th_recon, crtx_recon], ncol=1,loc=4, frameon = False, fontsize = 10)
    return min_scale
    
loadir='./fig4_files/'
loadir2='./fig6_files/'
channel=130

# exp_design(('A',1.07), (0,6,0,8), 'th_space.png')
# correlation_map(('B',1.08), (7,16,0,10), 'cor_score_th18.npy')
th_channel_list = [113,157,151,26,26,33,16,14,123,133,129]


# th_channel_list = [113,151,151,27,26,33,14,14,124,134,130]

crtx_list  = [60,78,78,5,6,7,7,5,320,330,320]
sov_labels = [16,17,18,'01','02','11','06','09',19,20,21]
# for i in range(3,4):
# py.tick_params(axis='both', which='minor', labelsize=6)
for num in range(8,9):
    fig = py.figure(figsize=(12,10), dpi=200)
    gs = fig.add_gridspec(14, 11)
    min_scale = ex_lfp(('A',1.05), (0,5,0,5), rat=str(sov_labels[num]), ch_th=th_channel_list[num],ch_crtx=crtx_list[num])
    mpl.rcParams['axes.spines.right'] = False
    lfp_map_stat(('B',1.05), (7,11,0,2))
    cor_map_stat(('D',1.05), (0,5,7,14), title='Cortical channels')
    cor_map_stat(('E',1.05), (6,11,7,14), title='Thalamic channels')
    
    py.savefig('fig4_new'+str(sov_labels[num]))
    # py.close()
 