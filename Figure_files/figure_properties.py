import matplotlib.pyplot as plt

def set_axis(ax, x, y, letter=None):
    ax.text(
        x,
        y,
        letter,
        fontsize=20,
        weight='bold',
        transform=ax.transAxes)
    return ax

def pval(pvalue):
    if pvalue<0.05: nap = '*'
    else: nap='n.s.'
    if pvalue<0.01: nap='**' 
    if pvalue<0.001: nap='***'
    return nap

siz=14
plt.rcParams.update({
    'xtick.labelsize': siz,
    'xtick.major.size': siz,
    'ytick.labelsize': siz,
    'ytick.major.size': siz,
    'font.size': siz,
    'axes.labelsize': siz,
    'axes.titlesize': 12,
    'axes.titlepad' : 5,
    'legend.fontsize': 10,
    'figure.subplot.wspace': 0,
    'figure.subplot.hspace': 0.2,
    'figure.subplot.left': 0.1,
})

