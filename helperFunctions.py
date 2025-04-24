import numpy as np

from scipy.constants import physical_constants
from cycler import cycler


pres = {'axes.titlecolor':'white',
        'axes.edgecolor':'white', 
        'xtick.color':'white', 
        'ytick.color':'white', 
        'figure.facecolor':'none', 
        'axes.labelcolor':'white',
        'axes.facecolor':'none', 
        'legend.facecolor':'white',
        'legend.framealpha':0.2,
        'legend.labelcolor':'white',
        'figure.dpi':200,
        'axes.prop_cycle':cycler(color=['deepskyblue', 'orange', 'yellowgreen', 'tomato', 'orchid', 'w']),
        }

post = {'figure.dpi':300,
        'axes.prop_cycle':cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#000000']),
       }

regu = {'figure.dpi':100,
        'axes.prop_cycle':cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#000000']),
       }


def scaleCS (cs_ref, fa, fa_ref=1e4):
    '''
    function to scale the cross-section to ther values of fa than simulated with Madgraph5
    ! fa_ref is a factor of 4 smaller than the simulated fa_MG!
    '''
    return cs_ref * (fa_ref/fa)**2
    

def plotNaturalness (ax, **kwargs): 
    
    # naturalness line
    m_a_n = np.logspace(np.log10(2.3e-2), np.log10(7e-1), 1000)
    ax.plot(m_a_n, 1 / (4e5 * (1)**2 * 0.2/m_a_n), '-.', c='#663300')
    ax.text(6e-2, 5e-7, 'naturalness', c='#663300', rotation=18, size=10)


def plotConstraints (ax, which, **kwargs):

    ###########################
    # EXISTING CONSTRAINTS
    ###########################

    if which=='beamdumps':
        data = np.genfromtxt('data/limit_data/constraints/BeamDump.txt').T
        ax.fill(data[0]/1e9, data[1], c='#DDD')

    elif which=='opal_2017':
        data = np.genfromtxt('data/limit_data/constraints/OPAL_2017.txt').T
        ax.fill(data[0]/1e9, data[1], c='#DDD')
    
    elif which=='primex_2019':
        data = np.genfromtxt('data/limit_data/constraints/PrimEx_2019.txt').T
        ax.fill(*data, c='#DDD')
    
    elif which=='na64_2020':
        data = np.genfromtxt('data/limit_data/constraints/NA64_2020.txt', delimiter=', ').T
        ax.fill(*data, c='#DDD')

    elif which=='belle2_2020':
        data = np.genfromtxt('data/limit_data/constraints/Belle2_2020.txt').T
        ax.fill(data[0], data[2], c='#DDD')

    elif which=='bes3_2023':
        data = np.genfromtxt('data/limit_data/constraints/BESIII_2023.txt').T
        ax.fill(data[0]/1e9, data[1], c='#DDD')

    elif which=='miniboone_2023':
        data = np.genfromtxt('data/limit_data/constraints/MiniBooNE_2023.txt').T
        ax.fill(data[0]/1e9, data[1], c='#DDD')

    elif which=='faser_2025':
        data = np.genfromtxt('data/limit_data/constraints/FASER_2025.txt').T
        ax.fill(data[0]/1e9, data[1], c='#DDD')

    
    ###########################
    # PROJECTIONS
    ###########################

    elif which=='na62_1e18pot':
        data = np.genfromtxt('data/limit_data/projections/NA62_dump_1e18_POT.txt').T
        ax.plot(*data, '--', c='gray', lw=0.7, label='NA62')
    
    elif which=='na64_5e12eot':
        data = np.genfromtxt('data/limit_data/projections/NA64_5e12_EOT.txt').T
        ax.plot(*data, c='gray', lw=0.7, dashes=[1,1], label='NA64')
    
    elif which=='belle2_proj':
        data = np.genfromtxt('data/limit_data/projections/Belle2a.txt').T
        ax.plot(*data, c='gray', dashes=[3,1,1,1], lw=0.7)
        data = np.genfromtxt('data/limit_data/projections/Belle2b.txt').T
        ax.plot(*data, c='gray', dashes=[3,1,1,1], lw=0.7, label='Belle-II')
    
    elif which=='faser_proj':
        data = np.genfromtxt('data/limit_data/projections/FASER.txt').T
        ax.plot(*data, c='gray', lw=0.7, dashes=[3,1,1,1,1,1])
        data = np.genfromtxt('data/limit_data/projections/FASER2.txt').T
        ax.plot(*data, c='gray', lw=0.7, label='FASER(2)', dashes=[3,1,1,1,1,1])

    elif which=='ship_proj':
        data = np.genfromtxt('data/limit_data/projections/SHiP.txt', delimiter=', ').T
        ax.plot(*data, c='gray', lw=0.7, label='SHiP', dashes=[2,1])

    elif which=='primex_proj':
        data = np.genfromtxt('data/limit_data/projections/primex_all.txt').T
        ax.plot(*data, c='gray', lw=0.7, dashes=[3,1,1,1,1,1,1,1], label='PrimEx')

    elif which=='gluex_fb':
        data = np.genfromtxt('data/limit_data/projections/gluex_fb.txt').T
        ax.plot(*data, c='gray', lw=0.7, dashes=[3,1,3,1,1,1,1,1], label='GlueX')

    else:
        print(f"dataset '{which}' does not exist")
       