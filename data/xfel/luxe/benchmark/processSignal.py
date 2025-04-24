import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot as root

import os
import h5py
import time
import gzip

from scipy.constants import physical_constants
from glob import glob
from multiprocessing import Pool
from IPython.display import clear_output
from xml.etree.ElementTree import parse as xmlParse

setup = False

madgraphDir = '/home/sivo/Documents/LUXE/40_simulations/NPOD_physicsCaseExt/data/madgraph/ilc'

path = '/home/sivo/Documents/lcvision/40_simulations/xfel/luxe/benchmark/'

setupFile = path+'setup.npz'



#####################################
#  CONFIGURATION
#####################################

if setup:

    # number of electrons per bunch
    N_e = 1.56e9
    
    # number of bunch crossings
    N_bx = 1e7 * 1
    
    # dump length in [m]
    L_d = 2
    
    # decay volume length in [m]
    L_v = 10
    
    # max. radial distance of decay photons on detector plane in [m]
    R_max = 5
    
    # min. energy of decay photons in [GeV]
    E_min = 0.5
    
    # ALP mass and coupling in [GeV]
    G_A = np.logspace(np.log10(3e-8), np.log10(3e-3), 61)

else:

    L_d = np.load(setupFile)['L_d']
    L_v = np.load(setupFile)['L_v']
    R_max = np.load(setupFile)['R_max']
    E_min = np.load(setupFile)['E_min']
    G_A = np.load(setupFile)['G_A']



#####################################
#  CONSTANTS
#####################################

# reduced Planck constant in [GeV s]
h_bar = physical_constants['reduced Planck constant in eV s'][0] / 1e9

# speed of light in vacuum in [m/s]
c = physical_constants['speed of light in vacuum'][0]

# electron mass in [MeV]
m_e = physical_constants['electron mass energy equivalent in MeV'][0]



#####################################
#  FUNCTIONS
#####################################

# function to scale cross-section to other values of fa than simulated with Madgraph5
# !!! apparently a factor of 4 is in the MG model
# !!! so even the simulated fa_ref=4e4, the value here should be fa_ref=1e4
def scaleCS (cs_ref, fa, fa_ref=1e4):
    return cs_ref * (fa_ref/fa)**2

def getLHEevent (file):

    # if file is zipped, unzip first
    if file[-2:]=='gz':
        file = gzip.open(file, 'rb')

    # load the file and get the root
    tree = xmlParse(file)
    root = tree.getroot()
    
    # remove the header and empty lines and split into events
    events = [root[i+2].text.split('\n')[1:-1][1:] for i in range(len(root[2:]))]
    
    # convert the particle string into data floats
    events = np.array([[parameter.split() for parameter in particle] for particle in events], dtype=float)
    
    return events
    
def track_job(job, update_interval=2):
    # total number of jobs
    jobLength = job._number_left
    while job._number_left > 0:
        print('\rTasks (Chunks) remaining = {0}({1}) / {2}({3})     '.format(
        job._number_left*job._chunksize,job._number_left,jobLength*job._chunksize,jobLength), end='')
        time.sleep(update_interval)
    clear_output()
    print('Task finished')

def calcALPphotons(g_a):

    # ALP decay constant in [GeV]
    Lambda_a = 1 / g_a
    
    # calculate the decay rate of the ALP into two photons in [GeV]
    Gamma_a = m_a**3 / 64 / np.pi / Lambda_a**2
    
    # calculate the lifetime of the ALP in [s]
    tau_a = h_bar / Gamma_a

    Ndet_single = np.zeros_like(ePhotons)

    # iterate through all the Compton photon energies
    for i,ePhoton in enumerate(ePhotons):
    
        #print('\r{:3d} / {:3d}'.format(i+1, len(ePhotons)), end='')
        #print('progress: {:3d} / {:3d}'.format(i+1, len(nPhotons)))
        
        # madgraph file with the corresponding dataset
        filename = f'{madgraphDir}/signalSim/unweightedEvents_mass{m_a:08.6f}GeV_beam{ePhoton:05.1f}GeV_sig.lhe.gz'

        fileExt = os.path.splitext(filename)[1]

        # only process the file if there are any photons at that energy
        if nPhotons[i]>0:
    
            # calculate the ALP momentum from incoming Compton photon in [GeV]
            p_a = np.sqrt(ePhoton**2 - m_a**2)
    
            # calculate the decay length of the ALP in [m]
            L_a = tau_a * c * p_a / m_a
    
            # randomly draw decay vertex distance in [m]
            r_a = np.random.exponential(L_a, size=nPhotons[i])

            # randomly select some particle events to weight with Compton photon spectrum
            choices = np.random.choice(10000, nPhotons[i], replace=True)

            if fileExt=='.root':

                # get the particle data from the ROOT file
                with root.open(filename) as f:
                    particles = f['LHEF;1']['Particle'].arrays()
                    particles = particles[choices]
                    
                # outgoing photons from the ALP decay
                gamma_1 = np.column_stack((particles['Particle.Px'][:,2], particles['Particle.Py'][:,2], particles['Particle.Pz'][:,2], particles['Particle.E'][:,2]))
                gamma_1 = pd.DataFrame(gamma_1, columns=['Px','Py','Pz','E'])
                gamma_2 = np.column_stack((particles['Particle.Px'][:,3], particles['Particle.Py'][:,3], particles['Particle.Pz'][:,3], particles['Particle.E'][:,3]))
                gamma_2 = pd.DataFrame(gamma_2, columns=['Px','Py','Pz','E'])

            elif fileExt=='.gz' or fileExt=='.lhe':

                # get the particle data from the LHE file
                particles = getLHEevent(filename)
    
                # outgoing photons from the ALP decay
                gamma_1 = particles[choices,2,6:10]
                gamma_1 = pd.DataFrame(gamma_1, columns=['Px','Py','Pz','E'])
                gamma_2 = particles[choices,3,6:10]
                gamma_2 = pd.DataFrame(gamma_2, columns=['Px','Py','Pz','E'])
            
            # calculate the normalized direction of the alp momentum
            p_a_dir = (gamma_1+gamma_2).drop(columns='E')
            p_a_norm = np.sqrt(p_a_dir.Px**2+p_a_dir.Py**2+p_a_dir.Pz**2)
            p_a_dir = p_a_dir.apply(lambda row, norm: row / norm, args=(p_a_norm,), axis=0)
            
            # calculate the position vector of the ALP decay
            R_a = p_a_dir.apply(lambda row, norm: row * norm, args=(r_a,), axis=0)
            R_a.columns = ['x', 'y', 'z']
                
            # select only the events decaying between the dump and the detector
            mask = (R_a.z>L_d) & (R_a.z<L_d+L_v)
    
            # only further process if any ALP could leave the dump
            if np.sum(mask)>0:
            
                # select the events
                R_a = R_a[mask]
                gamma_1 = gamma_1[mask]
                gamma_2 = gamma_2[mask]
                
                # calculate the normalized photon direction
                gamma_1_dir = gamma_1.apply(lambda row, norm: row / norm, args=(gamma_1.E,), axis=0).drop(columns='E')
                gamma_2_dir = gamma_2.apply(lambda row, norm: row / norm, args=(gamma_2.E,), axis=0).drop(columns='E')
                
                # calculate the radial position of the gamma hit in [m]
                R_g1_r = (L_d+L_v-R_a.z) * np.sqrt(gamma_1_dir.Px**2+gamma_1_dir.Py**2) / gamma_1_dir.Pz            
                R_g2_r = (L_d+L_v-R_a.z) * np.sqrt(gamma_2_dir.Px**2+gamma_2_dir.Py**2) / gamma_2_dir.Pz
        
                # select only events that have a certain energy and radial distribution at the detector
                mask = (R_g1_r<R_max) & (R_g2_r<R_max) & (gamma_1.E>E_min) & (gamma_2.E>E_min)
            
            # count the number of events that passed the constraints
            Ndet_single[i] = np.sum(mask)
            
    return Ndet_single



#####################################
#  LUMINOSITY
#####################################

if setup: 

    # atomic mass number of tungsten
    A_w = 183.84
    
    # density of tungsten in [g/cm³]
    rho_w = 19.30 
    
    # radiation length of tungsten in [cm]
    X0_w = 0.3504 
    
    # nucleon mass in [g]
    m_0 = 1.66e-24
    
    # effective / integrated luminosity in [1/cm²]
    L_eff = N_e * N_bx * (9*rho_w*X0_w) / (7*A_w*m_0)

    # effective luminosity in [1/pb]
    L_eff_pb = L_eff / 1e36

else:
    
    L_eff_pb = np.load(setupFile)['L_eff_pb']



#####################################
#  COMPTON SPECTRUM
#####################################

if setup:
    
    N_sim = 1e6

    bins_hw = 0.75
    bins = np.arange(2*bins_hw, 30+0.01, 2*bins_hw)-bins_hw

    filename = '../sim_001_particles.h5'
    with h5py.File(filename, 'r') as f:
        weight = f['final-state/photon/weight'][()]
        photonEnergy = f['final-state/photon/momentum'][()][:,0]
    
    fig, ax = plt.subplots(figsize=(6,4))

    n_norm,bins,_ = ax.hist(photonEnergy, bins, weights=weight/N_e, histtype='step')
        
    # number of particles to simulate for each beam energy
    n_sim = n_norm * N_sim / n_norm.max()
    n_sim = np.array(n_sim, dtype=int)

    beamEnergies = bins[1:]-bins_hw
    beamEnergies = beamEnergies.round(1)
        
    ax.set(xlabel='Compton photon energy [GeV]', yscale='log')
    fig.tight_layout()
    plt.show()

else:

    N_sim = np.load(setupFile)['N_sim']
    n_norm = np.load(setupFile)['n_norm']
    n_sim = np.load(setupFile)['n_sim']



#####################################
#  CROSS-SECTION
#####################################

if setup:

    filelist = sorted(glob(f'{madgraphDir}/runTags/*'))
    
    masses = []
    for i,file in enumerate(filelist):
        filename = file.split('/')[-1]
        mass = float(filename.split('_')[-3][4:-3])
        #if beam<=bins_centers.max():
        masses.append(mass)
    masses = np.array(sorted(set(masses)))
    
    print(f'total number of simulations: {len(masses)*len(beamEnergies)}')
    
    
    crossSection = np.zeros((len(masses), len(beamEnergies)))
    
    for i,mass in enumerate(masses):
        for j,beamEnergy in enumerate(beamEnergies):
    
            # skip if mass ALP is higher than the incoming photon energy
            if mass>=beamEnergy:
                continue
                
            filename = f'{madgraphDir}/runTags/runTag_mass{mass:08.6f}GeV_beam{beamEnergy:05.1f}GeV_prod.txt'
            
            # open file if existent
            try:
                with open(filename) as fd:
                    weightLine = fd.readlines()[-4]
            except:
                print(f'file "{filename}" not existent')
    
            # some files may not have a cross section, check
            try:
                csUnit = weightLine.split()[3][1:-1]
                cs = float(weightLine.split()[-1])
                if csUnit!='pb':
                    print(f'cross-section unit in "{filename}" is "{csUnit}"')
                crossSection[i,j] = cs
            except:
                print(f'file "{filename}" has no cross section')
    
    fig, ax = plt.subplots()
    
    cax = ax.pcolormesh(crossSection)
    
    fig.colorbar(cax, label='cross-section [pb]')
    
    ax.contour(crossSection==0, levels=[0.1], colors='red', linestyles='-')
    
    xtick_idx = np.argwhere((beamEnergies==1) | (beamEnergies==5) | (beamEnergies==10) | (beamEnergies==15)).flatten()
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels(beamEnergies[xtick_idx].astype(int))
    ax.set_xlabel('Compton photon energy [GeV]')
    
    ytick_idx = np.argwhere((masses==0.01) | (masses==0.1) | (masses==1.0)).flatten()
    ax.set_yticks(ytick_idx)
    ax.set_yticklabels(masses[ytick_idx])
    ax.set_ylabel('ALP mass [GeV]')
     
    plt.show()

else:
    
    masses = np.load(setupFile)['masses']
    beamEnergies = np.load(setupFile)['beamEnergies']
    crossSection = np.load(setupFile)['crossSection']


#####################################
#  SAVE SETUP
#####################################

if setup:
    
    np.savez(setupFile, 
             L_d=L_d, L_v=L_v, R_max=R_max, E_min=E_min, L_eff_pb=L_eff_pb, 
             N_sim=N_sim, n_norm=n_norm, n_sim=n_sim, 
             G_A=G_A, masses=masses, beamEnergies=beamEnergies, crossSection=crossSection)



#####################################
#  ANALYZE
#####################################

if not setup:
    
    for m_a in masses:
    
        print(f'{m_a}')
        
        fname = path + f'npodSignal_signalEvents_ma{m_a:08.6f}GeV.npz'
    
        if os.path.exists(fname):
            continue
        
        # select the simulated range
        nPhotons = n_sim[beamEnergies>m_a]
        ePhotons = beamEnergies[beamEnergies>m_a]
        
        # process the data on multiple threads
        with Pool(processes=12) as pool:
            Ndet = pool.map_async(calcALPphotons, G_A)
            track_job(Ndet, update_interval=1)
        Ndet = np.asarray(Ndet.get()).T
        
        np.savez(fname, Ndet=Ndet)
