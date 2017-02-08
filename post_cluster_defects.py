
""" Find defects in an ensemble of filaments at high densities"""

##############################################################################

import argparse
import numpy as np
import os
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Circle
from numpy import linalg as LA
import misc_tools
import plot_defects
import examine_clusters
import recursion
from scipy.optimize import curve_fit

##############################################################################        
   
class Simulation:
    """ data structure for storing general simulation information"""

    def __init__(self, lx, ly, dt, nsteps, nbeads, nsamp, nbpf, density, kappa, \
               fp, kT, bl, sigma, gamma):
        
        self.lx = float(lx)
        self.ly = float(ly)
        self.dt = dt
        self.nsteps = int(nsteps)
        self.nbeads = int(nbeads)
        self.nsamp = 10000
        self.nbpf = 51
        self.density = 0.8
        self.fp = fp
        self.kappa = kappa
        self.bl = 0.5
        self.sigma = sigma
        self.kT = kT
        self.gamma_n = gamma
        
        ### normalize certain variables

        self.nfils = self.nbeads/self.nbpf        
        #self.lx /= self.bl
        #self.ly /= self.bl   
        self.dt *= self.nsamp
        
        ### define more simulation parameters
        
        self.length = self.nbpf*self.bl
        #self.N_avg = np.average(self.nbpc)
        #self.r_avg = self.bl*self.N_avg/2/np.pi
        #self.tau_D = self.r_avg**2 * self.gamma_n * self.N_avg / self.kT
        #self.tau_A = 2 * self.r_avg * self.gamma_n / self.fp
        
        return
        
##############################################################################

class Beads:
    """ data structure for storing particle/bead information"""
    
    def __init__(self, x, sim):
        
        ### assign bead positions
        
        self.x = x
        
        ### assign mol indices to beads
        
        self.cid = np.zeros((sim.nbeads), dtype=np.int32)-1
        k = 0
        for j in range(sim.nfils):
            for n in range(sim.nbpf):
                self.cid[k] = j
                k += 1
        
        return

##############################################################################
        
class Subplots:
    """ plot structure"""
    
    totcnt = -1             # Total number of subplots 
    
    def __init__(self, f, l, s, b, t):
        self.fig = f        # Figure axes handle
        self.length = l     # Length of the subplot box 
        self.sep = s        # Separation distance between subplots 
        self.beg = b        # Beginning (offset) in the figure box
        self.tot = t        # Total number of subplots in the x direction
        
        return
        
    def addSubplot(self):
        """ add a subplot in the grid structure"""
        
        ### increase the number of subplots in the figure
        
        self.totcnt += 1
        
        ### get indices of the subplot in the figure
        
        self.nx = self.totcnt%(self.tot)
        self.ny = self.totcnt/(self.tot)
        
        self.xbeg = self.beg + self.nx*self.length + self.nx*self.sep
        self.ybeg = self.beg + self.ny*self.length + self.ny*self.sep
        
        return self.fig.add_axes([self.xbeg,self.ybeg,self.length,self.length])
        
##############################################################################
        
class linked_list:
    """ linked list data structure to browse beads based on their position"""
    
    def __init__(self, x, y, sim, rcut):
        
        ### define starting values
        
        self.rcut = rcut
        self.rcut2 = rcut**2      
        self.nsegx = int(sim.lx/self.rcut)
        self.nsegy = int(sim.ly/self.rcut)
        self.llist = np.zeros((sim.nbeads), dtype = int) - 1
        self.head = np.zeros((self.nsegx*self.nsegy), dtype = int) - 1

        ### put all the beads inside the list
        
        for i in range(sim.nbeads):
            segx = int(x[i]/sim.lx*self.nsegx)
            segy = int(y[i]/sim.ly*self.nsegy)
            cell = segx*self.nsegy + segy
            self.llist[i] = self.head[cell]
            self.head[cell] = i

        return

##############################################################################

def read_data(folder):
    """ read simulation data through hdf5 file"""
    
    ### access the file
    
    fpath = folder + '/out.h5'
    assert os.path.exists(fpath), "out.h5 does NOT exist for " + fpath
    fl = h5py.File(fpath, 'r')
    
    ### read in the positions of beads
    
    x = np.asarray(fl['/positions/x'], dtype=np.float32)

    ### read in the box info

    lx = fl['/info/box/x'][...]
    ly = fl['/info/box/y'][...]

    ### read in the general simulation info
    
    dt = fl['/info/dt'][...]
    nsteps = fl['/info/nsteps'][...]
    nbeads = fl['/info/nbeads'][...]
    nsamp = fl['/info/nsamp'][...]

    ### read in the mol information
    
    nbpf = fl['/param/nbpf'][...]

    ### read in the simulation parameters
    
    density = fl['/param/density'][...]
    kappa = fl['/param/kappa'][...]
    fp = fl['/param/fp'][...]
    gamma = fl['/param/gamma'][...]
    bl = fl['/param/bl'][...]
    sigma = fl['/param/sigma'][...]
    kT = fl['/param/kT'][...]

    ### close the file

    fl.close()
    
    sim = Simulation(lx, ly, dt, nsteps, nbeads, nsamp, nbpf, density, kappa, \
               fp, kT, bl, sigma, gamma)
    beads = Beads(x, sim)
    
    return beads, sim            

##############################################################################

def load_data(loadfile):
    """ load the data on possible defect points"""
    
    data = np.transpose(np.loadtxt(loadfile, dtype=np.float64))

    return data
                   
##############################################################################

def main():
    
    ### get the data folder
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-fl", "--folder", nargs="?", \
                        const='/usr/users/iff_th2/duman/Defects/Output/', \
                        help="Folder containing data")
    parser.add_argument("-sfl", "--savefolder", nargs="?", \
                        const='/usr/users/iff_th2/duman/Defects/Output/cpp/', \
                        help="Folder to save the data inside")    
    parser.add_argument("-figfl", "--figfolder", nargs="?", \
                        const='/usr/users/iff_th2/duman/Defects/Output/Figures/', \
                        help="Folder to save the figures inside")  
    parser.add_argument("-ti", "--inittime", nargs="?", const=100, type=int, help="Initial time step")
    parser.add_argument("-tf", "--fintime", nargs="?", const=153, type=int, help="Final timestep")            
    parser.add_argument("-s","--save_eps", action="store_true", help="Decide whether to save in eps or not")            
    args = parser.parse_args()
    
    ### read the data and general information from the folder
    
    beads, sim = read_data(args.folder)
    
    rcut = 15.              # size of the interrogation circle
    dcut = 0.1              # defect strength cut
    dcrit = 10.             # cluster threshold criteria
    

    for step in range(args.inittime, args.fintime):
        
        print 'step / last_step: ', step, args.fintime
        
        ### load the defect points
        
        sfilepath = args.savefolder + 'defect_pts_' + str(step) + '.txt'
        defect_pts = load_data(sfilepath)
        defect_pts = np.array(defect_pts)
        xd = defect_pts[0]
        yd = defect_pts[1]
        dmax = defect_pts[2]
        tail = defect_pts[3]
        #xd, yd, dmax, tail = defect_pts.T
        cell_list = linked_list(beads.x[step, 0, :], beads.x[step, 1, :], sim, rcut)
        
        ### plot the defects
        
        sfilepath = args.figfolder + 'defect_pts_' + str(step) + '.png'
        plot_defects.pretty_plot(beads.x[step, 0, :], beads.x[step, 1, :], beads.cid, \
                                 xd, yd, tail, dmax, cell_list, sim, sfilepath)
    
    return
    
##############################################################################

if __name__ == '__main__':
    main()    
    
##############################################################################
