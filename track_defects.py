
""" Track defect points in time"""

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
import recursion

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

def load_data(loadfile, ):
    """ load the analysis data"""
    
    data = np.transpose(np.loadtxt(loadfile, dtype=np.float64))    

    return np.array([data[0][data[2]>0], data[1][data[2]>0]]), \
        np.array([data[0][data[2]<0], data[1][data[2]<0]]) 


##############################################################################

def gen_prev_defect_data(points, frame):
    """ generate the defect data of the investigated time frame
    the idea is to get the points that existed in the last time frame"""
    
    ### allocate the lists
    
    frame_points = []
    
    ### get the point ids
    
    pids = points.keys()
    
    ### run over the unique points and find the ones corresponding to the investigated frame
    
    for pid in pids:
        t = points[pid][0][0]
        if t == frame:
            x = points[pid][0][1]
            y = points[pid][0][2]
            frame_points.append([pid, x, y])
        
    return frame_points

##############################################################################

def determine_labels(pos_t1, labels_t0, points, frame, unique_pt_cnt, lx, ly):
    """ determine the new labels by a comparison with the old labels of the previous frame
    Hungarian algorithm with a linear cost function is used"""
    
    ### get the number of points
    
    npoints_t0 = len(labels_t0)
    npoints_t1 = len(pos_t1)
    
    ### build the correspondence matrix between the points by calculating displacements per edge
        
    threshold = 50.
    large_distance = lx*4.
    
    C = np.zeros((npoints_t0, npoints_t1), dtype=np.float32) - 1
    for i in range(npoints_t0):
        for j in range(npoints_t1):
            x0 = labels_t0[i][1]            
            x1 = pos_t1[j][0]
            y0 = labels_t0[i][2]            
            y1 = pos_t1[j][1]
            dx = misc_tools.nearest_neighbor(x1, x0, lx)
            dy = misc_tools.nearest_neighbor(y1, y0, ly)
            dr = np.sqrt(dx**2 + dy**2)
            if dr > threshold:
                dr = large_distance
            C[i][j] = dr
    
    ### suggest an assignment matrix
    # for all the profitable costs, label the minimum value column in C with the row label.
    # if there is no profitable cost, then create a new label/unique id for the point.

    A = np.zeros((npoints_t0, npoints_t1), dtype=np.float32) - 1
    for i in range(npoints_t0):
        min_distance = min(C[i])
        j = np.argmin(C[i])
        
        ### the label is suggested to be kept if the cost is within a profitable range
        
        if min_distance < large_distance:
            A[i][j] = min_distance

    ### handle conflicts in the assignment matrix
    # make sure that multiple points do not get assigned with the same label
    
    min_distance_per_column = np.zeros((npoints_t1), np.float32) - 1
    columns = np.transpose(A)
    for j in range(npoints_t1):
        c = columns[j]
        c = c[c>=0]
        if len(c) > 0:
            min_distance_per_column[j] = min(c)
            

    ### update the assignment matrix after the conflict handling
    
    for i in range(npoints_t0):
        for j in range(npoints_t1):
            if A[i][j] != min_distance_per_column[j]:
                A[i][j] = -1.
    
    ### update the labels 

    columns_filled = np.zeros((npoints_t1), dtype=np.int32)
    for i in range(npoints_t0):
        for j in range(npoints_t1):
            
            if A[i][j] != -1:
                points[i].append( [frame, pos_t1[j][0], pos_t1[j][1]] )
                columns_filled[j] = 1
                break
            
    ### create new labels for unmatched points (unfilled columns)
    
    for j in range(npoints_t1):
        if columns_filled[j] == 0:
            unique_pt_cnt += 1
            print 'A new point is created, in total, there are', unique_pt_cnt, 'unique points.'
            points[unique_pt_cnt] = []
            points[unique_pt_cnt].append( [frame, pos_t1[j][0], pos_t1[j][1]] )

    print points, '\n\n'

    return unique_pt_cnt
        
##############################################################################

def track_defects(ti, tf, folder, sim):
    """ track the defects in time"""
            
    ### load the first timeframe to build the initial labels
    # keep a dictionary of points,
    # keys = unique point ids.
    # values = list containing lists of time + pos information of points as:
    # [[timeframe, x, y], [...], ...]   per each timeframe the point has lived
    
    unique_pt_cnt_pos = -1      # to count the unique points
    #unique_pt_cnt_neg = -1
    
    points_pos = {}             # the keys are unique ids, values are lists containing [[time, x, y], ...]
    #points_neg = {}

    filepath = folder + 'defect_pts_' + str(ti) + '.txt'
    data = load_data(filepath)    
    defects_pos_t0 = np.transpose(data[0])
    #defects_neg_t0 = np.transpose(data[1])

    for j in range(len(defects_pos_t0)):
        unique_pt_cnt_pos += 1
        points_pos[unique_pt_cnt_pos] = []
        points_pos[unique_pt_cnt_pos].append([ti, defects_pos_t0[j][0], defects_pos_t0[j][1]])
        
    ### run over the frames 
    
    for frame in range(ti+1, tf):
        
        print "step / total step ", frame, tf
        
        ### generate the defect data of the previous frame
        # this will result in a list of the following form:
        # [[id, x, y], [id, x, y], ...]
        
        labels_t0 = gen_prev_defect_data(points_pos, frame-1)
        
        ### load the defect data of the current frame
        
        filepath = folder + 'defect_pts_' + str(frame) + '.txt'
        data = load_data(filepath)
        defects_pos_t1 = np.transpose(data[0])
        #defects_neg_t1 = np.transpose(data[1])   
        
        ### determine the new labels by comparing with the labels of the previous time frame
        
        unique_pt_cnt_pos = determine_labels(defects_pos_t1, labels_t0, points_pos, frame, unique_pt_cnt_pos, sim.lx, sim.ly)    
    
    
    return
        
##############################################################################

def main():
  
    ### get the data folder
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-fl", "--folder", nargs="?", \
                        const='/usr/users/iff_th2/duman/Defects/Output/', \
                        help="Folder containing data")
    parser.add_argument("-sfl", "--savefolder", nargs="?", \
                        const='/usr/users/iff_th2/duman/Defects/Output/', \
                        help="Folder to save the data inside")
    parser.add_argument("-figfl", "--figfolder", nargs="?", \
                        const='/usr/users/iff_th2/duman/Defects/Output/Figures/', \
                        help="Folder to save the figures inside")  
    parser.add_argument("-ti", "--inittime", type=int, help="Initial time step")
    parser.add_argument("-tf", "--fintime", type=int, help="Final timestep")            
    parser.add_argument("-s","--save_eps", action="store_true", help="Decide whether to save in eps or not")            
    args = parser.parse_args()
    
    ### read the data and general information from the folder
    
    beads, sim = read_data(args.folder)
    
    ### track defects
    
    track_defects(args.inittime, args.fintime, args.folder, sim)
            
    return
    
##############################################################################

if __name__ == '__main__':
    main()

##############################################################################    
    
    
    
    
    