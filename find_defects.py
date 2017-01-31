
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

def calculate_defect_strength(directors):
    """ calculate the defect strength"""
    
    dmax = (directors[-1] - directors[0])/2/np.pi
 
    return dmax

##############################################################################
    
def calculate_nematic_directors(qxx, qxy, qyy, nseg):
    """ calcualte the nematic directors inside the orthants"""
    
    ### calculate the nematic director
    # calculate the eigenvector corresponding to the largest eigenvalue
    # of the order parameter matrix
    
    directors = np.zeros((nseg), dtype=np.float64)
    for j in range(nseg):
        q = np.array([[qxx[j], qxy[j]], [qxy[j], qyy[j]]])
        w, v = LA.eig(q)
        idxmax = np.argmax(w)
        directors[j] = math.atan2(v[1,idxmax], v[0,idxmax])  

    ### correct nematic directors by assuming no jumps > |pi/2|, as we have line symmetry
    
    corrected_directors = np.copy(directors)
    for i in range(1, nseg):
        dif = directors[i] - directors[i-1]
        if dif > np.pi/2:
            dif -= np.pi
        if dif < -np.pi/2:
            dif += np.pi
        corrected_directors[i] = corrected_directors[i-1] + dif
        
    return directors, corrected_directors

##############################################################################

def calculate_defect_orientation(corrected_directors):
    """ calculate the angle of the defects"""
    
    ### All directors are with respective to the first (this will not work, most likely)
    #corrected_directors -= np.min(corrected_directors)
   
    ### Fit a straight line
    def line(x, A, B):
        return A*x + B
    xaxis = np.linspace(0, 2*np.pi, num=len(corrected_directors))
    A, B = curve_fit(line, xaxis, corrected_directors)[0]

    ### Find intersection point with x + np.pi in radians
    tail=-1
    n = -5

    ### 0 < comet tail < 2*pi
    while (tail<0.0 or tail>2.0*np.pi):
    	tail = (n*np.pi-B)/(A-1)
	n=n+1
	if n==6:
	    break

    ### uncomment this if to ouput vector
    #return np.cos(tail), np.sin(tail)	#x, y

    ### outputs the angle in radians
    return tail  
    
##############################################################################
    
def calculate_order_param_matrix(xd, yd, nseg, x, y, phi_nematic, sim, cell_list):
    """ calculate and average the order parameter matrix per orthant"""
    
    ### determine the cell of the current defect point
    
    segx = int(xd/sim.lx*cell_list.nsegx)
    segy = int(yd/sim.ly*cell_list.nsegy)
    cell = segx*cell_list.nsegy + segy
    
    ### allocate arrays
    # q: the order parameter matrix to be averaged per orthant
    # xcm, ycm: center of mass of orthants
    
    qxx = np.zeros((nseg), dtype=np.float64)
    qxy = np.zeros((nseg), dtype=np.float64)
    qyy = np.zeros((nseg), dtype=np.float64)

    xcm = np.zeros((nseg), dtype=np.float64)
    ycm = np.zeros((nseg), dtype=np.float64)

    counter = np.zeros((nseg), dtype=int)

    ### loop over the linked list of the current cell
    
    for a in range(-1,2):
        i2 = (segx+a+cell_list.nsegx) % cell_list.nsegx

        ### loop over the linked list of the neighboring cells

        for b in range(-1,2):
            j2 = (segy+b+cell_list.nsegy) % cell_list.nsegy

            ### get head value of the neighboring cell
            
            cell = i2*cell_list.nsegy + j2
            val = cell_list.head[cell]

            ### loop over the beads in the selected cell

            while val != -1:
                
                xi = x[val]
                yi = y[val]

                ### check the distance from the bead to the defect point
                
                dx = misc_tools.nearest_neighbor(xi, xd, sim.lx)
                dy = misc_tools.nearest_neighbor(yi, yd, sim.ly)
                dsq = dx**2 + dy**2
                
                if dsq <= cell_list.rcut2:
                    
                    ### compute the angle of the vector from the bead to the defect point
                    
                    theta = math.atan2(dy, dx)
                    if theta < 0:
                        theta += 2*np.pi
                        
                    ### determine which octant the bead-point angle falls into 
                    
                    segi = int(theta/2/np.pi*nseg)

                    ### calculate the bond orientations
                    
                    bx = np.cos(phi_nematic[val])
                    by = np.sin(phi_nematic[val])
                    
                    ### average the order parameter matrix
                    
                    qxx[segi] += 2*bx**2 - 1
                    qxy[segi] += 2*bx*by
                    qyy[segi] += 2*by**2 - 1

                    ### calculate the center of mass of the orthants
                    
                    xcm[segi] += xi
                    ycm[segi] += yi
                    
                    ### count the number of bonds in the orthants
                    
                    counter[segi] += 1

                ### get the next item from the linked list in the cell
                
                val = cell_list.llist[val]

    ### take the average of the order parameter matrix and center of masses per orthant
    
    qxx /= counter
    qxy /= counter
    qyy /= counter
    
    xcm /= counter
    ycm /= counter
    
    return qxx, qxy, qyy, xcm, ycm
    
##############################################################################
        
def compute_single_defect(xd, yd, x, y, phi, phi_nematic, cid, sim, cell_list, \
                          possible_defect_pts, pt_colors, \
                          rn, nn, total_rec_num, dcut, fig_cnt):
    """ compute the defect strength of a single point given with (xd, yd)"""
    
    ### allocate array to divide the full circle into orthants
    
    nseg = 10
    DEFECT_BOOL = False

    ### calculate and average the order parameter matrix per orthant
   
    qxx, qxy, qyy, xcm, ycm = calculate_order_param_matrix(xd, yd, nseg, x, y, phi_nematic, sim, cell_list)
    
    ### calculate the nematic directors per orthant
    
    directors, corrected_directors = calculate_nematic_directors(qxx, qxy, qyy, nseg)

    ### determine the defect strength
    
    dmax = calculate_defect_strength(corrected_directors)
    cc = 'colors'

    ### check for -1/2 defects
    
    if (dmax > -dcut-0.5 and dmax < dcut-0.5):
        DEFECT_BOOL = True
        cc = 'r'
        possible_defect_pts.append([xd, yd, dmax])
        pt_colors.append(cc)
        
    ### check +1/2 defects
    
    elif (dmax > 0.5-dcut and dmax < dcut+0.5):
        DEFECT_BOOL = True
        cc = 'g'
        possible_defect_pts.append([xd, yd, dmax])
        pt_colors.append(cc)

    ### if the point is not a defect
    
    else:
        return
    
    ### search around friends of friends total_rec_num times max 
    
    if DEFECT_BOOL:
        #print "Commencing recursion loop"
        friend_list = recursion.find_friends(dmax, xd, yd, cell_list.rcut, rn, nn)
        recursion.recursion(possible_defect_pts, friend_list, dmax, cc, rn, nn, x, y, \
						phi_nematic, nseg, \
						cid, sim, cell_list, \
						possible_defect_pts, \
						pt_colors, fig_cnt, total_rec_num, 0)
    


    ### plot the defects
        
#    if len(pt_colors) % 50 == 0:
#        savepath = '/usr/users/iff_th2/duman/Desktop/figcontainer/fig_' + str(fig_cnt) + '.png'
#        plot_defects.plot_defect(x, y, phi, phi_nematic, cid, xd, yd, directors, corrected_directors, \
#                        dmax, sim, cell_list, possible_defect_pts, pt_colors, xcm, ycm, savepath)
    
    return  

##############################################################################
        
def recompute_defects(xcmp, ycmp, beads, sim, rcut, dcut, step, spath):
    """ recompute the defect strengths of the ultimate defect points"""
    
    ### allocate array to divide the full circle into orthants
    
    nseg = 10
    
    ### generate a cell based linked list to browse beads based on position
    
    x = beads.x[step, 0, :]
    y = beads.x[step, 1, :]
    cell_list = linked_list(x, y, sim, rcut)
    
    ### calculate the bead orientations
    
    phi = misc_tools.compute_orientation(x, y, sim.lx, sim.ly, sim.nbpf)
    
    ### generate phi_nematic array by turning the orientations headless
    
    phi_nematic = np.zeros((len(x)))
    for i in range(len(x)):
        pi = phi[i]
        if pi < 0:
            pi += np.pi
        phi_nematic[i] = pi    

    ndefects = len(xcmp)
    fig_cnt = 0
    defect_pts = []
    pt_colors = []
    for j in range(ndefects):
        
        xd = xcmp[j]
        yd = ycmp[j]

        ### calculate and average the order parameter matrix per orthant
       
        qxx, qxy, qyy, xcm, ycm = calculate_order_param_matrix(xd, yd, nseg, x, y, \
                                                               phi_nematic, sim, cell_list)
        
        ### calculate the nematic directors per orthant
        
        directors, corrected_directors = calculate_nematic_directors(qxx, qxy, qyy, nseg)
    
        ### determine the defect strength
        
        dmax = calculate_defect_strength(corrected_directors)
        cc = 'colors'
        
        ### check for -1/2 defects
        
        if (dmax > -dcut-0.5 and dmax < dcut-0.5):
            cc = 'r'
            defect_pts.append([xd, yd, dmax])
            pt_colors.append(cc)
            
        ### check +1/2 defects
        
        elif (dmax > 0.5-dcut and dmax < dcut+0.5):
            cc = 'g'
            defect_pts.append([xd, yd, dmax])
            pt_colors.append(cc)
        
        ### plot the defects
        
        if cc == 'r' or cc == 'g':
            fig_cnt += 1
            savepath = spath + 'fig_' + str(step) + '_'+ str(fig_cnt) + '.png'
            plot_defects.plot_defect(x, y, phi, phi_nematic, beads.cid, xd, yd, \
                                     directors, corrected_directors, \
                                     dmax, sim, cell_list, defect_pts, pt_colors, \
                                     xcm, ycm, savepath)

    return defect_pts
            
##############################################################################
        
def find_defects(beads, sim, step, rcut, npoints, rn, dn, nn, total_rec_num, dcut):
    """ find the defect points"""
    
    ### generate a cell based linked list to browse beads based on position
    
    cell_list = linked_list(beads.x[step, 0, :], beads.x[step, 1, :], sim, rcut)
    
    ### calculate the bead orientations
    
    phi = misc_tools.compute_orientation(beads.x[step, 0, :], beads.x[step, 1, :], \
                                             sim.lx, sim.ly, sim.nbpf)

    ### generate phi_nematic array by turning the orientations headless
    
    phi_nematic = np.zeros((npoints))
    for i in range(npoints):
        pi = phi[i]
        if pi < 0:
            pi += np.pi
        phi_nematic[i] = pi
    
    ### choose random trial points and determine their defect strength
    ### do mesh refinement for points around existing defects
    
    possible_defect_pts = []
    pt_colors = []
    points = np.random.uniform(0., sim.lx, (npoints, 2))
    fig_cnt = 0
    for point in points:
        #print 'Search for a new point commences with the following coordinates: ', point[0], ' ', point[1]
        fig_cnt += 1
        compute_single_defect(point[0], point[1], beads.x[step, 0, :], beads.x[step, 1, :], \
                                  phi, phi_nematic, beads.cid, sim, cell_list, possible_defect_pts, \
                                      pt_colors, rn, nn, total_rec_num, dcut, fig_cnt)
        
    return possible_defect_pts
        
##############################################################################

def save_data(points, sfl):
    """ save the data on possible defect points"""

    #savefolder = '/usr/users/iff_th2/duman/Defects/Output/'
    
    fl = open(sfl, 'w')
    npoints = len(points)
    for j in range(npoints):
        fl.write(str(points[j][0]) + '\t' + str(points[j][1]) + '\t' + str(points[j][2]) + '\n')
    fl.close()

    return

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

    ### find defects
 
    rcut = 15.              # defect search radius
    npoints = 20000         # number of points that is going to be checked
    rn = 0.65*rcut          # inner radius of the shell where neighbor search is going to performed
    dn = 5.                 # search radius/depth of the shell (THIS IS NOT USED ATM!)
    nn = 5                  # number of neighbor points to pop up
    dcut = 0.1              # defect strength cut
    total_rec_num = 3       # total number of recursions to find friends of friends and so on    
    dcrit = 15.             # cluster threshold criteria

    for step in range(args.inittime, args.fintime):
        
        print 'step / last_step: ', step, args.fintime
    
        ### find the possible defect points
        
        possible_defect_pts = find_defects(beads, sim, step, rcut, npoints, \
                                           rn, dn, nn, total_rec_num, dcut)
        
        ### save the possible defect points
        
        sfilepath = args.savefolder + 'possible_defect_pts_' + str(step) + '.txt'
        save_data(possible_defect_pts, sfilepath)
        
        ### load the possible defect points
        
#        possible_defect_pts = load_data(sfilepath)
        
        ### cluster the possible defect points and plot the cluster
        
#        xcm, ycm = examine_clusters.cluster_analysis(possible_defect_pts, dcrit, sim, step, \
#                                                     beads.x[step, 0, :], beads.x[step, 1, :], beads.cid)
    
        ### for each of the defect points found by clustering recalculate defect strength and plot each point
    
#        defect_pts = recompute_defects(xcm, ycm, beads, sim, rcut, dcut, step, args.figfolder)    
    
        ### save the ultimate defect points
        
#        sfilepath = args.savefolder + 'defect_pts_' + str(step) + '.txt'
#        save_data(defect_pts, sfilepath)
    
    return
    
##############################################################################

if __name__ == '__main__':
    main()    
    
##############################################################################
