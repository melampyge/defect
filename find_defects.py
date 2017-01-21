
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
        
def determine_neigh(xd, yd, l, sim):
    """ determine the neighborhood upon which new points are sampled around the point (xd, yd)"""   

    xdown = xd - l
    if (xdown < 0.):
        xdown = 0.5
    
    xup = xd + l
    if (xup > sim.lx):
        xup = sim.lx-0.5
        
    ydown = yd - l
    if (ydown < 0.):
        ydown = 0.5
    
    yup = yd + l
    if (yup > sim.ly):
        yup = sim.ly-0.5  
        
    return xdown, xup, ydown, yup

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

    ### take the average of the nematic directors and center of masses per orthant
    
    qxx /= counter
    qxy /= counter
    qyy /= counter
    
    xcm /= counter
    ycm /= counter
    
    return qxx, qxy, qyy, xcm, ycm
    
##############################################################################
        
def compute_single_defect(xd, yd, x, y, phi, cid, sim, cell_list, \
                          possible_defect_pts, pt_colors, rn, nn, fig_cnt):
    """ compute the defect strength of a single point given with (xd, yd)"""
    
    ### allocate array to divide the full circle into orthants
    
    nseg = 10

    ### generate phi_nematic array by turning the orientations headless
    
    phi_nematic = np.zeros((len(x)))
    for i in range(len(x)):
        pi = phi[i]
        if pi < 0:
            pi += np.pi
        phi_nematic[i] = pi

    ### calculate and average the order parameter matrix per orthant
   
    qxx, qxy, qyy, xcm, ycm = calculate_order_param_matrix(xd, yd, nseg, x, y, phi_nematic, sim, cell_list)
    
    ### calculate the nematic directors per orthant
    
    directors, corrected_directors = calculate_nematic_directors(qxx, qxy, qyy, nseg)

    ### determine the defect strength
    
    dmax = calculate_defect_strength(corrected_directors)
    print dmax
    
#    if abs(dmax) < 0.3:
#        return
#    else:
#        
#        ### add the point to the possible defect points list
#        
#        print 'Point added is: ', xd, ' ', yd
#        possible_defect_pts.append([xd, yd])   
#        
#        ### pop up new neighbors in the vicinity of the point until all points are checked
#        
#        xdown, xup, ydown, yup = determine_neigh(xd, yd, r_new_points, sim)            
#        xnew = np.random.uniform(xdown, xup, n_new_points)
#        ynew = np.random.uniform(ydown, yup, n_new_points)
#        new_points = np.transpose(np.vstack((xnew, ynew)))
#        for point in new_points:
#            fig_cnt += 1
#            compute_single_defect(possible_defect_pts, point[0], point[1], x, y, phi, cid, sim, cell_list, \
#                                      r_new_points, n_new_points, fig_cnt)
    print 'Point added is: ', xd, ' ', yd
    cc = 'colors'

    # Determine whether or not its a defect: 
    DEFECT_BOOL  = FALSE;
    BREAK_ALL = FALSE; 

    # THESE ARE INPUT PARAMETERS!!
    defect_strength_cut = 0.1;		# cutoff around +/-0.5  
    defect_strength_cut_max = 0.01;	# maximum cutoff around the defects
    nfriends = 10;
    
    if (dmax>-defect_strength_cut-0.5 and dmax<defect_strength_cut-0.5):
	DEFECT_BOOL = TRUE;
        cc = 'r'
    elif (dmax>0.5-defect_strength_cut and dmax<defect_strength_cut+0.5):
	DEFECT_BOOL = TRUE;
        cc = 'g'
    else:
	return


    # Hit gold on first try
    if (dmax>-defect_strength_cut_max-0.5 and dmax<defect_strength_cut_max-0.5):
        cc = 'r'
	### jump past friends search

    elif (dmax_1>0.5-defect_strength_cut_max and dmax_1<defect_strength_cut_max+0.5):
        cc = 'g'
	### jump past friends search

    ### search around friends of friends THRICE max
    if DEFECT_BOOL:
            friend_list = find_friends(dmax, xd, yd, rcut, inner_radius, nfriends)

            ### First iteration
	    for xd_friend,yd_friend in friend_list:
		if BREAK_ALL:
		    break

		dmax_1 = calculate_defect_recursion(xd_friend,yd_friend, x, y,phi,cid,sim,cell_list, possible_defect_pts, pt_colors, rn, nn, fig_cnt)

		# Hit gold with first friends
		if (dmax_1>-defect_strength_cut_max-0.5 and dmax_1<defect_strength_cut_max-0.5):
		    DEFECT_BOOL = TRUE;
		    cc = 'r'
		    xd,yd = xd_friend,yd_friend
		    break

		elif (dmax_1>0.5-defect_strength_cut_max and dmax_1<defect_strength_cut_max+0.5):
		    DEFECT_BOOL = TRUE;
		    cc = 'g'
		    xd,yd = xd_friend,yd_friend
		    break

		# Try again
		elif np.abs(dmax-0.5) < np.abs(dmax_1-0.5) or np.abs(dmax+0.5) < np.abs(dmax_1+0.5):
		    friend_list2 = find_friends(dmax_1, xd, yd, rcut, inner_radius, nfriends)


		### Second iteration	    
		for xd_friend2,yd_friend2 in friend_list2:
			
		    dmax_2 = calculate_defect_recursion(xd_friend2,yd_friend2, x, y,phi,cid,sim,cell_list, possible_defect_pts, pt_colors, rn, nn, fig_cnt)

		    # Hit gold with second friends
		    if (dmax>-defect_strength_cut_max-0.5 and dmax<defect_strength_cut_max-0.5):
			DEFECT_BOOL = TRUE;
			cc = 'r'
			xd,yd = xd_friend2,yd_friend2
			BREAK_ALL = TRUE
			break

		    elif (dmax>0.5-defect_strength_cut_max and dmax<defect_strength_cut_max+0.5):
			DEFECT_BOOL = TRUE;
			cc = 'g'
			xd,yd = xd_friend2,yd_friend2
			BREAK_ALL = TRUE
			break

		    # Try again
		    elif np.abs(dmax_1-0.5) < np.abs(dmax_2-0.5) or np.abs(dmax_1+0.5) < np.abs(dmax_2+0.5):
			friend_list3 = find_friends(dmax, xd, yd, rcut, inner_radius, nfriends)

			### Last iteration	    
			for xd_friend3,yd_friend3 in friend_list3:
			    dmax_3 = calculate_defect_recursion(xd_friend2,yd_friend2, x, y,phi,cid,sim,cell_list, possible_defect_pts, pt_colors, rn, nn, fig_cnt)
			    # Hit gold on last try
			    if (dmax>-defect_strength_cut_max-0.5 and dmax<defect_strength_cut_max-0.5):
				DEFECT_BOOL = TRUE;
				cc = 'r'
				xd,yd = xd_friend3,yd_friend3
				BREAK_ALL = TRUE
				break
			    elif (dmax>0.5-defect_strength_cut_max and dmax<defect_strength_cut_max+0.5):
				DEFECT_BOOL = TRUE;
				cc = 'g'
				xd,yd = xd_friend3,yd_friend3
				BREAK_ALL = TRUE
				break

    ### plot the defect
    
    if cc == 'w' or cc == 'r' or cc == 'g':
        possible_defect_pts.append([xd, yd])   
        pt_colors.append(cc)
        
    if len(pt_colors) % 100 == 0:
        savepath = '/usr/users/iff_th2/duman/Desktop/figcontainer/figure_' + str(fig_cnt) + '.png'
        plot_defects.plot_defect(x, y, phi, phi_nematic, cid, xd, yd, directors, corrected_directors, \
                        dmax, sim, cell_list, possible_defect_pts, pt_colors, xcm, ycm, savepath)
    
    return  
            
##############################################################################
        
def find_defects(beads, sim, step, rcut, npoints, rn, dn, nn):
    """ find the defect points"""
    
    ### generate a cell based linked list to browse beads based on position
    
    cell_list = linked_list(beads.x[step, 0, :], beads.x[step, 1, :], sim, rcut)
    
    ### calculate the bead orientations
    
    phi = misc_tools.compute_orientation(beads.x[step, 0, :], beads.x[step, 1, :], \
                                             sim.lx, sim.ly, sim.nbpf)
    
    ### choose random trial points and determine their defect strength
    ### do mesh refinement for points around existing defects
    
    possible_defect_pts = []
    pt_colors = []
    points = np.random.uniform(0., sim.lx, (npoints, 2))
    fig_cnt = 0
    for point in points:
        print 'Search for a new point commences with the following coordinates: ', point[0], ' ', point[1]
        fig_cnt += 1
        compute_single_defect(point[0], point[1], beads.x[step, 0, :], beads.x[step, 1, :], \
                                  phi, beads.cid, sim, cell_list, possible_defect_pts, \
                                      pt_colors, rn, nn, fig_cnt)
        
               
##############################################################################

def main():
    
    ### get the data folder
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-fl", "--folder", help="Folder containing data")
    parser.add_argument("-s","--save_eps", action="store_true", help="Decide whether to save in eps or not")            
    args = parser.parse_args()
    
    ### read the data and general informaion from the folder
    
    beads, sim = read_data(args.folder)
    
    ### find defects
    
    rcut = 15.              # defect search radius
    step = 300              # time step search is being performed
    npoints = 3000          # number of points that is going to be checked
    rn = 10.                # inner radius of the shell where neighbor search is going to performed
    dn = 5.                 # search radius/depth of the shell
    nn = 10                 # number of neighbor points to pop up
    find_defects(beads, sim, step, rcut, npoints, rn, dn, nn)

    return
    
##############################################################################

if __name__ == '__main__':
    main()    
    
##############################################################################
