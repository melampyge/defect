
""" Find defects in an ensemble of filaments at high densities"""

##############################################################################

import argparse
import numpy as np
import os
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import misc_tools
import math
from matplotlib.patches import Circle
from numpy import linalg as LA

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

#    idxmax = np.argmax(directors)
#    idxmin = np.argmin(directors)
#    omax = directors[idxmax]
#    omin = directors[idxmin]
#    dmax = omax - omin
#    dmax = 2*dmax/2/np.pi
#    dmax = round(dmax)
#    dmax /= 2
#    if idxmax < idxmin:
#        dmax *= -1

#    nseg = len(directors)
#    rotations_around_circle = np.zeros((nseg/2), dtype=np.float32) 
#    for j in range(nseg/2):
#        rotations_around_circle[j] = directors[j] - directors[nseg-j-1]
#    avg_rotation = np.mean(rotations_around_circle)
#    dmax = 2*avg_rotation/2/np.pi
#    dmax = round(dmax)
#    dmax /= 2
 
    return dmax
        
##############################################################################
        
def compute_single_defect(possible_defect_pts, xd, yd, x, y, phi, cid, sim, cell_list, \
                              r_new_points, n_new_points, fig_cnt):
    """ compute the defect strength of a single point given with (xd, yd)"""
    
    ### allocate array to divide the full circle into orthants
    
    nseg = 10
    directors = np.zeros((nseg))
    counter = np.zeros((nseg), dtype=int)
    
    ### generate a "considered" array to check whether bead was used in the analysis
    
    considered = np.zeros((len(x)))

    ### generate phi_nematic array by turning the orientations headless
    # consider that the orientation of beads are from -pi to pi
    # adding pi to the negative orientations to make them positive
    # as in nematic -pi and pi are the same!
    
    phi_nematic = np.zeros((len(x)))
    for i in range(len(x)):
        pi = phi[i]
        if pi < 0:
            pi += np.pi
        phi_nematic[i] = pi
   
    ### determine the cell of the current defect point
    
    segx = int(xd/sim.lx*cell_list.nsegx)
    segy = int(yd/sim.ly*cell_list.nsegy)
    cell = segx*cell_list.nsegy + segy
    
    ### loop over the linked list of the current cell
    
    qxx = np.zeros((nseg), dtype=np.float64)
    qxy = np.zeros((nseg), dtype=np.float64)
    qyy = np.zeros((nseg), dtype=np.float64)
    
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
                    
                    ### average the order parameter tensor
                    
                    bx = np.cos(phi_nematic[val])
                    by = np.sin(phi_nematic[val])
                    qxx[segi] += 2*bx**2 - 1
                    qxy[segi] += 2*bx*by
                    qyy[segi] += 2*by**2 - 1
#                    qxx[segi] += 1.5**bx**2 - 0.5
#                    qxy[segi] += 1.5*bx*by
#                    qyy[segi] += 1.5*by**2 - 0.5
                    
                    ### histogram the average nematic orientation in each octant 
                    # increase the counter and add information on beads
                    
                    #directors[segi] += phi_nematic[val]
                    counter[segi] += 1
                    considered[val] += 1

                ### get next item from the linked list in the cell
                
                val = cell_list.llist[val]

    ### take the average of the nematic directors
    
    qxx /= counter
    qxy /= counter
    qyy /= counter
    
    ### calculate the eigenvalues and eigenvectors of the order parameter tensor
    
    for j in range(nseg):
        q = np.array([[qxx[j], qxy[j]], [qxy[j], qyy[j]]])
        w, v = LA.eig(q)
        idxmax = np.argmax(w)
        directors[j] = math.atan2(v[1,idxmax], v[0,idxmax])
        eigenstuff = [v[0,idxmax], v[1,idxmax]]
        
        #print directors[j]
        
    
    ### correct nematic directors by assuming no jumps > pi/2
    # the reason we have to revert to this is because we have line symmetry now.
    
    corrected_directors = np.copy(directors)
    for i in range(1, nseg):
        dif = directors[i] - directors[i-1]
        if dif > np.pi/2:
            dif -= np.pi
        if dif < -np.pi/2:
            dif += np.pi
        corrected_directors[i] = corrected_directors[i-1] + dif

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
    
    ### plot the defect
    
    savepath = '/usr/users/iff_th2/duman/Desktop/figcontainer/figure_' + str(fig_cnt) + '.png'
    plot_defect(x, y, phi, phi_nematic, cid, xd, yd, directors, corrected_directors, \
                    dmax, sim, cell_list, possible_defect_pts, savepath, eigenstuff)
    
    return  

##############################################################################

def plot_defect(xp, yp, phi, phi_nematic, cid, xdp, ydp, directors, corrected_directors, \
                    dmax, sim, cell_list, possible_defect_pts_p, savepath, eigenstuff):
    """ plot the data"""
    
    ### normalize for plotting purposes
    
    lx = sim.lx/sim.bl
    ly = sim.ly/sim.bl
    x = xp/sim.bl
    y = yp/sim.bl
    xd = xdp/sim.bl
    yd = ydp/sim.bl
    possible_defect_pts = np.zeros((len(possible_defect_pts_p),2))
    for j in range(len(possible_defect_pts_p)):
        possible_defect_pts[j][0] = possible_defect_pts_p[j][0]/sim.bl
        possible_defect_pts[j][1] = possible_defect_pts_p[j][1]/sim.bl
    
    ### set plot properties

    ax_len = 0.9                          # Length of one subplot square box
    ax_b = 0.05                           # Beginning/offset of the subplot in the box
    ax_sep = 0.3                          # Separation length between two subplots
    total_subplots_in_x = 2               # Total number of subplots    
    fig = plt.figure()
    
    ### set more plot properties
    
    quant_steps = 2056
    norm_orient = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi) 
    norm_nematic = mpl.colors.Normalize(vmin=0, vmax=np.pi) 
    norm_filid = mpl.colors.Normalize(vmin=0, vmax=sim.nfils)
    num_ticks = 5
    
    ### plot the frame

    subp = Subplots(fig, ax_len, ax_sep, ax_b, total_subplots_in_x) 
    
    ### 1) the entire field with bead orientations
        
    ax0 = subp.addSubplot()
    #ax0.axis('equal')
    
    xline = np.linspace(0, 20, 10)
    yline = np.zeros((10))
    
    #ax0.quiver(xline, yline, np.cos(directors), np.sin(directors))
    ax0.quiver(xline, yline, eigenstuff[0], eigenstuff[1])
    
#    line0 = ax0.scatter(x, y, s=1, c=phi, cmap=plt.cm.get_cmap('hsv',quant_steps), 
#                edgecolors='None', alpha=0.7, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True)
#       
#    ### title
#    
##    ax0.set_title("$t/\\tau_{D}$ = " + "{0:.4f}".format(time/tau_D) + \
##        ", $t/\\tau_{A}$ = " + "{0:.4f}".format(time/tau_A), fontsize=20)
#    
#    ### labels
#        
#    ax0.set_xlabel("$x/r_{0}$", fontsize=40)
#    ax0.set_ylabel("$y/r_{0}$", fontsize=40)
#
#    ### limits
#
#    ax0.set_xlim((lx, lx))
#    ax0.set_ylim((ly, ly))
#    
#    ### ticks
#    
#    ax0.xaxis.set_ticks(np.linspace(0, lx, num_ticks, endpoint=True))
#    ax0.yaxis.set_ticks(np.linspace(0, ly, num_ticks, endpoint=True))
#    ax0.tick_params(axis='both', which='major', labelsize=20)
#
#    ax0.add_patch(Circle((xd, yd), cell_list.rcut, edgecolor='gray',facecolor='gray',alpha=0.8))
#
#    for j in range(len(possible_defect_pts)):
#        ax0.plot(possible_defect_pts[j][0], possible_defect_pts[j][1], \
#                     ls = '', marker='x', color='k', markersize=15, alpha=0.7)
        
    ### colorbar
            
    cax0 = plt.axes([subp.xbeg+ax_len+0.01, subp.ybeg+ax_len/3, ax_len/4.6, ax_len/4.6], projection='polar')
    xval = np.arange(-np.pi, np.pi, 0.01)
    yval = np.ones_like(xval)
    cax0.scatter(xval, yval, c=xval, s=300, cmap=plt.cm.get_cmap('hsv',quant_steps), norm=norm_orient, linewidths=0)
    cax0.set_xticks([])
    cax0.set_yticks([])
    cax0.set_title('$\\phi$',fontsize=40)
    cax0.set_rlim([-1,1])
    cax0.set_axis_off()
    
    
    ### 2) the entire field with filament ids
        
    ax1 = subp.addSubplot()
    ax1.axis('equal')
    
    line1 = ax1.scatter(x, y, s=1, c=cid, cmap=plt.cm.get_cmap('jet',quant_steps), 
                edgecolors='None', alpha=0.7, vmin=0, vmax=sim.nfils, norm=norm_filid, rasterized=True)
       
    ### title
    
#    ax0.set_title("$t/\\tau_{D}$ = " + "{0:.4f}".format(time/tau_D) + \
#        ", $t/\\tau_{A}$ = " + "{0:.4f}".format(time/tau_A), fontsize=20)
    
    ### labels
        
    ax1.set_xlabel("$x/r_{0}$", fontsize=40)
    #ax1.set_ylabel("$y/r_{0}$", fontsize=40)

    ### limits

    ax1.set_xlim((lx, lx))
    ax1.set_ylim((ly, ly))
    
    ### ticks
    
    ax1.xaxis.set_ticks(np.linspace(0, lx, num_ticks, endpoint=True))
    ax1.yaxis.set_ticks(np.linspace(0, ly, num_ticks, endpoint=True))
    ax1.tick_params(axis='both', which='major', labelsize=20)    
    
    ax1.add_patch(Circle((xd, yd), cell_list.rcut, edgecolor='gray',facecolor='gray',alpha=0.8))
    
    ### plot all checked possible points
    
    for j in range(len(possible_defect_pts)):
        ax1.plot(possible_defect_pts[j][0], possible_defect_pts[j][1], \
                     ls = '', marker='x', color='k', markersize=15, alpha=0.7)
    
    ### 3) zoom in to the defect
        
    ax2 = subp.addSubplot()
    ax2.axis('equal')
    
    size = 10
    
    ### plot all the beads along with their images 
    
    ax2.scatter(x, y, c=phi_nematic, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.7, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True)
    
    ax2.scatter(x-lx, y-ly, c=phi_nematic, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.7, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True) 
    
    ax2.scatter(x+lx, y-ly, c=phi_nematic, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.7, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True)
    
    ax2.scatter(x, y-ly, c=phi_nematic, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.7, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True)

    ax2.scatter(x-lx, y, c=phi_nematic, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.7, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True)

    ax2.scatter(x+lx, y, c=phi_nematic, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.7, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True)

    ax2.scatter(x-lx, y+ly, c=phi_nematic, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.7, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True)

    ax2.scatter(x, y+ly, c=phi_nematic, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.7, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True)

    ax2.scatter(x+lx, y+ly, c=phi_nematic, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.7, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True)
                
    ### plot the point
    
    ax2.plot(xd, yd, ls = '', marker='x', color='k', markersize=10, label=str(dmax))
    
    t = np.linspace(0, 2*np.pi, 1000)
    
    ### plot the circle around the point
    
    ax2.plot(cell_list.rcut*np.cos(t) + xd, cell_list.rcut*np.sin(t) + yd, color='k')
           
    ### title
    
#    ax0.set_title("$t/\\tau_{D}$ = " + "{0:.4f}".format(time/tau_D) + \
#        ", $t/\\tau_{A}$ = " + "{0:.4f}".format(time/tau_A), fontsize=20)
    
    ### labels
        
    ax2.set_xlabel("$x/r_{0}$", fontsize=40)
    ax2.set_ylabel("$y/r_{0}$", fontsize=40)

    ### limits

    xdownlim = xd - 1.5*cell_list.rcut
    xuplim = xd + 1.5*cell_list.rcut
    ydownlim = yd - 1.5*cell_list.rcut
    yuplim = yd + 1.5*cell_list.rcut    
    
    ax2.set_xlim([xdownlim, xuplim])
    ax2.set_ylim([ydownlim, yuplim])
    
    ### ticks
    
    ax2.xaxis.set_ticks(np.linspace(xdownlim, xuplim, num_ticks, endpoint=True))
    ax2.yaxis.set_ticks(np.linspace(ydownlim, yuplim, num_ticks, endpoint=True))
    ax2.tick_params(axis='both', which='major', labelsize=20)    

    
    ### 4) avg. director-point angles over the octants
        
    ax3 = subp.addSubplot()
    ax3.axis('equal')
    
    ax3.plot(directors/np.pi, color='r', label='defect = ' + str(dmax), marker='^')       
    ax3.plot(corrected_directors/np.pi, color='g', label='corrected_defect = ' + str(dmax), marker='^')
    
    ### title
    
#    ax0.set_title("$t/\\tau_{D}$ = " + "{0:.4f}".format(time/tau_D) + \
#        ", $t/\\tau_{A}$ = " + "{0:.4f}".format(time/tau_A), fontsize=20)
    
    ### labels
        
    ax3.set_xlabel("$j$", fontsize=40)
    ax3.set_ylabel("$\\theta(j)\\pi$", fontsize=40)
    
    ### legend
    
    ax3.legend()

    ### limits

#    xdownlim = xd - 1.5*cell_list.rcut
#    xuplim = xd + 1.5*cell_list.rcut
#    ydownlim = yd - 1.5*cell_list.rcut
#    yuplim = yd + 1.5*cell_list.rcut    
    
#    ax3.set_xlim([xdownlim, xuplim])
#    ax3.set_ylim([ydownlim, yuplim])
    
    ### ticks
    
#    ax3.xaxis.set_ticks(np.linspace(xdownlim, xuplim, num_ticks, endpoint=True))
#    ax3.yaxis.set_ticks(np.linspace(ydownlim, yuplim, num_ticks, endpoint=True))
    ax3.tick_params(axis='both', which='major', labelsize=20)   
    
    ### save
    
    plt.savefig(savepath, dpi=200, bbox_inches='tight', pad_inches=0.08)     
    fig.clf()
    
    return    
            
##############################################################################
        
def find_defects(beads, sim, step, rcut, npoints, r_new_points, n_new_points):
    """ find the defects"""
    
    ### generate a cell based linked list to browse beads based on position
    
    cell_list = linked_list(beads.x[step, 0, :], beads.x[step, 1, :], sim, rcut)
    
    ### calculate the bead orientations
    
    #phi = misc_tools.compute_orientation(beads.x[step, 0, :], beads.x[step, 1, :], sim.lx, sim.ly, sim.nbpf)
    phi = np.ones((sim.nbeads), dtype=np.float64)*np.pi/8.
    
    ### choose random trial points and determine their defect strength
    ### do mesh refinement for points around existing defects
    
    possible_defect_pts = []
    points = np.random.uniform(0.5, sim.lx, (npoints, 2))
    fig_cnt = 0
    for point in points:
        print 'Search for a new point commences with the following coordinates: ', point[0], ' ', point[1]
        fig_cnt += 1
        compute_single_defect(possible_defect_pts, point[0], point[1], \
                                  beads.x[step, 0, :], beads.x[step, 1, :], \
                                    phi, beads.cid, sim, cell_list, r_new_points, n_new_points, \
                                        fig_cnt)
    
               
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
    
    rcut = 50.
    step = 300
    npoints = 1500
    r_new_points = 15.
    n_new_points = 3
    find_defects(beads, sim, step, rcut, npoints, r_new_points, n_new_points)

    return
    
##############################################################################

if __name__ == '__main__':
    main()    
    
##############################################################################
