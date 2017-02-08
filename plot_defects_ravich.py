
""" Plot defects"""

##############################################################################

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import time

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

def pretty_plot(xp, yp, cid, xdp, ydp, tail, dmax, cell_list, sim, savepath):
    """ plot the data"""
    
    print "Plotting"
    
    ### normalize for plotting purposes
    lx = sim.lx/sim.bl
    ly = sim.ly/sim.bl
    x = xp/sim.bl
    y = yp/sim.bl
    xd = xdp/sim.bl
    yd = ydp/sim.bl
    rcut = cell_list.rcut/sim.bl
    
    ### set plot properties

    fig,ax1 = plt.subplots()
 
    ### set more plot properties
    
    quant_steps = 2056
    norm_orient = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi) 
    norm_filid = mpl.colors.Normalize(vmin=0, vmax=sim.nfils)
    num_ticks = 5
    
    ### plot the frame

    ### 2) the entire field with filament ids
        
    line1 = ax1.scatter(x, y, s=1, c=cid, cmap=plt.cm.get_cmap('jet',quant_steps), 
                edgecolors='None', alpha=0.4, vmin=0, vmax=sim.nfils, norm=norm_filid)

    pscale = 0.2
    boundary=[]

    ### Plot the periodic images
    # image on the left
    xl = x[x>lx*(1-pscale)]-lx
    yl = y[x>lx*(1-pscale)]
    cl = cid[x>lx*(1-pscale)]

    # image on the right
    xr = x[x<lx*pscale]+lx
    yr = y[x<lx*pscale]
    cr = cid[x<lx*pscale]
 
    # image on the bottom
    xb = x[y>ly*(1-pscale)]
    yb = y[y>ly*(1-pscale)]-ly
    cb = cid[y>ly*(1-pscale)]

    # image on the top
    xt = x[y<ly*pscale]
    yt = y[y<ly*pscale]+ly
    ct = cid[y<ly*pscale]

    # image on the top left
    xtl = x[  (x>lx*(1-pscale)) & (y<ly*pscale)] - lx
    ytl = y[  (x>lx*(1-pscale)) & (y<ly*pscale)] + ly
    ctl = cid[(x>lx*(1-pscale)) & (y<ly*pscale)]

    # image on the top right
    xtr = x[  (x<lx*pscale) & (y<ly*pscale)] + lx
    ytr = y[  (x<lx*pscale) & (y<ly*pscale)] + ly
    ctr = cid[(x<lx*pscale) & (y<ly*pscale)]

    # image on the bottom left
    xbl = x[  (x>lx*(1-pscale)) & (y>ly*(1-pscale))] - lx
    ybl = y[  (x>lx*(1-pscale)) & (y>ly*(1-pscale))] - ly
    cbl = cid[(x>lx*(1-pscale)) & (y>ly*(1-pscale))]

    # image on the bottome right
    xbr = x[  (x<lx*pscale) & (y>ly*(1-pscale))] + lx
    ybr = y[  (x<lx*pscale) & (y>ly*(1-pscale))] - ly
    cbr = cid[(x<lx*pscale) & (y>ly*(1-pscale))]

    xpp = np.concatenate((xl, xr, xb, xt, xtl, xtr, xbl, xbr))
    ypp = np.concatenate((yl, yr, yb, yt, ytl, ytr, ybl, ybr))
    cpp = np.concatenate((cl, cr, cb, ct, ctl, ctr, cbl, cbr))

    line2 = ax1.scatter(xpp, ypp, s=1, c=cpp, cmap=plt.cm.get_cmap('jet',quant_steps), 
                edgecolors='None', alpha=0.4, vmin=0, vmax=sim.nfils, norm=norm_filid)

    # left
    boundary.append([[0,ly*(-pscale/2)],[0,ly*(1+pscale/2)]])
    # right
    boundary.append([[lx,ly*(-pscale/2)],[lx,ly*(1+pscale/2)]])
    # bottom
    boundary.append([[lx*(-pscale/2),0],[lx*(1+pscale/2),0]])
    # top
    boundary.append([[lx*(-pscale/2),ly],[lx*(1+pscale/2),ly]])

    ### limits
    # Now for making pretty points
    nxy=[]; pxy=[];
    px =[]; py =[];
    nx =[]; ny =[];

    scale=70
    for i,dd in enumerate(dmax):
	angle = tail[i]
	if dd<0:
	    for ii in range(3):
		angle = angle+np.pi*2/3
		dx = np.cos(angle)*scale
		dy = np.sin(angle)*scale
		nx.append(xd[i])
		ny.append(yd[i])
		nxy.append([[xd[i],yd[i]],[xd[i]+dx,yd[i]+dy]]) 

	if dd>0:
	    dx = np.cos(angle)*scale
	    dy = np.sin(angle)*scale
	    px.append(xd[i]); py.append(yd[i])
	    pxy.append([[xd[i],yd[i]],[xd[i]+dx,yd[i]+dy]]) 
 
    lc0 = LineCollection(nxy, colors='k', linewidths=(5.0,)) # everything is a hack
    lc1 = LineCollection(nxy, colors='r', linewidths=(3,))


    lc2 = LineCollection(pxy, colors='k', linewidths=(5.0,)) # everything is a hack
    lc3 = LineCollection(pxy, colors='g', linewidths=(3,))
    lc0.set_zorder(99);     
    lc1.set_zorder(100)
    ax1.scatter(nx, ny, c='r', s=60, edgecolors='k', alpha=1.0, zorder=100)

    lc2.set_zorder(100);
    lc3.set_zorder(101)
    ax1.scatter(px, py, c='g', s=60, edgecolors='k', alpha=1.0, zorder=101)

    ax1.add_collection(lc0)
    ax1.add_collection(lc1)
    ax1.add_collection(lc2)
    ax1.add_collection(lc3)

    bc1 = LineCollection(boundary, colors='k', linewidths=(1,))

    lenbar=[]
    lenbar.append([[0,ly*(-pscale*1.3)],[100,ly*(-pscale*1.3)]])
    le1 = LineCollection(lenbar, colors='k', linewidths=(5,))
   
    ax1.add_collection(le1) 
    ax1.add_collection(bc1)

    ax1.axis('equal')
    plt.axis('off')
    #plt.savefig(savepath, dpi=200, bbox_inches='tight')

    plt.tight_layout()
    plt.savefig("./test2.png", bbox_inches='tight', dpi=200)
    fig.clf()
   
    return    
 

def plot_defect(xp, yp, phi, phi_nematic, cid, xdp, ydp, directors, corrected_directors, \
                    dmax, sim, cell_list, possible_defect_pts_p, pt_colors, xcm, ycm, savepath):
    """ plot the data"""
    
    print "Plotting"
    
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
    xcm2 = xcm/sim.bl
    ycm2 = ycm/sim.bl
    rcut = cell_list.rcut/sim.bl
    
    ### set plot properties

    ax_len = 0.9                          # Length of one subplot square box
    ax_b = 0.05                           # Beginning/offset of the subplot in the box
    ax_sep = 0.3                          # Separation length between two subplots
    total_subplots_in_x = 2               # Total number of subplots    
    fig = plt.figure()
    
    ### set more plot properties
    
    quant_steps = 2056
    norm_orient = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi) 
    norm_filid = mpl.colors.Normalize(vmin=0, vmax=sim.nfils)
    num_ticks = 5
    
    
    ### plot the frame

    subp = Subplots(fig, ax_len, ax_sep, ax_b, total_subplots_in_x) 
    
    ### 1) the entire field with bead orientations
        
    ax0 = subp.addSubplot()
    ax0.axis('equal')
    
    line0 = ax0.scatter(x, y, s=1, c=phi, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.7, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True)
    
    ### labels
        
    ax0.set_xlabel("$x/r_{0}$", fontsize=40)
    ax0.set_ylabel("$y/r_{0}$", fontsize=40)

    ### limits

    ax0.set_xlim((0, lx))
    ax0.set_ylim((0, ly))
    
    ### ticks
    
    ax0.xaxis.set_ticks(np.linspace(0, lx, num_ticks, endpoint=True))
    ax0.yaxis.set_ticks(np.linspace(0, ly, num_ticks, endpoint=True))
    ax0.tick_params(axis='both', which='major', labelsize=20)

    ax0.add_patch(Circle((xd, yd), rcut, edgecolor='gray',facecolor='gray',alpha=0.8))

    for j in range(len(possible_defect_pts)):
        ax0.plot(possible_defect_pts[j][0], possible_defect_pts[j][1], \
                     ls = '', marker='o', color=pt_colors[j], markersize=15, alpha=0.8)
        
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
                edgecolors='None', alpha=0.4, vmin=0, vmax=sim.nfils, norm=norm_filid, rasterized=True)
    
    ### labels
        
    ax1.set_xlabel("$x/r_{0}$", fontsize=40)

    ### limits

    ax1.set_xlim((0, lx))
    ax1.set_ylim((0, ly))
    
    ### ticks
    
    ax1.xaxis.set_ticks(np.linspace(0, lx, num_ticks, endpoint=True))
    ax1.yaxis.set_ticks(np.linspace(0, ly, num_ticks, endpoint=True))
    ax1.tick_params(axis='both', which='major', labelsize=20)    
    
    ax1.add_patch(Circle((xd, yd), rcut, edgecolor='gray',facecolor='gray',alpha=0.4))
    
    ### plot all checked possible points
    
    for j in range(len(possible_defect_pts)):
        ax1.plot(possible_defect_pts[j][0], possible_defect_pts[j][1], \
                     ls = '', marker='x', color=pt_colors[j], markersize=10, alpha=1.0)
        
        
    ### 3) zoom in to the defect
        
    ax2 = subp.addSubplot()
    ax2.axis('equal')
        
    ### plot all the beads along with their images 
    
    ax2.scatter(x, y, c=phi_nematic, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.3, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True)
    
    ax2.scatter(x-lx, y-ly, c=phi_nematic, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.3, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True) 
    
    ax2.scatter(x+lx, y-ly, c=phi_nematic, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.3, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True)
    
    ax2.scatter(x, y-ly, c=phi_nematic, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.3, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True)

    ax2.scatter(x-lx, y, c=phi_nematic, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.3, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True)

    ax2.scatter(x+lx, y, c=phi_nematic, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.3, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True)

    ax2.scatter(x-lx, y+ly, c=phi_nematic, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.3, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True)

    ax2.scatter(x, y+ly, c=phi_nematic, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.3, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True)

    ax2.scatter(x+lx, y+ly, c=phi_nematic, cmap=plt.cm.get_cmap('hsv',quant_steps), 
                edgecolors='None', alpha=0.3, vmin=-np.pi, vmax=np.pi, norm=norm_orient, rasterized=True)
                
    ### plot the point
    
    ax2.plot(xd, yd, ls='', marker='x', color='k', markersize=10, label=str(dmax))
    
    t = np.linspace(0, 2*np.pi, 1000)
    
    ### plot the circle around the point
    
    ax2.plot(rcut*np.cos(t) + xd, rcut*np.sin(t) + yd, color='k')
    ax2.quiver(xcm2, ycm2, np.cos(directors)*10, np.sin(directors)*10, alpha=1.0, headlength=0, headaxislength=0)

    
    tarvind = np.tan(np.linspace(0, np.pi, num=5, endpoint=False))
    xarvind = np.linspace(-rcut, rcut, num=100)

    for i in range(len(tarvind)):
        ax2.plot(xarvind*2.+xd, (xarvind*tarvind[i])*2.+yd, color='k', alpha=0.7)
    
    ### labels
        
    ax2.set_xlabel("$x/r_{0}$", fontsize=40)
    ax2.set_ylabel("$y/r_{0}$", fontsize=40)

    ### limits

    xdownlim = xd - 1.5*rcut
    xuplim = xd + 1.5*rcut
    ydownlim = yd - 1.5*rcut
    yuplim = yd + 1.5*rcut    
    
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
    
    ### labels
        
    ax3.set_xlabel("$j$", fontsize=40)
    ax3.set_ylabel("$\\theta(j)\\pi$", fontsize=40)
    
    ### legend
    
    ax3.legend()
    ax3.tick_params(axis='both', which='major', labelsize=20)   
    
    ### save

    plt.savefig(savepath, dpi=200, bbox_inches='tight', pad_inches=0.08)     
    fig.clf()
    
    return    
    
##############################################################################
    
