
""" Plot defects"""

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
    ax2.quiver(xcm2, ycm2, np.cos(directors)*10, np.sin(directors)*10, alpha=1.0)

    
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
    