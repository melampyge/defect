
import numpy as np
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import misc_tools

##################################################################

# Density class: computes and stores the density as histograms

##################################################################

class PointDefects:


    ##############################################################
      
    def __init__(self, nsteps, line):
        """ initialize: allocate density array"""
        ### set histogram parameters
        line = line.split()
        self.rcut = float(line[1])
        self.nbins = int(line[2])
        self.ndmax = int(line[3])
        ### allocate arrays to store data
        self.points = np.zeros((nsteps, self.ndmax, 2))
        self.idx = np.zeros((nsteps, self.ndmax), dtype = int)
        return

    ##############################################################

    def gen_linked_list(self,x,y,lx,ly,natoms):
        """ generate the linked list for browsing the data"""
        ### define starting values
        nsegx = int(lx/self.rcut)
        nsegy = int(ly/self.rcut)
        llist = np.zeros((natoms), dtype = int) - 1
        head = np.zeros((nsegx*nsegy), dtype = int) - 1
        # loop over all atoms
        for i in range(natoms):
            segx = int(x[i]/lx*nsegx)
            segy = int(y[i]/ly*nsegy)
            cell = segx*nsegy + segy
            llist[i] = head[cell]
            head[cell] = i
        return llist, head, nsegx, nsegy

    ##############################################################

    def compute_single_defect(self,xd,yd,x,y,phi,llist,head,nsegx,nsegy,lx,ly,fig_cnt):
        """ compute the defect for a single position (xd, yd)"""
        # allocate array to store orientation
        nseg = 10
        orient = np.zeros((nseg))
        counter = np.zeros((nseg), dtype = int)
        # generate a ``considered'' array to check whether bead was used in the analysis
        considered = np.zeros((len(x)))
        # generate phi_nematic array
        phi_nematic = np.zeros((len(x)))
        for i in range(len(x)):
            pi = phi[i]
            if pi < 0:
                pi += np.pi
            phi_nematic[i] = pi
        # determine the segment of the current grid point
        segx = int(xd/lx*nsegx)
        segy = int(yd/ly*nsegy)
        cell = segx*nsegy + segy
        # loop over the linked list of the current bin
        for a in range(-1,2):
            i2 = (segx+a+nsegx)%nsegx
            for b in range(-1,2):
                j2 = (segy+b+nsegy)%nsegy
                # get head value
                cell = i2*nsegy + j2
                val = head[cell]
                while val != -1:
                    xi = x[val]
                    yi = y[val]
                    # check distance criterion
                    dx = misc_tools.nearest_neighbor(xi,xd,lx)
                    dy = misc_tools.nearest_neighbor(yi,yd,ly)
                    #dx = xi - xd
                    #dx = dx / lx
                    #dx = dx - math.floor(dx)
                    #dx = dx * lx
                    #dy = yi - yd
                    #dy = dy / ly
                    #dy = dy - math.floor(dy)
                    #dy = dy * ly
                    dsq = dx**2 + dy**2
                    if dsq <= self.rcut**2:
                        # compute the current angle
                        theta = math.atan2(dy,dx)
                        if theta < 0:
                            theta += 2*np.pi
                        # compute segment
                        segi = int(theta/2/np.pi*nseg)
                        # increase counter and add information
                        orient[segi] += phi_nematic[val]
                        counter[segi] += 1
                        considered[val] += 1
                    # get next item from the linked list
                    val = llist[val]
        # normlize orient function by the number of counted entries
        orient /= counter
        # correct orient by assuming no jumps > pi/2
        orient2 = np.copy(orient)
        for i in range(1,nseg):
            do = orient[i] - orient[i-1]
            if do > np.pi/4:
                do -= np.pi
            if do < -np.pi/4:
                do += np.pi
            orient2[i] = orient2[i-1] + do
        # scale orient and orient 2 by 2 pi
        #orient /= 2*np.pi
        #orient2 /= 2*np.pi

        # determine the defect strength: 
        idxmax = np.argmax(orient2)
        idxmin = np.argmin(orient2)
        omax = orient2[idxmax]
        omin = orient2[idxmin]
        dmax = omax - omin
        dmax = 2*dmax/2/np.pi
        dmax = round(dmax)
        dmax /= 2
        if idxmax < idxmin:
            dmax *= -1
        print dmax

        # go around the circle, show the cumulated function
        ### plot results for testing purpose
        fig = plt.figure()
        ax1 = plt.subplot(121)
        color = phi_nematic
        cmap = 'hsv'
        #color = considered
        #cmap = 'jet'

        plt.scatter(x,y,linewidth = 0, marker = 'o', c = color, s = 40, cmap = cmap)
        plt.scatter(x-lx,y-ly,linewidth = 0, marker = 'o', c = color, s = 40, cmap = cmap)
        plt.scatter(x,y-ly,linewidth = 0, marker = 'o', c = color, s = 40, cmap = cmap)
        plt.scatter(x+lx,y-ly,linewidth = 0, marker = 'o', c = color, s = 40, cmap = cmap)

        plt.scatter(x-lx,y,linewidth = 0, marker = 'o', c = color, s = 40, cmap = cmap)
        plt.scatter(x+lx,y,linewidth = 0, marker = 'o', c = color, s = 40, cmap = cmap)

        plt.scatter(x-lx,y+ly,linewidth = 0, marker = 'o', c = color, s = 40, cmap = cmap)
        plt.scatter(x,y+ly,linewidth = 0, marker = 'o', c = color, s = 40, cmap = cmap)
        plt.scatter(x+lx,y+ly,linewidth = 0, marker = 'o', c = color, s = 40, cmap = cmap)

        plt.colorbar()
        ax1.plot(xd,yd,ls = '', marker = 'x', color = 'k',label=str(dmax))
        ax1.axis('equal')
        t = np.linspace(0,2*np.pi,1000)
        ax1.plot(self.rcut*np.cos(t) + xd, self.rcut*np.sin(t) + yd, color = 'k')
        ax1.set_xlim([(xd - 1.5*self.rcut), (xd + 1.5*self.rcut)])
        ax1.set_ylim([(yd - 1.5*self.rcut), (yd + 1.5*self.rcut)])
        ax2 = plt.subplot(122)
        ax2.plot(orient, color = 'r', label = 'defect = ' + str(dmax), marker = '^')
        ax2.plot(orient2, color = 'g', label = 'defect = ' + str(dmax), marker = '^')
        ax2.legend()
        #plt.show()
        plt.savefig('/usr/users/iff_th2/duman/Desktop/Defect/figure_' + str(fig_cnt) + '.png', dpi=200, bbox_inches='tight', pad_inches=0.08)
        plt.close()
        
        dfct = 1
        return dfct

    ##############################################################

    def grid_search(self,x,y,phi,lx,ly,natoms,llist,head,nsegx,nsegy):
        """ define defect parameter on a grid"""
        # allocate arrays
        xgrid = np.zeros((self.nbins))
        ygrid = np.zeros((self.nbins))
        defect_grid = np.zeros((self.nbins, self.nbins))
        # generate grid points
        wx = lx/self.nbins
        wy = ly/self.nbins
        for i in range(self.nbins):
            xgrid[i] = wx*i
        for i in range(self.nbins):
            ygrid[i] = wy*i
        # loop over grid points and compute grid defect
        fig_cnt = 0
        for i in range(self.nbins):
            for j in range(self.nbins):
                xgi = xgrid[i]
                ygi = ygrid[j]
                fig_cnt += 1
                defect_grid[i,j] = self.compute_single_defect(xgi,ygi,x,y,phi,llist,head,nsegx,nsegy,lx,ly,fig_cnt)
        exit()
        return xgrid, ygrid, defect_grid
        

    ##############################################################

    def compute(self,step,x,y,phi,lx,ly,natoms):
        """ compute the location of point defects, check whether this
            defect already appeared in the previous step, assign id
            for the defect based on whether it is old or new"""
        ### generate a linked list of the particles
        llist, head, nsegx, nsegy = self.gen_linked_list(x,y,lx,ly,natoms)
        ### search for defect points on a grid
        xgrid, ygrid, defect_grid = self.grid_search(x,y,phi,lx,ly,natoms,llist,head,nsegx,nsegy)
        ### scan surrounding of the defect for exact position
        
        ### check whether observed defects are new or old
        return

