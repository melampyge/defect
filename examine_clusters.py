
""" Cluster the defect points to find the ultimate defect points"""

##############################################################################

import numpy as np
import math
import performance_toolsWrapper
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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

def nearest_neighbor(x1, x2, lx):
    """ compute the nearest neighbor distance between two points"""
    
    dx1 = x1 - x2
    dx2 = x1 - x2 + lx
    dx3 = x1 - x2 - lx
    if dx1**2 < dx2**2 and dx1**2 < dx3**2:
        return dx1
    if dx2**2 < dx3**2:
        return dx2
    return dx3

##############################################################################

def gen_linked_list(x, y, lx, ly, dcrit, npoints):
    """ generate a hashed linked list to browse data points based on distance"""
    
    ### determine the number of cells in each direction

    nsegx = int(lx/dcrit)
    nsegy = int(ly/dcrit)

    ### allocate head and llist
    
    ncells = nsegx*nsegy
    head = np.zeros((ncells), dtype=np.int32) - 1
    llist = np.zeros((npoints), dtype=np.int32) - 1

    ### fill list and head
    
    for i in range(npoints):
        segx = int(x[i]/lx*nsegx)
        segy = int(y[i]/ly*nsegy)
        if segx >= nsegx:
            segx -= nsegx
            print x[i], lx, segx, nsegx
        elif segx < 0:
            segx += nsegx
            print x[i], lx, segx, nsegx            
        if segy >= nsegy:
            segy -= nsegy-1
            print y[i], ly, segy, nsegy
        elif segy < 0:
            segy += nsegy
            print y[i], ly, segy, nsegy            
        cell = segx*nsegy + segy
        llist[i] = head[cell]
        head[cell] = i

    return nsegx, nsegy, head, llist
    
##############################################################################

def find_clusters(x, y, dcrit, npoints, sim):
    """ search for clusters; two points are defined as being
        part of the same cluster if they are within dcrit distance of each other"""

    ### allocate required arrays
    
    clusters = np.zeros((npoints), dtype=np.int32) - 1
    neighbors = np.zeros((npoints, npoints), dtype=np.int32)
    
    ### generate a linked list
    
    nsegx, nsegy, head, llist = gen_linked_list(x, y, sim.lx, sim.ly, dcrit, npoints)
    
    ### buld a neighborhood matrix based on the criterion dcrit
 
    neighs = neighbors.ravel()
    performance_tools = performance_toolsWrapper.Performance_tools()
    performance_tools.fill_neigh_matrix(neighs, llist, head, nsegx, nsegy, x, y, npoints, sim.lx, sim.ly, dcrit)
    
    ### recursive search for clusters within the neighbor matrix

    performance_tools.cluster_search(neighs, clusters, npoints)
    
    return clusters
    
##############################################################################

def transform_cluster_data(clusters, npoints):
    """ transform the data representation of clusters such that cluster list contains 
    the points inside the cluster"""
    
    ### find the maximum number of clusters
    
    clmax = max(clusters)
    
    ### add empty lists to cluster list
    
    cl_list = []
    for j in range(clmax+1):
        cl_list.append([])
        
    ### fill lists with point ids
    
    for i in range(npoints):
        cid = clusters[i]
        cl_list[cid].append(i)
    
    return cl_list

##############################################################################
    
def correct_cluster_pbc(x, y, clusters, lx, ly, dcrit, npoints):
    """ correct the cluster point coordinates with periodic boundary conditions"""
    
    ### get number of clusters
    
    nclusters = len(clusters)
    
    ### initialize output clusters and isolated array
    
    xcluster = np.copy(x)
    ycluster = np.copy(y)
    isolated = np.ones((nclusters), dtype=int)
    
    ### loop over all clusters
    
    for i in range(nclusters):
        
        ### allocate array to store cluster + copies
        
        npts = len(clusters[i])
        
        ### expand cluster to periodic images
        
        xcl, ycl = expand_cluster(x, y, lx, ly, clusters, i)
        
        ### try first a simple 1D histogram method
        
        flag_1d = cluster_hist_1d(xcl, ycl, lx, ly, npts, dcrit)
        
        ### if the 1D histogram approach fails, correct the pbcs point by point
        
        if flag_1d == 0:
            isolated[i] = 0     # cluster is not isolated
            correct_pbc_single(xcl, ycl, lx, ly, npts)
            
        ### adjust center of mass of the entire cluster
        
        adjust_com_cluster(xcl, ycl, lx, ly, npts)
        
        ### copy coordinates to the cluster array
        
        for j in range(npts):
            mi = clusters[i][j]
            xcluster[mi] = xcl[j]
            ycluster[mi] = ycl[j]
            
    return xcluster, ycluster, isolated
    
##############################################################################

def expand_cluster(x, y, lx, ly, clusters, i):
    """ expand the current cluster to periodic images"""
    
    npts = len(clusters[i])
    xcl = np.zeros((9*npts))
    ycl = np.zeros((9*npts))
    
    ### create copies of the atoms of all clusters
    
    l = 0
    for j in range(npts):
        mi = clusters[i][j]
        xcl[l] = x[mi]
        ycl[l] = y[mi]
        l = l + 1
        
    ### add pbc copies
    
    for j in range(-1,2):
        for k in range(-1,2):
            if j == 0 and k == 0:
                continue
            for m in range(npts):
                xcl[l] = xcl[m] + j*lx
                ycl[l] = ycl[m] + k*ly
                l = l + 1
                
    return xcl, ycl

##############################################################################

def cluster_hist_1d(xcl, ycl, lx, ly, npts, dcrit):
    """ try to detect cluster by 1d histograms to correct for pbcs"""
    
    ### compute 1D histograms
    
    nxbins = int(3*lx/(dcrit))
    #bx = 3*lx/nxbins
    hx, xedges = np.histogram(xcl, bins = nxbins, range = (-lx,2*lx))
    nybins = int(3*ly/(dcrit))
    #by = 3*ly/nybins
    hy, yedges = np.histogram(ycl, bins = nybins, range = (-ly,2*ly))
    
    ### check whether the approach is feasible
    
    if 0 in hx and 0 in hy:  # 1d approach works
    
        ### find boundaries
        
        xmin = -1
        xmax = -1
        ymin = -1
        ymax = -1
        jstop = -1
        for j in range(nxbins-1):
            if hx[j] == 0 and hx[j+1] > 0:
                xmin = xedges[j+1]
                jstop = j
                break
        for j in range(jstop,nxbins-1):
            if hx[j] > 0 and hx[j+1] == 0:
                xmax = xedges[j+1]
                break
        for j in range(nybins-1):
            if hy[j] == 0 and hy[j+1] > 0:
                ymin = yedges[j+1]
                jstop = j
                break
        for j in range(jstop,nybins-1):
            if hy[j] > 0 and hy[j+1] == 0:
                ymax = yedges[j+1]
                break
            
        ### map all point coordinates into the relevant area
        
        for j in range(npts):
            while xcl[j] < xmin:
                xcl[j] += lx
            while xcl[j] > xmax:
                xcl[j] -= lx
            while ycl[j] < ymin:
                ycl[j] += ly
            while ycl[j] > ymax:
                ycl[j] -= ly
        return 1
    return 0

##############################################################################

def correct_pbc_single(xcl, ycl, lx, ly, npts):
    """ correct periodic boundary conditions of all points one by one separately"""
    
    ### loop over individual points in the cluster and connect nearest neighbors
    
    for j in range(npts-1):
        x0 = xcl[j]
        y0 = ycl[j]
        x1 = xcl[j+1]
        y1 = ycl[j+1]
        dx = nearest_neighbor(x0, x1, lx)
        dy = nearest_neighbor(y0, y1, ly)
        xcl[j+1] = x0 - dx
        ycl[j+1] = y0 - dy

    ### loop over all points in the cluster and adjust com position
    
    for j in range(npts):
        xcl[j] += -math.floor(xcl[j]/lx)*lx
        ycl[j] += -math.floor(ycl[j]/ly)*ly

    return

##############################################################################

def adjust_com_cluster(xcl, ycl, lx, ly, npts):
    """ move cluster such that com is in periodic box"""
    
    comx = np.average(xcl[0:npts])
    comy = np.average(ycl[0:npts])
    xcl[0:npts] += -math.floor(comx/lx)*lx
    ycl[0:npts] += -math.floor(comy/ly)*ly

    return

##############################################################################
    
def find_com_clusters(xcl, ycl, clusters, lx, ly):
    """ find the center of mass of clusters"""
    
    ### get the number of clusters
    
    nclusters = len(clusters)   
    xcm = np.zeros((nclusters), dtype=np.float64)
    ycm = np.zeros((nclusters), dtype=np.float64)
    
    ### run over the clusters
    
    for i in range(nclusters):
        
        ### run over the points in the cluster
        
        npts = len(clusters[i])
        for j in range(npts):
            mi = clusters[i][j]     # this accesses the particle index
            xcm[i] += xcl[mi]
            ycm[i] += ycl[mi]
        xcm[i] /= npts
        ycm[i] /= npts

    ### put the center of masses back inside the box
    
    xcm += -np.floor(xcm/lx)*lx
    ycm += -np.floor(ycm/ly)*ly
    
    return xcm, ycm  
    
##############################################################################
    
def find_com_clusters_weight(xcl, ycl, clusters, lx, ly, d):
    """ find the center of mass of clusters"""
    
    # clusters is a list containing the cluster id in the first dimension
    # and the number of points inside the cluster inside the each list element
    # d: is an array containing defect strength of the point
    # periodic boundary conditions are taken care of
    
    ### get the number of clusters
    
    nclusters = len(clusters)
    
    xcm = np.zeros((nclusters), dtype=np.float64)
    ycm = np.zeros((nclusters), dtype=np.float64)
    
    ### build weights array based on defect strengths
    
    dweight = np.zeros(d.shape)
    dtotal  = np.zeros(d.shape)         # normalization of weights
                                        # sum of wghts of defects in a cluster=1
    dweight[d<0] = 1./np.abs(d[d<0]+0.5)    # -1/2 defect weights
    dweight[d>0] = 1./np.abs(d[d>0]-0.5)    # +1/2 defect weights
    
    ### run over all clusters
    
    for i in range(nclusters):
        
        ### run over all points in the cluster
        
        npts = len(clusters[i])
        for j in range(npts):
            mi = clusters[i][j]         # accesses the particle index
            dtotal[i] += dweight[mi]    # adds up total weight of all defects
    
    ### run over the clusters
    
    for i in range(nclusters):
        
        ### run over the points in the cluster
        
        npts = len(clusters[i])
        for j in range(npts):
            mi = clusters[i][j]         # accesses the particle index
            xcm[i] += xcl[mi]*dweight[mi]
            ycm[i] += ycl[mi]*dweight[mi]
        xcm[i] /= dtotal[i]
        ycm[i] /= dtotal[i]

    ### put the center of masses back inside the box
    
    xcm += -np.floor(xcm/lx)*lx
    ycm += -np.floor(ycm/ly)*ly
    
    return xcm, ycm  
    
##############################################################################
    
def separate_clusters(cl_list, clusters, d):
    """ separate clusters by their defect strength"""

    ### get the number of clusters
    
    nclusters = len(cl_list)
    cl_defect_strength = {}
    new_cluster_cnt = 0             # assign new clusters in case of a defect strength conflict
    
    ### run over the clusters
    
    for i in range(nclusters):
        
        ### assign a placeholder defect strength to the cluster
        
        cl_defect_strength[i] = 0
        cluster_is_separated = False
        
        ### run over the points in the cluster
        
        npts = len(cl_list[i])
        k = -1
        
        for j in range(npts):
            
            k += 1
            mi = cl_list[i][k]
            #print 'cluster_d = ', cl_defect_strength[i], ' / d = ', d[mi]
            
            ### if the defect strength is assigned for the first time ...
            
            if cl_defect_strength[i] == 0:
                #print 'Cluster ', i, ' is assigned ', d[mi], ' for the first time.'
                cl_defect_strength[i] = d[mi] 

            ### if the defect strength of the current point is different than the assigned one ...
            
            elif cl_defect_strength[i] > 0 and d[mi] < 0:
                
                if cluster_is_separated == False:
                    cluster_is_separated = True
                    new_cluster_cnt += 1
                    new_cluster_id = len(cl_list) 
                    cl_list.append([])
                    #print 'New cluster is created. new_cluster_cnt = ', new_cluster_cnt, 'new_cluster_id = ', new_cluster_id
                
                cl_defect_strength[new_cluster_id] = d[mi] 
                clusters[mi] = new_cluster_id
                cl_list[new_cluster_id].append(mi)
                del cl_list[i][k]
                k -= 1
                
            ### if the defect strength of the current point is different than the assigned one ...
            
            elif cl_defect_strength[i] < 0 and d[mi] > 0:
                
                if cluster_is_separated == False:
                    cluster_is_separated = True
                    new_cluster_cnt += 1
                    new_cluster_id = len(cl_list) 
                    cl_list.append([])
                    #print 'New cluster is created. new_cluster_cnt = ', new_cluster_cnt, 'new_cluster_id = ', new_cluster_id
                
                cl_defect_strength[new_cluster_id] = d[mi] 
                clusters[mi] = new_cluster_id
                cl_list[new_cluster_id].append(mi) 
                del cl_list[i][k]
                k -= 1
               
    return
    
##############################################################################

def plot_clusters(xp, yp, xcmp, ycmp, cl_list, cl_id, sim, xallp, yallp, cid):
    """ generate a plot of the atoms color coded with the cluster they belong"""
    
    savefolder = '/usr/users/iff_th2/duman/Desktop/figcontainer'
    savepath = savefolder + '/cluster.png'

    print 'Number of clusters: ', len(cl_list)
#    print 'List of clusters with point ids: ', cl_list
#    for j in range(len(cl_list)):
#        clustersize = len(cl_list[j])
#        if clustersize > 0:
#            for i in range(clustersize):
#                pid = cl_list[j][i]
#                print 'Particle id of the point in the cluster: ', j, i, pid, xp[pid], yp[pid]
#    print 'Cluster id of points: ', cl_id
    print "Plotting the clusters"
    
    ### normalize for plotting purposes
    
    lx = sim.lx/sim.bl
    ly = sim.ly/sim.bl
    x = xp/sim.bl
    y = yp/sim.bl 
    xcm = xcmp/sim.bl
    ycm = ycmp/sim.bl
    xall = xallp/sim.bl
    yall = yallp/sim.bl
    
    ### set plot properties

    ax_len = 0.9                          # Length of one subplot square box
    ax_b = 0.05                           # Beginning/offset of the subplot in the box
    ax_sep = 0.3                          # Separation length between two subplots
    total_subplots_in_x = 2               # Total number of subplots    
    fig = plt.figure()
    
    ### set more plot properties
    
    quant_steps = 2056
    norm_cluster = mpl.colors.Normalize(vmin=0, vmax=len(cl_list)) 
    norm_filid = mpl.colors.Normalize(vmin=0, vmax=sim.nfils)    
    num_ticks = 5
    
    ### plot the frame

    subp = Subplots(fig, ax_len, ax_sep, ax_b, total_subplots_in_x)         
    ax0 = subp.addSubplot()
    ax0.axis('equal')
    line0 = ax0.scatter(x, y, s=10, c=cl_id, cmap=plt.cm.get_cmap('jet',quant_steps), 
                edgecolors='None', alpha=0.7, vmin=0, vmax=len(cl_list), norm=norm_cluster, rasterized=True)
#    line1 = ax0.scatter(xall, yall, s=1, c=cid, cmap=plt.cm.get_cmap('jet',quant_steps), 
#                edgecolors='None', alpha=0.4, vmin=0, vmax=sim.nfils, norm=norm_filid, rasterized=True)
    ax0.plot(xcm, ycm, 'x', markersize=10, color='k')    
    
    ### labels
        
    ax0.set_xlabel("$x/r_{0}$", fontsize=40)
    ax0.set_ylabel("$y/r_{0}$", fontsize=40)

    ### limits

    ax0.set_xlim((-50, lx+50))
    ax0.set_ylim((-50, ly+50))
    
    ### ticks
    
    ax0.xaxis.set_ticks(np.linspace(0, lx, num_ticks, endpoint=True))
    ax0.yaxis.set_ticks(np.linspace(0, ly, num_ticks, endpoint=True))
    ax0.tick_params(axis='both', which='major', labelsize=20)

    ### save
    
    plt.savefig(savepath, dpi=200, bbox_inches='tight', pad_inches=0.08) 
    fig.clf()
      
    return

##############################################################################

def cluster_analysis(points, dcrit, sim, step, xall, yall, cid):
    """ find clusters within the list of data points with a distance criterion"""
    
    ### discern information about the data points
    
    npoints = len(points[0])  
    x = np.array(points[0], dtype=np.float64)
    y = np.array(points[1], dtype=np.float64)
    d = np.array(points[2], dtype=np.float64)

    ### find the clusters among the data points
    # clusters is a per point array with each element representing the cluster id the point belongs to
    
    clusters = find_clusters(x, y, dcrit, npoints, sim)
        
    ### transform the cluster data such that each cluster contains a list of points in that cluster
    
    cl_list = transform_cluster_data(clusters, npoints)
    
    ### separate clusters by their defect strengths
    
    separate_clusters(cl_list, clusters, d)
    
    ### correct the cluster point positions with periodic boundary conditions
    
    xclusters, yclusters, isolated = correct_cluster_pbc(x, y, cl_list, sim.lx, sim.ly, dcrit, npoints)
    
    ### find the center of mass of clusters

    #xcm, ycm = find_com_clusters(xclusters, yclusters, cl_list, sim.lx, sim.ly)
    xcm, ycm = find_com_clusters_weight(xclusters, yclusters, cl_list, sim.lx, sim.ly, d)    
    
    ### plot the clusters
    
    #plot_clusters(xclusters, yclusters, xcm, ycm, cl_list, clusters, sim, xall, yall, cid)    
    
    return xcm, ycm      
                
##############################################################################