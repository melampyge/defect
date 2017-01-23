
import ctypes
lib = ctypes.cdll.LoadLibrary('/usr/users/iff_th2/duman/Defects/Scripts/Defects/libperformance_tools.so')

class Performance_tools(object):
    def __init__(self):
        self.obj = lib.Performance_tools_new()

    ########################################################################
        
    def fill_neigh_matrix(self, neighs, llist, head, nsegx, nsegy, x, y, npoints, lx, ly, dcrit):

        neighsc = neighs.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        llistc = llist.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        headc = head.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        xc = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        yc = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        nsegxc = ctypes.c_int(nsegx)
        nsegyc = ctypes.c_int(nsegy)
        npointsc = ctypes.c_int(npoints)
        lxc = ctypes.c_double(lx)
        lyc = ctypes.c_double(ly)
        dcritc = ctypes.c_double(dcrit)
        
        lib.fill_neigh_matrix(self.obj, neighsc, llistc, headc,
                              nsegxc, nsegyc, xc, yc, npointsc, lxc,
                              lyc, dcritc)
        return

    
    ########################################################################

    def cluster_search(self, neighs, cl, npoints):

        neighsc = neighs.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        clc = cl.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        npointsc = ctypes.c_int(npoints)

        lib.cluster_search(self.obj, neighsc, clc, npointsc)
        return

        
