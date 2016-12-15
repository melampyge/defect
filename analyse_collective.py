#!/usr/local/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import codecs
import read_char
import misc_tools
import density
import numberfluctuation
import voronoidensity
import orientvel
import pointdefects
import velocityworticity

##################################################################

# AnalyseCollective class: runs the analysis

##################################################################

class AnalyseCollective:
    

    ##############################################################

    def init_analysis(self, infilename):
        """ Initialize the analysis:
                (1) open files, get initial header information
                (2) read analysis options one by one and
                (3) initizalize analysis tools"""
        ### Get Gloal information from input file
        optfile = open(infilename, 'r')
        optfile.readline()
        optfile.readline()
        line = optfile.readline()
        line = line.split()
        self.fname = line[-1]
        line = optfile.readline()
        line = line.split()
        self.hname = line[-1]
        line = optfile.readline()
        line = line.split()
        self.npol = int(line[-1])
        optfile.readline()

        ### open charfile and read header; get natoms + nsteps
        self.ifile = codecs.open(self.fname, 'r', 'UTF-8')
        self.hfile = open(self.hname, 'r')
        self.natoms, self.nsteps = read_char.read_first(self.hfile)

        ### loop over the input file an initialize analysis methods
        # density
        optfile.readline()
        line = optfile.readline()
        self.density_flag = int(line[0])
        if self.density_flag:
            self.density = density.Density(self.nsteps, line)
        optfile.readline()
        # number fluctuations
        optfile.readline()
        line = optfile.readline()
        self.nf_flag = int(line[0])
        if self.nf_flag:
            self.numberfluctuation = numberfluctuation.NumberFluctuation(self.nsteps, self.natoms, line)
        optfile.readline()
        # voronoi density
        optfile.readline()
        line = optfile.readline()
        self.voronoi_flag = int(line[0])
        if self.voronoi_flag:
            self.voronoidensity = voronoidensity.VoronoiDensity(self.nsteps, self.natoms, self.npol, line)
        optfile.readline()
        # velocity / worticity
        optfile.readline()
        line = optfile.readline()
        self.velocity_flag = int(line[0])
        if self.velocity_flag:
            self.velocityworticity = velocityworticity.VelocityWorticity(self.nsteps, line)
        optfile.readline()
        # orientation / velocity
        optfile.readline()
        line = optfile.readline()
        self.orientvel_flag = int(line[0])
        if self.orientvel_flag:
            self.orientvel = orientvel.OrientVel(self.nsteps, self.natoms, line)
        optfile.readline()
        # point defect detection
        optfile.readline()
        line = optfile.readline()
        self.pointdefects_flag = int(line[0])
        if self.pointdefects_flag:
            self.pointdefects = pointdefects.PointDefects(self.nsteps, line)
        optfile.readline()
        # add further methods here
        ### close the file with the analysis options
        optfile.close()
        return


    ##############################################################

    def run_analysis(self):
        """ loop over all timesteps and call different analysis tools"""
        ### skip some snapshots for testing purposes
        nskip = 199
        read_char.skip_snapshots(self.hfile, self.ifile, nskip)
        ### read in the first two steps (required for velocity related computations
        xs_old, ys_old, lx_old, ly_old, tstep_old, natoms_old = read_char.read_snapshot(self.hfile, self.ifile)
        x_old = xs_old*lx_old
        y_old = ys_old*ly_old
        xs,ys,lx,ly,tstep,natoms = read_char.read_snapshot(self.hfile, self.ifile)
        x = xs*lx
        y = ys*ly
        ### loop over all steps of the input file
        for step in range(nskip+1,self.nsteps-1):
            print step
            ### read in coordinates (as required)
            xs_new,ys_new,lx_new,ly_new,tstep_new,natoms_new = read_char.read_snapshot(self.hfile, self.ifile)
            x_new = xs_new*lx_new
            y_new = ys_new*ly_new
            ### compute further current per/atom quantities
            phi = misc_tools.compute_orientation(x,y,lx,ly,self.npol)
            vx,vy = misc_tools.compute_velocity(x_old,y_old, x_new, y_new, lx, ly, tstep_old, tstep_new, natoms)
            ### start desired analysis methods
            # density
            if self.density_flag:
                self.density.compute(step,x,y,lx,ly,natoms, plot = 'False')
            # number fluctuations
            if self.nf_flag:
                self.numberfluctuation.compute(step,xs,ys, plot = 'False')
            # voronoi density
            if self.voronoi_flag:
                self.voronoidensity.compute(step,x,y,lx,ly,natoms, plot = 'False')
            # velocity / worticity
            if self.velocity_flag:
                self.velocityworticity.compute(step,x,y,vx,vy,natoms,lx,ly, plot = 'False')
            # orientation / velocity
            if self.orientvel_flag:
                self.orientvel.compute(step,x,y,vx,vy,phi,natoms, plot = 'False')
            # defect points
            if self.pointdefects_flag:
                self.pointdefects.compute(step,x,y,phi,lx,ly,natoms)
            ### move coordinate arrays
            xs_old = np.copy(xs)
            ys_old = np.copy(ys)
            x_old = np.copy(x)
            y_old = np.copy(y)
            tstep_old = tstep
            xs = np.copy(xs_new)
            ys = np.copy(ys_new)
            x = np.copy(x_new)
            y = np.copy(y_new)
            tstep = tstep_new
        return

    ##############################################################

    def run(self,infilename):
        """ equivalent to main function, runs the analysis""" 
        ### initizlize the analysis
        self.init_analysis(infilename)
        ### run the analysis
        self.run_analysis()
        ### store selected results
        self.store_results()
        return

    ##############################################################

try:
    infilename = sys.argv[1]
except:
    print 'Usage: ' + sys.argv[0] + '       input file'
    exit()

if __name__ == '__main__':
    AnalyseCollective().run(infilename)
