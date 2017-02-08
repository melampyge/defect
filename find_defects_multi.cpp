
/* identify possible topological defect structures on multiple processors with MPI */

// COMPILATION AND RUN COMMANDS:
// mpicxx -Wl,-rpath=${HOME}/hdf5_parallel/lib -L${HOME}/hdf5_parallel/lib -I${HOME}/hdf5_parallel/include -I${HOME}/Eigen -lhdf5 find_defects_multi.cpp compute_defect.cpp misc_tools.cpp read_write.cpp -o fdef_multi
//  mpirun -np 16 /usr/users/iff_th2/duman/Defects/Scripts/Defects/fdef_multi /usr/users/iff_th2/duman/Defects/Output/outcpp.h5 /usr/users/iff_th2/duman/Defects/Output
// mpirun -np 8 valgrind --leak-check=full --track-origins=yes /usr/users/iff_th2/duman/Defects/Scripts/Defects/fdef_multi /usr/users/iff_th2/duman/Defects/Output/outcpp.h5 /usr/users/iff_th2/duman/Defects/Output

//////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include "read_write.hpp"
#include "misc_tools.hpp"
#include "compute_defect.hpp"
#include "MersenneTwister.h"
#include "mpi.h"

#define pi M_PI

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char *argv[]) {

  // get the file name by parsing
  
  char *filename = argv[1];
  char *savefolder = argv[2];

  // initiate MPI 
  
  MPI_Init(&argc, &argv);
  MPI_Comm MPI_COMM_WORLD;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);		// process rank
  MPI_Comm_size(MPI_COMM_WORLD, &size);		// number of processes
  
  // read in general simulation data
  
  int nsteps, nbeads, nsamp, nfils, nbpf;
  nsteps = nbeads = nsamp = nfils = nbpf = 0;
  double lx, ly, dt, density, kappa, fp, bl, sigma;
  lx = ly = dt = density = kappa = fp = bl = sigma = 0.;
  read_sim_data(filename, nsteps, nbeads, nsamp, nfils, nbpf, lx, ly, dt, density, kappa, fp, bl, sigma);
  
  // print simulation information
  
  if (rank == 0) {
    cout << "Finding the defects in the following file: \n" << filename << endl;
    cout << "nsteps = " << nsteps << endl;
    cout << "nfils = " << nfils << endl;
  }
  
  // random number generator
  
  MTRand randi;
  unsigned int saat = (unsigned int)(rank+43217);	// process unique seed for PRNG
  randi.seed(saat);
  
  // define analysis properties
  
  double rcut = 15.; 		// cut radius to calculate the order parameter inside
  int npoints = 60000;		// number of main points to search for
  npoints /= size;  		// divide the points between the processes
  double rn = 0.65*rcut;	// inner radius of the shell where neighbor search is going to be performed
  double dn = 5.;		// depth of the shell (THIS IS NOT USED AT THE MOMENT!)
  int nr = 2;			// total number of recursions to search friends and then friends of friends and so on 
  int nn = 10; 			// number of neighbor points to check in each recursion
  double dcut = 0.1;		// defect strength cut interval
  int max_number_of_defects = npoints*nn*nr; 	// maximum number of defects possible
  
  // allocate main points to check array
  
  double *xpoints = new double[npoints];
  double *ypoints = new double[npoints];
  int *check_again = new int[npoints];
  for (int i = 0; i < npoints; i++) {
    xpoints[i] = 0.;  ypoints[i] = 0.;  check_again[i] = 0;
  }
  
  // define the hashed linked list properties
  
  int nsegx = static_cast<int>(lx/rcut);
  int nsegy = static_cast<int>(ly/rcut);
  int nseg = nsegx*nsegy;

  // open the file pointer
  
  hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

  // get the dataset in the file
  /* DATASET (H5D) is the raw data (either singular or in arrays) */
  
  hid_t dataset = H5Dopen(file, "/positions/x", H5P_DEFAULT);
  
  // get dataspace of the selected dataset
  /* DATASPACE (H5S) describes the number of dimensions and the size of the dataset in those dimension */
  
  hid_t dataspace = H5Dget_space(dataset);
 
  /* READ THE DATA
  load the position data into the memory step by step 
  */
   
  for (int step = 0; step < nsteps; step++) {
    
    // print progress
    
    if (rank == 0) {
      cout << "at step: " << step << " of " << nsteps << " steps" << endl;
    }
    
    // allocate point arrays
    
    double *x = new double[nbeads];
    double *y = new double[nbeads];
    double *phi = new double[nbeads];
    for (int i = 0; i < nbeads; i++) {
      x[i] = 0.;  y[i] = 0.;  phi[i] = 0.;
    }
    
    // load the positions of this timestep
    
    read_single_pos_data(step, dataset, dataspace, x, y, nbeads);

    // allocate hashed linked list arrays
    
    double *llist = new double[nbeads];
    for (int i = 0; i < nbeads; i++) llist[i] = -1.;
    double *head = new double[nseg];
    for (int i = 0; i < nseg; i++) head[i] = -1.;
    
    // generate hashed linked list to browse data points via positions
    
    gen_linked_list(llist, head, rcut, nsegx, nsegy, nseg, nbeads, x, y, lx, ly);

    // calculate bead orientations and turn them headless 
    
    calc_headless_orient(phi, x, y, lx, ly, nfils, nbpf);
    
    // allocate the arrays for possible defect points at this timestep with some large buffer
    
    double *xbuffer = new double[max_number_of_defects];
    double *ybuffer = new double[max_number_of_defects]; 
    double *dbuffer = new double[max_number_of_defects];
    for (int i = 0; i < max_number_of_defects; i++) {
      xbuffer[i] = -20.;  ybuffer[i] = -20.;  dbuffer[i] = -20.;
    }
    int cnt_defects = -1;
      
    // check the main points and compute the defect strength per point. start the recursion loop for possible defect points.
    
    search_main_pts(xbuffer, ybuffer, dbuffer, check_again, cnt_defects, xpoints, ypoints, npoints, x, y, phi, head, llist, rcut, nsegx, nsegy, rn, nn, nr, dcut, lx, ly, randi);
        
    // store only the possible defect points for each process
    
    double *xdefects, *ydefects, *ddefects;
    if (cnt_defects != -1) {
      xdefects = new double[cnt_defects+1];
      ydefects = new double[cnt_defects+1];
      ddefects = new double[cnt_defects+1];
      int cnter = -1;
      for (int i = 0; i < max_number_of_defects; i++) {
	if (xbuffer[i] != -20.) {
	  cnter += 1;
	  xdefects[cnter] = xbuffer[i];
	  ydefects[cnter] = ybuffer[i];
	  ddefects[cnter] = dbuffer[i];
	}
      }
    }
    else {
      xdefects = NULL;
      ydefects = NULL;
      ddefects = NULL;
    }
    delete [] xbuffer;
    delete [] ybuffer; 
    delete [] dbuffer;
    
    // communicate number of defects across the processes to create the hyperslab dimensions for writing data

    int count[size];
    int num_of_defects = cnt_defects+1;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgather(&num_of_defects, 1, MPI_INT, count, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    int offset[size];
    offset[0] = 0;
    int total_num_of_defects = count[0];    
    for (int i = 1; i < size; i++) {
      offset[i] = 0;
      int j = i;
      while (j > 0) {
	j--;
	offset[i] += count[j];  
      }
      total_num_of_defects += count[i];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // save the possible defect points for this timestep among all the processes

    write_h5_data(savefolder, step, xdefects, ydefects, ddefects, total_num_of_defects, rank, size, count, offset, MPI_COMM_WORLD);
    
    // deallocate the arrays
  
    delete [] llist;
    delete [] head;
    delete [] phi;
    delete [] x;
    delete [] y;
    delete [] xdefects; 
    delete [] ydefects;
    delete [] ddefects;
    
    // sync all the processes at the end of the timestep
    
    MPI_Barrier(MPI_COMM_WORLD);

  } // timestep loop
  
  // deallocate the arrays
  
  delete [] xpoints;	
  delete [] ypoints;
  delete [] check_again;
    
  // close the file pointers
  
  H5Sclose(dataspace);
  H5Dclose(dataset);
  H5Fclose(file);
  
  MPI_Finalize();
    
  return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
