
/* routines to read from and write to hdf5 files */

#include "read_write.hpp"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////////////

int read_integer_data (hid_t file, char *path_in_file, int *buffer) {
  /* wrapper to read integer data from hdf5 file --note that buffer needs to be an array of size 1 for single entries-- */
    
  hid_t dataset = H5Dopen(file, path_in_file, H5P_DEFAULT);
  herr_t status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);

  H5Dclose(dataset);

  return buffer[0];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

double read_double_data (hid_t file, char *path_in_file, double *buffer) {
  /* wrapper to read double data from hdf5 file --note that buffer needs to be an array of size 1 for single entries-- */
    
  hid_t dataset = H5Dopen(file, path_in_file, H5P_DEFAULT);
  herr_t status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);

  H5Dclose(dataset);

  return buffer[0];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

void read_integer_array (char *filename, char *path_in_file, int *buffer) {
  /* wrapper to read integer array data from hdf5 file --note that buffer needs to be the array size-- */

  // open the file pointer
  
  hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  
  // read the array
  
  hid_t dataset = H5Dopen(file, path_in_file, H5P_DEFAULT);
  herr_t status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);

  H5Dclose(dataset);
  H5Fclose(file);

  return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

void read_double_array (char *filename, char *path_in_file, double *buffer) {
  /* wrapper to read double array data from hdf5 file --note that buffer needs to be the array size-- */

  // open the file pointer
  
  hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  
  // read the array
  
  hid_t dataset = H5Dopen(file, path_in_file, H5P_DEFAULT);
  herr_t status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);

  H5Dclose(dataset);
  H5Fclose(file);

  return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

void read_sim_data (char *filename, int &nsteps, int &nbeads, int &nsamp, int &nfils, int &nbpf, double &lx, double &ly, double &dt, double &density, double &kappa, double &fp, double &bl, double &sigma) {
  /* read general simulation data in hdf5 format */
  
  // open the file pointer
  
  hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

  // create buffers to get single data --note that buffer needs to be an array of size 1 for single entries--
  
  int i_buffer[1];
  i_buffer[0] = 0;
  double d_buffer[1];
  d_buffer[0] = 0.;
    
  // read in the box info

  lx = read_double_data(file, "/info/box/x", d_buffer);
  ly = read_double_data(file, "/info/box/y", d_buffer);
  
  // read in the general simulation info

  dt = read_double_data(file, "/info/dt", d_buffer);
  nsteps = read_integer_data(file, "/info/nsteps", i_buffer);
  nbeads = read_integer_data(file, "/info/nbeads", i_buffer);
  nsamp = read_integer_data(file, "/info/nsamp", i_buffer);  
  nfils = read_integer_data(file, "/info/nfils", i_buffer);
  nbpf = read_integer_data(file, "/info/nbpf", i_buffer);

  // read in the simulation parameters
  
  density = read_double_data(file, "/param/density", d_buffer);
  kappa = read_double_data(file, "/param/kappa", d_buffer);
  fp = read_double_data(file, "/param/fp", d_buffer);
  bl = read_double_data(file, "/param/bl", d_buffer);
  sigma = read_double_data(file, "/param/sigma", d_buffer);

  H5Fclose(file);
  
  // normalize some of the variables
  
  dt *= nsamp;
  
  return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

void read_single_pos_data (int step, hid_t dataset, hid_t dataspace, double *x, double *y, int natoms) {
  /* read the position data at a single timestep 
  the position data is stored in the following format:
  (nsteps, 2, nbeads)
  the data will be loaded as follows:
  (nbeads) per timestep
  */
  
  // read the data in the x direction
  
  // define the hyperslab in the dataset
  /* we are gonna reduce the data in the dataset from 3 dimensions to two 1 dimensional arrays */
    
  hsize_t offset[3];
  offset[0] = step; offset[1] = 0; offset[2] = 0;
  
  hsize_t count[3];
  count[0] = 1; count[1] = 1; count[2] = natoms;
  
  // select a 2D hyperslab from the original 3D dataset
  
  herr_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);

  // define memory dataspace
  
  hsize_t dimsm[3];	// dimensions and sizes in each dimension
  dimsm[0] = 1; dimsm[1] = 1; dimsm[2] = natoms;
  hid_t memspace = H5Screate_simple(3, dimsm, NULL);
  
  // define memory hyperslab
  
  hsize_t offset_out[3];
  offset_out[0] = 0; offset_out[1] = 0; offset_out[2] = 0;
  
  hsize_t count_out[3];
  count_out[0] = 1; count_out[1] = 1; count_out[2] = natoms;
  
  status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL, count_out, NULL);
  
  // read data from hyperslab in the file into the hyperslab in memory 
  
  status = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace, H5P_DEFAULT, x); 
      
  H5Sclose(memspace);
  
  // read the data in the y direction
  
  offset[0] = step; offset[1] = 1; offset[2] = 0;
  count[0] = 1; count[1] = 1; count[2] = natoms;
  status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);   
  memspace = H5Screate_simple(3, dimsm, NULL);
  status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL, count_out, NULL);
  status = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace, H5P_DEFAULT, y);   
  H5Sclose(memspace);
  
  return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

void write_data (int step, double *x, double *y, double *d, int N) {
  /* save the defect point data at this timestep */
    
  char step_string[30];
  sprintf(step_string, "%d", step);
  char savefilepath[200] = "/usr/users/iff_th2/duman/Defects/Output/possible_defect_pts_cpp_";
  strcat(savefilepath, step_string);
  char buffer[30] = ".txt";
  strcat(savefilepath, buffer);
    
  ofstream fl(savefilepath);
  for (int i = 0; i < N; i++) {
    if (x[i] != -20.) {
      fl << x[i] << "\t\t" << y[i] << "\t\t" << d[i] << "\n";
    }
  }
  
  fl.close();

  return;
}
    
//////////////////////////////////////////////////////////////////////////////////////////////////////////
 
void write_h5_data (char* savefolder, int step, double *x, double *y, double *d, int N, int proc_rank, int procs_size, int *counts, int *offsets, MPI_Comm comm) {
  /* save the defect point data at this timestep in hdf5 format */
  
  if (N > 0) {
    
    // generate the save path address
    
    char step_string[30];
    char buffer[30] = ".h5";
    char spath[100] = "/possible_defect_pts_cpp_";
    sprintf(step_string, "%d", step);
    char savefilepath[200];
    sprintf(savefilepath, "%s", savefolder);
    strcat(savefilepath, spath);
    strcat(savefilepath, step_string);
    strcat(savefilepath, buffer);
    
    // declare variables for data writing
    
    int rank;
    hsize_t dims;
    hid_t acc_template;
    hid_t fl;
    int ierr;
    hid_t dataspace, memspace, dataset;
    hid_t err;
    hsize_t count;
    hsize_t offset;
    
    // open the file handle
    
    acc_template = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(acc_template, comm, MPI_INFO_NULL);
    fl = H5Fcreate(savefilepath, H5F_ACC_TRUNC, H5P_DEFAULT, acc_template);
    ierr = H5Pclose(acc_template);
    
    /* write x position */
    
    // create the dataspace
    
    rank = 1;
    dims = N;
    dataspace = H5Screate_simple(rank, &dims, NULL);

    // create the dataset
    
    dataset = H5Dcreate(fl, "/xpos", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(dataspace);
    
    // figure out the offset into the dataspace for the current process
    /* each process defines dataset in memory and writes it to the hyperslab in each file */
    
    offset = offsets[proc_rank];
    count = counts[proc_rank];
    memspace = H5Screate_simple(rank, &count, NULL);
    dataspace = H5Dget_space(dataset);
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, &offset, NULL, &count, NULL);
      
    // write the data collectively
    // -- for independent access: H5FD_MPIO_INDEPENDENT
    // -- for collective access: H5FD_MPIO_COLLECTIVE
    
    acc_template = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(acc_template, H5FD_MPIO_COLLECTIVE);
    err = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace, acc_template, x);
 
    H5Pclose(acc_template);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
    
    /* write y position */
    
    dataspace = H5Screate_simple(rank, &dims, NULL);    
    dataset = H5Dcreate(fl, "/ypos", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(dataspace);
    offset = offsets[proc_rank];
    count = counts[proc_rank];
    memspace = H5Screate_simple(rank, &count, NULL);
    dataspace = H5Dget_space(dataset);
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, &offset, NULL, &count, NULL);
    acc_template = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(acc_template, H5FD_MPIO_COLLECTIVE);
    err = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace, acc_template, y);

    H5Pclose(acc_template);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
    
    /* write d defect strength */
    
    dataspace = H5Screate_simple(rank, &dims, NULL);    
    dataset = H5Dcreate(fl, "/dstr", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(dataspace);
    offset = offsets[proc_rank];
    count = counts[proc_rank];
    memspace = H5Screate_simple(rank, &count, NULL);
    dataspace = H5Dget_space(dataset);
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, &offset, NULL, &count, NULL);
    acc_template = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(acc_template, H5FD_MPIO_COLLECTIVE);
    err = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace, acc_template, d);
    
    // close the handles
    
    H5Pclose(acc_template);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
    H5Fclose(fl);
  
  }

  return;
}
    
//////////////////////////////////////////////////////////////////////////////////////////////////////////
