
/* miscellanous tools */

//////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "misc_tools.hpp"

#define pi M_PI

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////////////

double neigh_min (double dx, double lx) {
  /* find the minimum image distance between two particles inside the central box */
  
  double dx1 = dx + lx;
  double dx2 = dx - lx;
  if (dx*dx < dx1*dx1 && dx*dx < dx2*dx2) return dx;
  if (dx1*dx1 < dx2*dx2) return dx1;
  return dx2;
  
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

void gen_linked_list (double *llist, double *head, double rcut, int nsegx, int nsegy, 
		      int nseg, int nbeads, double *x, double *y, double lx, double ly) {
  /* generate a hashed linked list to browse data points via positions */
  
  for (int i = 0; i < nbeads; i++) {
    
    int segx = static_cast<int>(x[i]/lx*nsegx);
    segx = (segx+nsegx) % nsegx;
    int segy = static_cast<int>(y[i]/ly*nsegy);
    segy = (segy+nsegy) % nsegy;
    int seg = segx*nsegy + segy;
    llist[i] = head[seg];
    head[seg] = i;
    
  }
  
  return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

void calc_headless_orient(double *phi, double *x, double *y, double lx, double ly, int nfils, int nbpf) {
  /* calculate bead orientations with line symmetry through bond angles between successive beads of filaments */
  
  double dx, dy;
  int k = 0;	// bead counter
  
  for (int i = 0; i < nfils; i++) {
    for (int j = 0; j < nbpf; j++) {
      
      // calculate the displacements
      
      if (j == 0) { 
	dx = x[k+1] - x[k];
	dy = y[k+1] - y[k];
      }
      else if (j == nbpf-1) {
	dx = x[k] - x[k-1];
	dy = y[k] - y[k-1];
      }
      else {
	dx = x[k+1] - x[k-1];
	dy = y[k+1] - y[k-1];
      }
      
      // correct for periodic boundary conditions
      
      dx = neigh_min(dx, lx);
      dy = neigh_min(dy, ly);
      
      // calculate the angle and turn it headless
      
      phi[k] = atan2(dy, dx);	// in radians
      if (phi[k] < 0) 
	phi[k] += pi;
      
      k++;	// increment the bead counter
    }
  }
  
  return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
