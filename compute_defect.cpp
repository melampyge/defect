
/* compute the defect strength of a point */

//////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "compute_defect.hpp"

#define pi M_PI

using namespace Eigen;
using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////////////

void average_order_param_matrix (double *qxx, double *qxy, double *qyy, int *counter, int northants, double xd, double yd, double *x, double *y, double *phi, double *head, double *llist, double rcut, int nsegx, int nsegy, double lx, double ly) {
  /* average the elements of the order parameter matrix */
  
  double rcutsq = rcut*rcut;
  
   // find the segment (hash index) of the point
  
  int segx = static_cast<int>(xd/lx*nsegx);
  int segy = static_cast<int>(yd/ly*nsegy);
  int seg = segx*nsegy + segy;
  
  // loop over the neighboring segments to the current segment
  
  for (int a = -1; a < 2; a++) {
    int i = (segx+a+nsegx)%nsegx;
        
    for (int b = -1; b < 2; b++) {
      int j = (segy+b+nsegy)%nsegy;
      
      // get the neighboring segment id
      
      int neigh_seg = i*nsegy + j;
      int val = head[neigh_seg];
      
      while (val != -1) {
	
	double xi = x[val];
	double yi = y[val];
	
	// calculate the vector from the bead to the point
	
	double dx = xi - xd;
	dx = neigh_min(dx, lx);
	double dy = yi - yd;
	dy = neigh_min(dy, ly);
	double dsq = dx*dx + dy*dy;
	
	// if the bead is within a distance rcut of the point

	if (dsq <= rcutsq) {
	  
	  // compute the angle of the vector from the bead to the point
	  
	  double theta = atan2(dy, dx);
	  if (theta < 0) 
	    theta += 2*pi;
	  	  
	  // determine the orthant id of the bead to the point angle
	  
	  int orthid = static_cast<int>(theta/2/pi*northants);
	  
	  // calculate the bond orientation vector of the bead in line symmetry
	  
	  double bx = cos(phi[val]);
	  double by = sin(phi[val]);
	  
	  // average the order parameter matrix element by element
	  
	  qxx[orthid] += 2*bx*bx - 1;
	  qxy[orthid] += 2*bx*by;
	  qyy[orthid] += 2*by*by - 1;
	  counter[orthid] += 1;
	}
	
	// get the next item from the linked list of the neighboring segment
	
	val = llist[val];
	
      }
    }
  }
      
  // take the average of the order parameter matrix elements
  
  for (int i = 0; i < northants; i++) {
    if (counter[i] > 0) {
      qxx[i] /= counter[i];
      qxy[i] /= counter[i];
      qyy[i] /= counter[i];
    }
  }
  
  return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

void calc_nematic_directors (double *directors, double *qxx, double *qxy, double *qyy, int northants) {
  /* calculate the nematic director inside each orthant */
 
  // run over the orthants
  
  for (int i = 0; i < northants; i++) {
    
    // construct the order parameter matrix
    
    Matrix2d q = Matrix2d::Zero();
    q << qxx[i], qxy[i],
	 qxy[i], qyy[i];
	 
    // calculate the eigenvalues and eigenvectors of the order parameter matrix
    
    SelfAdjointEigenSolver<Matrix2d> eigensolver(q);
    
    // choose the eigenvectors of the larger eigenvalue
    
    if ( eigensolver.eigenvalues()(0) > eigensolver.eigenvalues()(1) ) {
      directors[i] = atan2( eigensolver.eigenvectors()(1), eigensolver.eigenvectors()(0) );
    }
    else {
      directors[i] = atan2( eigensolver.eigenvectors()(3), eigensolver.eigenvectors()(2) );      
    }
	
  } 
  
  // correct nematic directors by assumming no jumps larger than |pi/2| to ensure line symmetry
  
  for (int i = 1; i < northants; i++) {
    double dif = directors[i] - directors[i-1];
    
    if (dif < -pi/2) 
      dif += pi;
    if (dif > pi/2) 
      dif -= pi;
    
    directors[i] = directors[i-1] + dif;
  }
  
  return;
}
  
//////////////////////////////////////////////////////////////////////////////////////////////////////////

double compute_single_defect (double xd, double yd, double *x, double *y, double *phi, double *head, double *llist, double rcut, int nsegx, int nsegy, double lx, double ly) {
  /* compute the defect strength of a point */
  
  // allocate arrays to divide the full circle into orthants 
  
  int northants = 10;
  double *qxx = new double[northants];
  double *qxy = new double[northants];
  double *qyy = new double[northants];
  int *counter = new int[northants];
  double *directors = new double[northants];
  for (int i = 0; i < northants; i++) {
    qxx[i] = 0.;  qxy[i] = 0.;  qyy[i] = 0.;  counter[i] = 0;  directors[i] = 0.;
  }
  
  // calculate and average the order parameter matrix per orthant 
  
  average_order_param_matrix(qxx, qxy, qyy, counter, northants, xd, yd, x, y, phi, head, llist, rcut, nsegx, nsegy, lx, ly);
  
  // calculate the nematic directors per orthant
  
  calc_nematic_directors(directors, qxx, qxy, qyy, northants);
  
  // calculate the defect strength
  
  double dst = (directors[northants-1] - directors[0])/2./pi;
    
  return dst; 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

void search_neighborhood (double xd, double yd, double dmax, double dlabel, double *xdefects, double *ydefects, double *ddefects, int &cnt_defects, double *x, double *y, double *phi, double *head, double *llist, double rcut, int nsegx, int nsegy, double rn, int nn, int nr, double dcut, double lx, double ly, MTRand randi, int cnt_recursion) {
  /* search the neighborhood of the main point to compute defects
  rn: inner radius of the shell
  nn: number of neighbors to pop up
  nr: number of recursions to conduct */
  
  // increase the number of recursions conducted
  
  if (cnt_recursion == nr) {
    return;
  }
  cnt_recursion++;
  
  // pop up neighbor points within the neighborhood
  
  for (int i = 0; i < nn; i++) {
    
    bool point_is_defect = false;
    
    // generate a random point within the circular shell
    
    double angle = randi.rand()*2.*pi;
    double radius = sqrt(randi.rand()*(rcut-rn));
    double xn = xd + radius*cos(angle);
    double yn = yd + radius*sin(angle);
       
    // compute the defect strength of the point
    
    double dst = compute_single_defect(xn, yn, x, y, phi, head, llist, rcut, nsegx, nsegy, lx, ly);
    
    // check for -1/2 defects and +1/2 defects with Metropolis Monte Carlo sampling, that is, only accept the point if it is better in terms of defect strength than the parent point
    
    if (dlabel == 0.5 && abs(dmax-0.5) > abs(dst-0.5)) {
      point_is_defect = true;
      dlabel = 0.5;
    }
    else if (dlabel == -0.5 && abs(dmax+0.5) > abs(dst+0.5)) {
      point_is_defect = true;
      dlabel = -0.5;
    }

    // register the point into the array if the point is a defect and start recursion loop
    
    if (point_is_defect) {

      // save the point as defect
      
      cnt_defects++;				// increase the index of the current defect point in the global list of defects
      xdefects[cnt_defects] = xn;		// save the x coordinate of the defect point
      ydefects[cnt_defects] = yn;		// save the y coordinate of the defect point
      ddefects[cnt_defects] = dst;		// save the defect strength of the point
      
      // search the neighborhood of the point
      
      search_neighborhood(xn, yn, dst, dlabel, xdefects, ydefects, ddefects, cnt_defects, x, y, phi, head, llist, rcut, nsegx, nsegy, rn, nn, nr, dcut, lx, ly, randi, cnt_recursion);
    }    
  }
    
  return;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////

void search_main_pts (double *xdefects, double *ydefects, double *ddefects, int *check_again, int &cnt_defects, double *xpoints, double *ypoints, int npoints, double *x, double *y, double *phi, double *head, double *llist, double rcut, int nsegx, int nsegy, double rn, int nn, int nr, double dcut, double lx, double ly, MTRand randi) {
  /* search the main points to compute defects */
  
  for (int i = 0; i < npoints; i++) {

    bool point_is_defect = false;

    // generate a new main point that is random if the point doesnt need to be checked again
    
    if (check_again[i] == 0) {
      xpoints[i] = randi.rand()*lx;
      ypoints[i] = randi.rand()*ly;
    }
    
    // compute the defect strength of the point
    
    double dst = compute_single_defect(xpoints[i], ypoints[i], x, y, phi, head, llist, rcut, nsegx, nsegy, lx, ly);
    
    // check for -1/2 defects and +1/2 defects, else regenerate a random point in the next time step
    
    double dlabel = 0.;
    if (dst > -dcut-0.5 && dst < dcut-0.5) {
      point_is_defect = true;
      dlabel = -0.5;
    }
    else if (dst > 0.5-dcut && dst < dcut+0.5) {
      point_is_defect = true;
      dlabel = 0.5;
    }
    else {
      check_again[i] = 0;
    }
    
    // register the point into the array if the point is a defect and start recursion loop
    
    int cnt_recursion = 0;
    if (point_is_defect) {

      // save the point as defect
      
      check_again[i] = 1;			// ensure to check the point again in the next time frame as well
      cnt_defects++;				// increase the index of the current defect point in the global list of defects
      xdefects[cnt_defects] = xpoints[i];	// save the x coordinate of the defect point
      ydefects[cnt_defects] = ypoints[i];	// save the y coordinate of the defect point
      ddefects[cnt_defects] = dst;		// save the defect strength of the point
      
      // search the neighborhood of the point
      
      search_neighborhood(xpoints[i], ypoints[i], dst, dlabel, xdefects, ydefects, ddefects, cnt_defects, x, y, phi, head, llist, rcut, nsegx, nsegy, rn, nn, nr, dcut, lx, ly, randi, cnt_recursion);
    }
  
  }
    
  return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
