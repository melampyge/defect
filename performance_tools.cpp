
//////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cmath>

//////////////////////////////////////////////////////////////////////////////////////

using namespace std;

class Performance_tools {
  public:

  void fill_neigh_matrix(int* neighs, int* llist, int* head,
			 int nsegx, int nsegy, double* x, double* y,
			 int npoints, double lx, double ly,
			 double dcrit) {
    /* generate a neighborhood matrix to disseminate information about neighbors of data points */
   
    double dcrit2 = dcrit*dcrit;
      
    // loop over all central cells
    
    for (int i = 0; i < nsegx; i++) {
      
      for (int j = 0; j < nsegy; j++) {
	
        // get the head point of the current cell
	
        int sv1 = head[i*nsegy + j];
	
        // loop over neighboring cells around the head cell
	
        for (int a = 0; a < 3; a++) {
          int i2 = (i-1+a+nsegx)%nsegx;
	  
          for (int b = 0; b < 3; b++) {
            int j2 = (j-1+b+nsegy)%nsegy;
	    
            // get the head point of the neighbor cell
	    
            int sv2 = head[i2*nsegy + j2];
                    
            // restore particle ids in the cells
	    
            int val1 = sv1;
            int val2 = sv2;
	    
            // walk along the linked list in the head cell

	    while (val1 != -1) {

	      double x1 = x[val1]/lx;
              double y1 = y[val1]/ly;
	      
	          // walk along the linked list in the neighbor cell

              while (val2 != -1) {
                  if (val1 != val2) {
                    double x2 = x[val2]/lx;
                    double y2 = y[val2]/ly;

                    double dx = x2-x1;
                    dx = dx - floor(dx + 0.5);
                    dx = dx*lx;
                    
                    double dy = y2-y1;
                    dy = dy - floor(dy + 0.5);
                    dy = dy*ly;
                    
                    double rsq = dx*dx + dy*dy;

                    if (rsq < dcrit2) {
                      neighs[val1*npoints + val2] += 1;
                      neighs[val2*npoints + val1] += 1;
                    }
                  }
                val2 = llist[val2];
              } // neighbor cell linked list
	      
              val1 = llist[val1];
              val2 = sv2;
            } // head cell linked list
          } // neighbor cell index y
        } // neighbor cell index x
      } // head cell index y
    } // head cell index x
  } 

  /*****************************************************************************************/

  void recursion(int* neighs, int* cl, int i, int npoints) {
    /* walk along the neighbors of a point to identify the points as clusters */
    
    for (int j = 0; j < npoints; j++) {
      if (cl[j] == -1) {
        if (neighs[i*npoints + j] > 0) {
          cl[j] = cl[i];
          recursion(neighs, cl, j, npoints);
        }
      }
    }
  }

  void cluster_search(int* neighs, int* cl, int npoints) {
    /* search for clusters */
    
    // loop over all the points
    
    int clmax = 0;
    for (int i = 0; i < npoints; i++) {
      
      // assign a cluster id to the current point
      
      if (cl[i] == -1) {
        cl[i] = clmax;
        clmax++;
        recursion(neighs, cl, i, npoints);
      }
    }
  }
  
};

/*******************************************************************************************/
/*******************************************************************************************/
/*******************************************************************************************/

extern "C" {
  Performance_tools* Performance_tools_new() { return new Performance_tools(); }

  void fill_neigh_matrix(Performance_tools* performance_tools,
			 int* neighs, int* llist, int* head,
			 int nsegx, int nsegy, double* x, double* y,
			 int npoints, double lx, double ly,
			 double dcrit) {

    performance_tools->fill_neigh_matrix(neighs, llist, head,
					 nsegx, nsegy, x, y,
					 npoints, lx, ly,
					 dcrit);
					 
  }

  /*****************************************************************************************/

  void cluster_search(Performance_tools* performance_tools,
		      int* neighs, int* cl, int npoints) {
    
    performance_tools->cluster_search(neighs, cl, npoints);
    
  }
  
}

//////////////////////////////////////////////////////////////////////////////////////

