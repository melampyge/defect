# defect
Detection and tracking of topological defect structures. 

The idea is to identify +1/2 and -1/2 charged topological defects and then to track them in time. 

IDENTIFICATION:
 - Random points on the simulation box is chosen.
 - Within a cut radius, nematic directors around the point are calculated.
 - From the rotation of the nematic directors around the chosen points, nematic defect charge/strength is computed.
 - With a Monte Carlo algorithm with Metropolis sampling based on the found defect strength, more points are sampled around sensible points with recursion. 
 - The close points are grouped together in clusters.
 - The centroid of clusters correspond to the core of the defects.
 
TRACKING:
- A complete bipartite graph of points in succesive time frames is formed.
- Hungarian algorithm with a linear cost function is used to uniquely label the nodes in the graph. 

* The program is prototyped in Python. The time-consuming identification part is going to be ported to C++. The idea is to identify the possible defect points in C++ and then the clustering and tracking can be conducted in Python.

 
