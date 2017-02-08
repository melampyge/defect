#!/bin/bash

mpicxx -Wl,-rpath=${HOME}/hdf5_parallel/lib -L${HOME}/hdf5_parallel/lib -I${HOME}/hdf5_parallel/include -I${HOME}/Eigen -lhdf5 find_defects_multi.cpp compute_defect.cpp misc_tools.cpp read_write.cpp -o fdef_multi
mpirun -np ${1} /usr/users/iff_th2/duman/Defects/Scripts/Defects/fdef_multi /usr/users/iff_th2/duman/Defects/Output/outcpp.h5 /usr/users/iff_th2/duman/Defects/Output
#mpirun -np 8 valgrind --leak-check=full --track-origins=yes /usr/users/iff_th2/duman/Defects/Scripts/Defects/fdef_multi /usr/users/iff_th2/duman/Defects/Output/outcpp.h5 /usr/users/iff_th2/duman/Defects/Output

