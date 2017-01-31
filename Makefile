
GCC=g++
GFLAGS=-Wl,-rpath=${HOME}/hdf5/lib -L${HOME}/hdf5/lib -I${HOME}/hdf5/include -I${HOME}/Eigen -lhdf5 -fopenmp	

objects = compute_defect.o read_write.o misc_tools.o find_defects.o
fdef : $(objects)
	$(GCC) $(GFLAGS) -o fdef $(objects)

fdef.o : find_defects.cpp compute_defect.hpp read_write.hpp misc_tools.hpp MersenneTwister.h 
	$(GCC) $(GFLAGS) find_defects.cpp

compdef.o : compute_defect.cpp compute_defect.hpp misc_tools.hpp
	${GCC} ${GFLAGS} compute_defect.cpp 

.PHONY: cleanall cleanobj cleanprogram

cleanall :
	rm fdef $(objects)

cleanobj : 
	rm $(objects)

cleanprogram : 
	rm fdef

