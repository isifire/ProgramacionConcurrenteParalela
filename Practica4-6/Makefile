cflags=-Xcompiler -O3,-Wall,-fPIC,-fopenmp
libs=-lcublas -lcudart -fopenmp

all: cleanall DCTGPU

DCTGPU.o: DCTGPU.cu
	nvcc $(cflags) -c DCTGPU.cu -o DCTGPU.o 

DCTGPU: DCTGPU.o
	g++ -shared -Wl,-soname,DCTGPU.so -o DCTGPU.so DCTGPU.o -L$(CUDADIR)/lib64 $(libs)
	rm -f DCTGPU.o

clean:
	rm -f *~ *.o core

cleanall: clean
	rm -f DCTGPU.so DCTGPU.o

