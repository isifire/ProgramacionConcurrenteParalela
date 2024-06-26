#ifndef PRAC06
#define PRAC06

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>
#include <cublas_v2.h>

#define CHECKNULL(x) do { if((x)==NULL) { \
   printf("Error reservando memoria en %s linea %d\n", __FILE__, __LINE__);\
   exit(EXIT_FAILURE);}} while(0)

#define CUDAERR(x) do { if((x)!=cudaSuccess) { \
   printf("CUDA error: %s : %s, line %d\n", cudaGetErrorString(x), __FILE__, __LINE__);\
   exit(EXIT_FAILURE);}} while(0)

#define CUBLASERR(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
   printf("CUBLAS error: %s, line %d\n", __FILE__, __LINE__);\
   exit(EXIT_FAILURE);}} while(0)


#define CHECKLASTERR()  __getLastCudaError(__FILE__, __LINE__)
inline void __getLastCudaError(const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i): getLastCudaError() CUDA error: (%d) %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


#define pi       3.141592
#define MaskSize 8


extern "C" void TriggerDCT(double *, double *, const unsigned int, const unsigned int);
extern "C" void TriggerIDCT(double *, double *, const unsigned int, const unsigned int);

extern "C" void FPB(double *, const unsigned int, const unsigned int, const unsigned int);
extern "C" void FPA(double *, const unsigned int, const unsigned int, const unsigned int);

#endif
