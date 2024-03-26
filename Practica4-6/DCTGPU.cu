
#include "Prototipos_GPU.h"

__constant__ int zzgpu[][2]={
    {0, 0},
    {0, 1}, {1, 0},
    {2, 0}, {1, 1}, {0, 2},
    {0, 3}, {1, 2}, {2, 1}, {3, 0},
    {4, 0}, {3, 1}, {2, 2}, {1, 3}, {0, 4},
    {0, 5}, {1, 4}, {2, 3}, {3, 2}, {4, 1}, {5, 0},
    {6, 0}, {5, 1}, {4, 2}, {3, 3}, {2, 4}, {1, 5}, {0, 6},
    {0, 7}, {1, 6}, {2, 5}, {3, 4}, {4, 3}, {5, 2}, {6, 1}, {7, 0},
    {7, 1}, {6, 2}, {5, 3}, {4, 4}, {3, 5}, {2, 6}, {1, 7},
    {2, 7}, {3, 6}, {4, 5}, {5, 4}, {6, 3}, {7, 2},
    {7, 3}, {6, 4}, {5, 5}, {4, 6}, {3, 7},
    {4, 7}, {5, 6}, {6, 5}, {7, 4},
    {7, 5}, {6, 6}, {5, 7},
    {6, 7}, {7, 6},
    {7, 7}
};


// PLANTEAMIENTO BASE --------------------------------------------------------------------------------
__global__  void kernel_DCT8x8(double *ORIG, double *DST, const unsigned int DimX, const unsigned int DimY)
{
   int u = blockIdx.x * blockDim.x + threadIdx.x;
   int v = blockIdx.y * blockDim.y + threadIdx.y;

   if(u < DimX && v < DimY){
      double sum = 0.0;
         for(int x=0; x<MaskSize; x++){
            for(int y=0; y<MaskSize; y++){
               double cos_val1 = cos(((2*x+1)*threadIdx.x*pi)/(2*MaskSize));
               double cos_val2 = cos(((2*y+1)*threadIdx.y*pi)/(2*MaskSize));
               double Csu, Csv;
               if(threadIdx.x==0){
                  Csu = 0.5/sqrt(2.0);
               }
               else{
                  Csu = 0.5;
               }
               if(threadIdx.y==0){
                  Csv = 0.5/sqrt(2.0);
               }
               else{
                  Csv = 0.5;
               }
               sum += Csu * Csv * ORIG[(x+blockIdx.x*blockDim.x)*DimY+(y+blockIdx.y*blockDim.y)] * cos_val1 * cos_val2;
            }
         }
      DST[u*DimY+v] = sum;     
   }
}

__global__ void kernel_IDCT8x8(double *ORIG, double *DST, const unsigned int DimX, const unsigned int DimY)
{
 	
   int u = blockIdx.x * blockDim.x + threadIdx.x;
   int v = blockIdx.y * blockDim.y + threadIdx.y;

   if(u < DimX && v < DimY){
      double sum = 0.0;
         for(int x=0; x<MaskSize; x++){
            for(int y=0; y<MaskSize; y++){
               double cos_val1 = cos(((2*threadIdx.x+1)*x*pi)/(2*MaskSize));
               double cos_val2 = cos(((2*threadIdx.y+1)*y*pi)/(2*MaskSize));
               double Csx, Csy;
               if(x==0){
                  Csx = 0.5/sqrt(2.0);
               }
               else{
                  Csx = 0.5;
               }
               if(y==0){
                  Csy = 0.5/sqrt(2.0);
               }
               else{
                  Csy = 0.5;
               }
               sum += Csx * Csy * ORIG[(x+blockIdx.x*blockDim.x)*DimY+(y+blockIdx.y*blockDim.y)] * cos_val1 * cos_val2;
            }
         }
      DST[u*DimY+v] = round(sum);     
   }

}

__global__ void kernel_FPB(double *Imag, const int n, const unsigned int DimX, const unsigned int DimY)
{
	
   if(threadIdx.x==0 && threadIdx.y==0){
      for(int z=0; z<n; z++){
         Imag[(zzgpu[63-z][0]+blockDim.x*blockIdx.x)*DimY+(zzgpu[63-z][1]+blockDim.y*blockIdx.y)] = 0.0;
      }
   }

}

__global__ void kernel_FPA(double *Imag, const int n, const unsigned int DimX, const unsigned int DimY)
{

   if(threadIdx.x==0 && threadIdx.y==0){

      for(int z=0; z<n; z++){
         Imag[(zzgpu[z][0]+blockDim.x*blockIdx.x)*DimY+(zzgpu[z][1]+blockDim.y*blockIdx.y)] = 0.0;
      }
   }
}


extern "C" void TriggerDCT(double *ORIG, double *DST, const unsigned int DimX, const unsigned int DimY){
   int    NumGPUs;
   cudaError_t e=cudaGetDeviceCount(&NumGPUs);
   if (NumGPUs <=0||e!=0) 
      printf("no hay GPU\n");
   else
   {
      double *ORIG_GPU, *DST_GPU;
      int size = DimX*DimY*sizeof(double);

      // !!!!! Host es el python (ORIG y DST) y Device es la gpu (ORIG_GPU y DST_GPU)

      // Reservamos memoria para las matrices origen y destino en GPU
      cudaMalloc((void **) &ORIG_GPU, size);
      cudaMalloc((void **) &DST_GPU, size);

      // Copiamos la matriz origen en ORIG_GPU (el espacio reservado para ella en la GPU)
      // De momento no hace falta copiar DST ya que esta vacia
      cudaMemcpy(ORIG_GPU, ORIG, size ,cudaMemcpyHostToDevice);

      // Llamada al kernel
      dim3 TPBLK(8, 8);
      dim3 NBLCK(DimX/8,DimY/8);
      kernel_DCT8x8<<<NBLCK, TPBLK>>>(ORIG_GPU, DST_GPU, DimX, DimY);
   
      // Copiamos la imagen DST_GPU generada por la GPU en nuestra DST
      cudaMemcpy(DST, DST_GPU, size, cudaMemcpyDeviceToHost);
   
      // Liberamos las memoria en GPU
      cudaFree(ORIG_GPU);
      cudaFree(DST_GPU);

   }

}

extern "C" void TriggerIDCT(double *ORIG, double *DST, const unsigned int DimX, const unsigned int DimY){
   int    NumGPUs;
   cudaError_t e=cudaGetDeviceCount(&NumGPUs);
  if (NumGPUs <=0||e!=0) 
      printf("no hay GPU\n");
   else
   {
      double *ORIG_GPU, *DST_GPU;
      int size = DimX*DimY*sizeof(double);

      // !!!!! Host es el python (ORIG y DST) y Device es la gpu (ORIG_GPU y DST_GPU)

      // Reservamos memoria para las matrices origen y destino en GPU
      cudaMalloc((void **) &ORIG_GPU, size);
      cudaMalloc((void **) &DST_GPU, size);

      // Copiamos la matriz origen en ORIG_GPU (el espacio reservado para ella en la GPU)
      // De momento no hace falta copiar DST ya que esta vacia
      cudaMemcpy(ORIG_GPU, ORIG, size ,cudaMemcpyHostToDevice);

      // Llamada al kernel
      dim3 TPBLK(8, 8);
      dim3 NBLCK(DimX/8,DimY/8);
      kernel_IDCT8x8<<<NBLCK, TPBLK>>>(ORIG_GPU, DST_GPU, DimX, DimY);
   
      // Copiamos la imagen DST_GPU generada por la GPU en nuestra DST
      cudaMemcpy(DST, DST_GPU, size, cudaMemcpyDeviceToHost);
   
      // Liberamos las memoria en GPU
      cudaFree(ORIG_GPU);
      cudaFree(DST_GPU);
   }
}


extern "C" void FPB(double *Im, const unsigned int n, const unsigned int DimX, const unsigned int DimY){
   int    NumGPUs;
   cudaError_t e=cudaGetDeviceCount(&NumGPUs);
  if (NumGPUs <=0||e!=0) 
      printf("no hay GPU\n");
   else
   {
      double *ImagGPU;
      int size = DimX*DimY*sizeof(double);

      cudaMalloc((void **) &ImagGPU, size);

      cudaMemcpy(ImagGPU, Im, size ,cudaMemcpyHostToDevice);

      dim3 TPBLK(8, 8);
      dim3 NBLCK(DimX/8,DimY/8);
      kernel_FPB<<<NBLCK, TPBLK>>>(ImagGPU, n, DimX, DimY);

      cudaMemcpy(Im, ImagGPU, size, cudaMemcpyDeviceToHost);

      cudaFree(ImagGPU);
   }
}

extern "C" void FPA(double *Im, const unsigned int n, const unsigned int DimX, const unsigned int DimY){
   int    NumGPUs;
   cudaError_t e=cudaGetDeviceCount(&NumGPUs);
  if (NumGPUs <=0||e!=0) 
      printf("no hay GPU\n");
   else
   {
      double *ImagGPU;
      int size = DimX*DimY*sizeof(double);

      cudaMalloc((void **) &ImagGPU, size);

      cudaMemcpy(ImagGPU, Im, size ,cudaMemcpyHostToDevice);

      dim3 TPBLK(8, 8);
      dim3 NBLCK(DimX/8,DimY/8);
      kernel_FPA<<<NBLCK, TPBLK>>>(ImagGPU, n, DimX, DimY);

      cudaMemcpy(Im, ImagGPU, size, cudaMemcpyDeviceToHost);

      cudaFree(ImagGPU);
   }
}


// EXTRA : Memoria pinned -----------------------------------------------------------------------------------

extern "C" void TriggerDCT_pinned(double *ORIG, double *DST, const unsigned int DimX, const unsigned int DimY){
   int    NumGPUs;
   cudaError_t e=cudaGetDeviceCount(&NumGPUs);
   if (NumGPUs <=0||e!=0) 
      printf("no hay GPU\n");
   else
   {
      double *ORIG_GPU, *DST_GPU;
      double *m1, *m2;
      int size = DimX*DimY*sizeof(double);

      // Reservamos memoria en la RAM del Host (CPU), guardando la imagen ORIG
      cudaHostAlloc((void **)&ORIG_GPU, size, cudaHostAllocMapped);
      cudaHostAlloc((void **)&DST_GPU, size, cudaHostAllocMapped);

      // Accedenis a ORIG mediante el puntero
      cudaHostGetDevicePointer((void **)&m1,(void *)ORIG_GPU, 0);
      cudaHostGetDevicePointer((void **)&m2,(void *)DST_GPU, 0);
      
      // Copiamos la imagen ORIG en ORIG_GPU
      cudaMemcpy(ORIG_GPU, ORIG, size ,cudaMemcpyHostToHost);

      // Llamada al kernel
      dim3 TPBLK(8, 8);
      dim3 NBLCK(DimX/8,DimY/8);
      kernel_DCT8x8<<<NBLCK, TPBLK>>>(m1, m2, DimX, DimY);

      cudaMemcpy(DST, DST_GPU, size, cudaMemcpyHostToHost);

      cudaFreeHost(ORIG_GPU);
      cudaFreeHost(DST_GPU);

   }

}

extern "C" void TriggerIDCT_pinned(double *ORIG, double *DST, const unsigned int DimX, const unsigned int DimY){
   int    NumGPUs;
   cudaError_t e=cudaGetDeviceCount(&NumGPUs);
  if (NumGPUs <=0||e!=0) 
      printf("no hay GPU\n");
   else
   {
      double *ORIG_GPU, *DST_GPU;
      double *m1, *m2;
      int size = DimX*DimY*sizeof(double);

      // Reservamos memoria en la RAM del Host (CPU), guardando la imagen ORIG
      cudaHostAlloc((void **)&ORIG_GPU, size, cudaHostAllocMapped);
      cudaHostAlloc((void **)&DST_GPU, size, cudaHostAllocMapped);

      // Accedenis a ORIG mediante el puntero
      cudaHostGetDevicePointer((void **)&m1,(void *)ORIG_GPU, 0);
      cudaHostGetDevicePointer((void **)&m2,(void *)DST_GPU, 0);
      
      // Copiamos la imagen ORIG en ORIG_GPU
      cudaMemcpy(ORIG_GPU, ORIG, size ,cudaMemcpyHostToHost);

      // Llamada al kernel
      dim3 TPBLK(8, 8);
      dim3 NBLCK(DimX/8,DimY/8);
      kernel_IDCT8x8<<<NBLCK, TPBLK>>>(m1, m2, DimX, DimY);

      cudaMemcpy(DST, DST_GPU, size, cudaMemcpyHostToHost);

      cudaFreeHost(ORIG_GPU);
      cudaFreeHost(DST_GPU);
   }
}

extern "C" void FPB_pinned(double *Im, const unsigned int n, const unsigned int DimX, const unsigned int DimY){
   int    NumGPUs;
   cudaError_t e=cudaGetDeviceCount(&NumGPUs);
  if (NumGPUs <=0||e!=0) 
      printf("no hay GPU\n");
   else
   {
      double *ImagGPU;
      double *m;
      int size = DimX*DimY*sizeof(double);

      cudaHostAlloc((void **)&ImagGPU, size, cudaHostAllocMapped);

      cudaHostGetDevicePointer((void **)&m,(void *)ImagGPU, 0);

      cudaMemcpy(ImagGPU, Im, size ,cudaMemcpyHostToHost);

      dim3 TPBLK(8, 8);
      dim3 NBLCK(DimX/8,DimY/8);
      kernel_FPB<<<NBLCK, TPBLK>>>(m, n, DimX, DimY);

      cudaMemcpy(Im, ImagGPU, size, cudaMemcpyHostToHost);

      cudaFreeHost(ImagGPU);
   }
}

extern "C" void FPA_pinned(double *Im, const unsigned int n, const unsigned int DimX, const unsigned int DimY){
   int    NumGPUs;
   cudaError_t e=cudaGetDeviceCount(&NumGPUs);
  if (NumGPUs <=0||e!=0) 
      printf("no hay GPU\n");
   else
   {
      double *ImagGPU;
      double *m;
      int size = DimX*DimY*sizeof(double);

      cudaHostAlloc((void **)&ImagGPU, size, cudaHostAllocMapped);

      cudaHostGetDevicePointer((void **)&m,(void *)ImagGPU, 0);

      cudaMemcpy(ImagGPU, Im, size ,cudaMemcpyHostToHost);

      dim3 TPBLK(8, 8);
      dim3 NBLCK(DimX/8,DimY/8);
      kernel_FPA<<<NBLCK, TPBLK>>>(m, n, DimX, DimY);

      cudaMemcpy(Im, ImagGPU, size, cudaMemcpyHostToHost);

      cudaFreeHost(ImagGPU);
   }
}

// EXTRA : Memoria unified ----------------------------------------------------------------------------------

extern "C" void TriggerDCT_unified(double *ORIG, double *DST, const unsigned int DimX, const unsigned int DimY){
   int    NumGPUs;
   cudaError_t e=cudaGetDeviceCount(&NumGPUs);
   if (NumGPUs <=0||e!=0) 
      printf("no hay GPU\n");
   else
   {
      double *ORIG_GPU, *DST_GPU;
      int size = DimX*DimY*sizeof(double);

      // Reservamos memoria para las matrices origen y destino en GPU
      cudaMallocManaged((void **) &ORIG_GPU, size);
      cudaMallocManaged((void **) &DST_GPU, size);

      // Copiamos la matriz origen en ORIG_GPU (el espacio reservado para ella en la GPU)
      // De momento no hace falta copiar DST ya que esta vacia
      cudaMemcpy(ORIG_GPU, ORIG, size ,cudaMemcpyHostToHost);

      // Llamada al kernel
      dim3 TPBLK(8, 8);
      dim3 NBLCK(DimX/8,DimY/8);
      kernel_DCT8x8<<<NBLCK, TPBLK>>>(ORIG_GPU, DST_GPU, DimX, DimY);
   
      // Copiamos la imagen DST_GPU generada por la GPU en nuestra DST
      cudaMemcpy(DST, DST_GPU, size, cudaMemcpyHostToHost);
   
      // Liberamos las memoria en GPU
      cudaFree(ORIG_GPU);
      cudaFree(DST_GPU);

   }

}

extern "C" void TriggerIDCT_unified(double *ORIG, double *DST, const unsigned int DimX, const unsigned int DimY){
   int    NumGPUs;
   cudaError_t e=cudaGetDeviceCount(&NumGPUs);
  if (NumGPUs <=0||e!=0) 
      printf("no hay GPU\n");
   else
   {
      double *ORIG_GPU, *DST_GPU;
      int size = DimX*DimY*sizeof(double);

      // Reservamos memoria para las matrices origen y destino en GPU
      cudaMallocManaged((void **) &ORIG_GPU, size);
      cudaMallocManaged((void **) &DST_GPU, size);

      // Copiamos la matriz origen en ORIG_GPU (el espacio reservado para ella en la GPU)
      // De momento no hace falta copiar DST ya que esta vacia
      cudaMemcpy(ORIG_GPU, ORIG, size ,cudaMemcpyHostToHost);

      // Llamada al kernel
      dim3 TPBLK(8, 8);
      dim3 NBLCK(DimX/8,DimY/8);
      kernel_IDCT8x8<<<NBLCK, TPBLK>>>(ORIG_GPU, DST_GPU, DimX, DimY);
   
      // Copiamos la imagen DST_GPU generada por la GPU en nuestra DST
      cudaMemcpy(DST, DST_GPU, size, cudaMemcpyHostToHost);
   
      // Liberamos las memoria en GPU
      cudaFree(ORIG_GPU);
      cudaFree(DST_GPU);
   }
}

extern "C" void FPB_unified(double *Im, const unsigned int n, const unsigned int DimX, const unsigned int DimY){
   int    NumGPUs;
   cudaError_t e=cudaGetDeviceCount(&NumGPUs);
  if (NumGPUs <=0||e!=0) 
      printf("no hay GPU\n");
   else
   {
      double *ImagGPU;
      int size = DimX*DimY*sizeof(double);

      cudaMallocManaged((void **) &ImagGPU, size);

      cudaMemcpy(ImagGPU, Im, size ,cudaMemcpyHostToHost);

      dim3 TPBLK(8, 8);
      dim3 NBLCK(DimX/8,DimY/8);
      kernel_FPB<<<NBLCK, TPBLK>>>(ImagGPU, n, DimX, DimY);

      cudaMemcpy(Im, ImagGPU, size, cudaMemcpyHostToHost);

      cudaFree(ImagGPU);
   }
}

extern "C" void FPA_unified(double *Im, const unsigned int n, const unsigned int DimX, const unsigned int DimY){
   int    NumGPUs;
   cudaError_t e=cudaGetDeviceCount(&NumGPUs);
  if (NumGPUs <=0||e!=0) 
      printf("no hay GPU\n");
   else
   {
      double *ImagGPU;
      int size = DimX*DimY*sizeof(double);

      cudaMallocManaged((void **) &ImagGPU, size);

      cudaMemcpy(ImagGPU, Im, size ,cudaMemcpyHostToHost);

      dim3 TPBLK(8, 8);
      dim3 NBLCK(DimX/8,DimY/8);
      kernel_FPA<<<NBLCK, TPBLK>>>(ImagGPU, n, DimX, DimY);

      cudaMemcpy(Im, ImagGPU, size, cudaMemcpyHostToHost);

      cudaFree(ImagGPU);
   }
}


// EXTRA : Version 1D --------------------------------------------------------------------------------------------------------
__global__  void kernel_DCT8x8_1D(double *ORIG, double *DST, const unsigned int DimX, const unsigned int DimY)
{
   int id = blockIdx.x * blockDim.x + threadIdx.x;

   if(id < DimX*DimY){

      int u = id/DimY;
      int v = id%DimY;
	
      int x1 = threadIdx.x/MaskSize;
      int y1 = threadIdx.x%MaskSize;

      double sum = 0.0;
         for(int x=0; x<MaskSize; x++){
            for(int y=0; y<MaskSize; y++){
               double cos_val1 = cos(((2*x+1)*x1*pi)/(2*MaskSize));
               double cos_val2 = cos(((2*y+1)*y1*pi)/(2*MaskSize));
               double Csu, Csv;
               if(x1==0){
                  Csu = 0.5/sqrt(2.0);
               }
               else{
                  Csu = 0.5;
               }
	            if(y1==0){
                  Csv = 0.5/sqrt(2.0);
               }
               else{
                  Csv = 0.5;
               }
              
               sum += Csu * Csv * ORIG[(x+blockIdx.x*blockDim.x)*DimY+(y+blockIdx.x*blockDim.x)] * cos_val1 * cos_val2;
            }
         }
      DST[u*DimY+v] = sum;     
   }
}


extern "C" void TriggerDCT_1D(double *ORIG, double *DST, const unsigned int DimX, const unsigned int DimY){
   int    NumGPUs;
   cudaError_t e=cudaGetDeviceCount(&NumGPUs);
   if (NumGPUs <=0||e!=0) 
      printf("no hay GPU\n");
   else
   {
      double *ORIG_GPU, *DST_GPU;
      int size = DimX*DimY*sizeof(double);

      // !!!!! Host es el python (ORIG y DST) y Device es la gpu (ORIG_GPU y DST_GPU)

      // Reservamos memoria para las matrices origen y destino en GPU
      cudaMalloc((void **) &ORIG_GPU, size);
      cudaMalloc((void **) &DST_GPU, size);

      // Copiamos la matriz origen en ORIG_GPU (el espacio reservado para ella en la GPU)
      // De momento no hace falta copiar DST ya que esta vacia
      cudaMemcpy(ORIG_GPU, ORIG, size ,cudaMemcpyHostToDevice);

      // Llamada al kernel
      dim3 TPBLK(64);
      dim3 NBLCK((DimX*DimY)/64);
      kernel_DCT8x8<<<NBLCK, TPBLK>>>(ORIG_GPU, DST_GPU, DimX, DimY);
   
      // Copiamos la imagen DST_GPU generada por la GPU en nuestra DST
      cudaMemcpy(DST, DST_GPU, size, cudaMemcpyDeviceToHost);
   
      // Liberamos las memoria en GPU
      cudaFree(ORIG_GPU);
      cudaFree(DST_GPU);

   }

}

// EXTRA : Memoria shared -----------------------------------------------------------------------------------

__global__  void kernel_DCT8x8_shared(double *ORIG, double *DST, const unsigned int DimX, const unsigned int DimY)
{
   int u = blockIdx.x * blockDim.x + threadIdx.x;
   int v = blockIdx.y * blockDim.y + threadIdx.y;

   __shared__ double sh[MaskSize][MaskSize];

   if(u < DimX && v < DimY){

      sh[threadIdx.x][threadIdx.y] = ORIG[u*DimY+v];
      __syncthreads();

      double sum = 0.0;
         for(int x=0; x<MaskSize; x++){
            for(int y=0; y<MaskSize; y++){
               double cos_val1 = cos(((2*x+1)*threadIdx.x*pi)/(2*MaskSize));
               double cos_val2 = cos(((2*y+1)*threadIdx.y*pi)/(2*MaskSize));
               double Csu, Csv;
               if(threadIdx.x==0){
                  Csu = 0.5/sqrt(2.0);
               }
               else{
                  Csu = 0.5;
               }
               if(threadIdx.y==0){
                  Csv = 0.5/sqrt(2.0);
               }
               else{
                  Csv = 0.5;
               }
               sum += Csu * Csv * sh[x][y] * cos_val1 * cos_val2;
            }
         }
      DST[u*DimY+v] = sum;     
   }
} 

extern "C" void TriggerDCT_shared(double *ORIG, double *DST, const unsigned int DimX, const unsigned int DimY){
   int    NumGPUs;
   cudaError_t e=cudaGetDeviceCount(&NumGPUs);
   if (NumGPUs <=0||e!=0) 
      printf("no hay GPU\n");
   else
   {
      double *ORIG_GPU, *DST_GPU;
      int size = DimX*DimY*sizeof(double);

      // !!!!! Host es el python (ORIG y DST) y Device es la gpu (ORIG_GPU y DST_GPU)

      // Reservamos memoria para las matrices origen y destino en GPU
      cudaMalloc((void **) &ORIG_GPU, size);
      cudaMalloc((void **) &DST_GPU, size);

      // Copiamos la matriz origen en ORIG_GPU (el espacio reservado para ella en la GPU)
      // De momento no hace falta copiar DST ya que esta vacia
      cudaMemcpy(ORIG_GPU, ORIG, size ,cudaMemcpyHostToDevice);

      // Llamada al kernel
      dim3 TPBLK(8, 8);
      dim3 NBLCK(DimX/8,DimY/8);
      kernel_DCT8x8_shared<<<NBLCK, TPBLK>>>(ORIG_GPU, DST_GPU, DimX, DimY);
   
      // Copiamos la imagen DST_GPU generada por la GPU en nuestra DST
      cudaMemcpy(DST, DST_GPU, size, cudaMemcpyDeviceToHost);
   
      // Liberamos las memoria en GPU
      cudaFree(ORIG_GPU);
      cudaFree(DST_GPU);

   }

}

__global__ void kernel_IDCT8x8_shared(double *ORIG, double *DST, const unsigned int DimX, const unsigned int DimY)
{
 	
   int u = blockIdx.x * blockDim.x + threadIdx.x;
   int v = blockIdx.y * blockDim.y + threadIdx.y;

   __shared__ double sh[MaskSize][MaskSize];

   if(u < DimX && v < DimY){

      sh[threadIdx.x][threadIdx.y] = ORIG[u*DimY+v];
      __syncthreads();

      double sum = 0.0;
         for(int x=0; x<MaskSize; x++){
            for(int y=0; y<MaskSize; y++){
               double cos_val1 = cos(((2*threadIdx.x+1)*x*pi)/(2*MaskSize));
               double cos_val2 = cos(((2*threadIdx.y+1)*y*pi)/(2*MaskSize));
               double Csx, Csy;
               if(x==0){
                  Csx = 0.5/sqrt(2.0);
               }
               else{
                  Csx = 0.5;
               }
               if(y==0){
                  Csy = 0.5/sqrt(2.0);
               }
               else{
                  Csy = 0.5;
               }
               sum += Csx * Csy * sh[x][y] * cos_val1 * cos_val2;
            }
         }
      DST[u*DimY+v] = round(sum);     
   }

}

extern "C" void TriggerIDCT_shared(double *ORIG, double *DST, const unsigned int DimX, const unsigned int DimY){
   int    NumGPUs;
   cudaError_t e=cudaGetDeviceCount(&NumGPUs);
   if (NumGPUs <=0||e!=0) 
      printf("no hay GPU\n");
   else
   {
      double *ORIG_GPU, *DST_GPU;
      int size = DimX*DimY*sizeof(double);

      // !!!!! Host es el python (ORIG y DST) y Device es la gpu (ORIG_GPU y DST_GPU)

      // Reservamos memoria para las matrices origen y destino en GPU
      cudaMalloc((void **) &ORIG_GPU, size);
      cudaMalloc((void **) &DST_GPU, size);

      // Copiamos la matriz origen en ORIG_GPU (el espacio reservado para ella en la GPU)
      // De momento no hace falta copiar DST ya que esta vacia
      cudaMemcpy(ORIG_GPU, ORIG, size ,cudaMemcpyHostToDevice);

      // Llamada al kernel
      dim3 TPBLK(8, 8);
      dim3 NBLCK(DimX/8,DimY/8);
      kernel_IDCT8x8_shared<<<NBLCK, TPBLK>>>(ORIG_GPU, DST_GPU, DimX, DimY);
   
      // Copiamos la imagen DST_GPU generada por la GPU en nuestra DST
      cudaMemcpy(DST, DST_GPU, size, cudaMemcpyDeviceToHost);
   
      // Liberamos las memoria en GPU
      cudaFree(ORIG_GPU);
      cudaFree(DST_GPU);

   }

}


