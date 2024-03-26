#include "Prototipos_Alumnx.h"

const double Cs[8]={0.5/sqrt(2.0), 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

const int zigzag[][2]={
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

 
void FPBB(double* im, const int n, const unsigned int DimY) 
{
   int i, size, N;
   N = 8;
   size = N*N;

   for(i = size - n; i < size; i++){
    im[zigzag[i][0] * DimY + zigzag[i][1]] = 0;
   }
}


void FPAB(double* im, const int n, const unsigned int DimY)
{
   int i;
   for(i = 0; i < n; i++){
    im[zigzag[i][0] * DimY + zigzag[i][1]] = 0;
   }

}

void FPB(double* im, const int n, const unsigned int DimX, const unsigned int DimY)
{
	int i, j;
    int dim = 8;

    #pragma omp parallel for private(j)
	for(i=0; i<DimY; i+=dim){
        for(j=0; j<DimX; j+=dim){
           FPBB(&im[j * DimY + i], n, DimY);
        }
    } 
}

void FPA(double* im, const int n, const unsigned int DimX, const unsigned int DimY)
{
	int i, j;
    int dim = 8;

    #pragma omp parallel for private(j)
	for(i=0; i<DimY; i+=dim){
        for(j=0; j<DimX; j+=dim){
            FPAB(&im[j * DimY + i], n, DimY);
        }
    }
}

void DCT8x8(double *ORIG, double *DST, const unsigned int DimX, const unsigned int DimY)
{
    int u,v,x,y;
    int dim = 8;
    double tmp;

    #pragma omp parallel for private(v, x, y)
    for(u=0;u<dim;u++){    
        for(v=0;v<dim;v++){
            tmp = 0.0;
            for(x = 0;x<dim;x++){            
                for (y = 0;y<dim;y++){               
                    tmp += Cs[u]*Cs[v]*ORIG[x*DimY+y]*cos( (( (2*x)+1) *u*pi) / (2*dim) )* cos( (((2*y)+1)*v*pi) / (2*dim) ); 
                }
                DST[u*DimY+v] = tmp;
            }
            
        }
    }
 
}



void IDCT8x8(double *ORIG, double *DST, const unsigned int DimX, const unsigned int DimY)
{
    int u, v, x, y;
    int dim = 8;
    double tmp, Cu, Cv;

    #pragma omp parallel for private(v, x, y, Cu, Cv, tmp)
    for(u = 0; u < dim; u++){
        for(v = 0; v < dim; v++){ 
            tmp = 0.0;
            for(x = 0; x < dim; x++){     
                for(y = 0; y < dim; y++){       
                    Cu = Cs[y];
                    Cv = Cs[x];
                    tmp += Cu * Cv * ORIG[y * DimY + x] * cos((((2 * v) + 1) * y * pi) / (2 * dim)) * cos((((2 * u) + 1) * x * pi) / (2 * dim)); 
                }
            }
            DST[v *DimY + u] = round(tmp);
        }
    }
}


void TriggerDCT(double *ORIG, double *DST, const unsigned int DimX, const unsigned int DimY)
{

    int i,j, dim;
    dim = 8;

    #pragma omp parallel for private(j)
    for(i = 0;i<DimX;i+=dim){
        for(j =0 ;j < DimY;j+=dim){
            DCT8x8(&ORIG[i*DimY+j],&DST[i*DimY+j],DimX,DimY);
        }
 
    }

}


void TriggerIDCT(double *ORIG, double *DST, const unsigned int DimX, const unsigned int DimY)
{
	int i,j, dim;
    dim = 8;

    #pragma omp parallel for private(j)
    for(i = 0;i<DimX;i+=dim){
        for(j =0 ;j < DimY;j+=dim){
            IDCT8x8(&ORIG[j*DimX+i],&DST[j*DimX+i],DimX,DimY);
        }
 
    }
}
