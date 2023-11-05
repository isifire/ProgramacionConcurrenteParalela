#include "Prototipos.h"

double Ctimer(void)
{
  struct timeval tm;

  gettimeofday(&tm, NULL);

  return tm.tv_sec + tm.tv_usec/1.0E6;
}


// k*n
void Transp(const size_t m, const size_t n, const double *B, double *Bt)
{
  for(int i = 0; i < m; i++){
    for(int j = 0;j < n; j++){
        Bt[j*m+i] = B[i*n+j];
    }
  }
}


double MyDGEMM(int tipo, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
  double timeini, timefin;

  int i,j,r; double tmp;
  double *Bt=NULL;

  switch (tipo)
  {
    case Normal:
      timeini=Ctimer();  

        // Llamar a la funcion del alumno normal. POR HACER POR EL ALUMNO
        #pragma omp parallel for private(tmp,j,r)
        for(i=0;i<m;i++){
          for(j=0;j<n;j++){
            tmp = 0;
            for(r = 0;r<k;r++){
              tmp+=A[i*lda+r]*B[r*ldb+j];
            }
            C[i*ldc+j]=beta*C[i*ldc+j]+alpha*tmp;
          }
        }
      timefin=Ctimer()-timeini;  
      break;
    case TransB:
    // RESERVAR MEMORIA PARA Bt. POR HACER POR EL ALUMNO

      Bt = (double *)calloc(k*n,sizeof(double));
      
      if(Bt == NULL){
        printf("Error");
      }
      timeini=Ctimer();
       Transp(k,n,B,Bt);
       
       for(i=0;i<m;i++)
         for(j=0;j<n;j++)
         {
           tmp = 0.0;
           for(r = 0;r<k;r++)
             tmp+=A[i*lda+r]*Bt[j*k+r];
           C[i*ldc+j]=beta*C[i*ldc+j]+alpha*tmp;
         }
      timefin=Ctimer()-timeini;
      // LIBERAR MEMORIA (free) de Bt. POR HACER POR EL ALUMNO
      free(Bt);
      break;
    default:
      timefin=-10;
  }
  return timefin;
}


double MyDGEMMB(int tipo, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc, int blk)
{
  double timeini, timefin;

  int i,j,r; double tmp;
  double *Bt=NULL;

  switch (tipo)
  {
    case Normal:
      timeini=Ctimer();  

        // Llamar a la funcion del alumno normal. POR HACER POR EL ALUMNO

        for(i=0;i<blk;i++){
          for(j=0;j<n;j++){ // o blk?
            for(r = 0;r<k;r++){
            MyDGEMM(tipo,m,n,k,alpha,&A[i],lda,&B[i],ldb,1,&C[i],ldc)
            }
          }
        }
  
      timefin=Ctimer()-timeini;  
      break;
    case TransB:
    // RESERVAR MEMORIA PARA Bt. POR HACER POR EL ALUMNO

      Bt = (double *)calloc(k*n,sizeof(double));
      
      if(Bt == NULL){
        printf("Error");
      }
      timeini=Ctimer();
       Transp(k,n,B,Bt);
       
       for(i=0;i<m;i++)
         for(j=0;j<n;j++)
         {
           tmp = 0.0;
           for(r = 0;r<k;r++)
             tmp+=A[i*lda+r]*Bt[j*k+r];
           C[i*ldc+j]=beta*C[i*ldc+j]+alpha*tmp;
         }
      timefin=Ctimer()-timeini;
      // LIBERAR MEMORIA (free) de Bt. POR HACER POR EL ALUMNO
      free(Bt);
      break;
    default:
      timefin=-10;
  }
  return timefin;
}


