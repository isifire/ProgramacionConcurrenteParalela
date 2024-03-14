#include "Prototipos.h"

double Ctimer(void)
{
  struct timeval tm;

  gettimeofday(&tm, NULL);

  return tm.tv_sec + tm.tv_usec / 1.0E6;
}

void Transp(const size_t m, const size_t n, const double *B, double *Bt)
{
  int i, j;

#pragma omp parallel for private(j)
  for (i = 0; i < m; i++)
  {
    for (j = 0; j < n; j++)
    {
      Bt[m * j + i] = B[i * n + j];
    }
  }
}

////////////////////////////// NORMAL ///////////////////////////////////////////////////////

void productoMatricial(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
  int i, j, r;
  double tmp;

#pragma omp parallel for private(tmp, j, r)
  for (i = 0; i < m; i++)
  {
    for (j = 0; j < n; j++)
    {
      tmp = 0;
      for (r = 0; r < k; r++)
      {
        tmp += A[i * lda + r] * B[r * ldb + j];
      }
      C[i * ldc + j] = beta * C[i * ldc + j] + alpha * tmp;
    }
  }
}

void productoTranspuesta(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldbt, double beta, double *C, int ldc)
{
  int i, j, r;
  double tmp;
#pragma omp parallel for private(tmp, j, r)
  for (i = 0; i < m; i++)
  {
    for (j = 0; j < n; j++)
    {
      tmp = 0.0;
      for (r = 0; r < k; r++)
      {
        tmp += A[i * lda + r] * B[j * ldbt + r];
      }
      C[i * ldc + j] = beta * C[i * ldc + j] + alpha * tmp;
    }
  }
}

double MyDGEMM(int tipo, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
  double timeini, timefin;
  double *Bt = NULL;

  switch (tipo)
  {
  case Normal:
    timeini = Ctimer();
    productoMatricial(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    timefin = Ctimer() - timeini;
    break;

  case TransB:
    Bt = (double *)calloc(k * n, sizeof(double));

    if (Bt == NULL)
    {
      printf("Error");
    }
    timeini = Ctimer();

    Transp(k, n, B, Bt);
    productoTranspuesta(m, n, k, alpha, A, lda, Bt, lda, beta, C, ldc);

    timefin = Ctimer() - timeini;

    free(Bt);

    break;

  default:
    timefin = -10;
  }
  return timefin;
}

////////////////////////////// BLOQUES ///////////////////////////////////////////////////////

void productoTranspuestaBloques(int m, int n, int k, double alpha, double *A, int lda, double *Bt, int ldbt, double beta, double *C, int ldc)
{
  int i, j, r;
  double tmp;
#pragma omp parallel for private(j, r)
  for (i = 0; i < m; i++)
  {
    for (j = 0; j < n; j++)
    {
      tmp = 0.0;
      for (r = 0; r < k; r++)
      {
        tmp += A[i * lda + r] * Bt[j * ldbt + r];
      }
      C[i * ldc + j] = beta * C[i * ldc + j] + alpha * tmp;
    }
  }
}

double MyDGEMMB(int tipo, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc, int blk)
{
  double timeini, timefin;

  int i, j, r, ldbt;
  double *Bt = NULL;

  for (i = 0; i < n; i++)
  {
    for (j = 0; j < m; j++)
    {
      C[i * ldc + j] = C[i * ldc + j] * beta;
    }
  }

  switch (tipo)
  {
  case Normal:
    timeini = Ctimer();

#pragma omp parallel for private(j, r)
    for (i = 0; i < m; i += blk)
    {
      for (j = 0; j < n; j += blk)
      {
        for (r = 0; r < k; r += blk)
        {
          MyDGEMM(1, blk, blk, blk, alpha, &A[i * lda + r], lda, &B[r * ldb + j], ldb, 1.0, &C[i * ldc + j], ldc);
        }
      }
    }

    timefin = Ctimer() - timeini;
    break;

  case TransB:
    Bt = (double *)malloc(k * n * sizeof(double));
    ldbt = k;

    for (i = 0; i < k; i++)
    {
      for (j = 0; j < n; j++)
      {
        Bt[j * k + i] = B[i * n + j];
      }
    }
    timeini = Ctimer();

#pragma omp parallel for private(j, r)
    for (i = 0; i < m; i += blk)
    {
      for (j = 0; j < n; j += blk)
      {
        for (r = 0; r < k; r += blk)
        {
          productoTranspuestaBloques(blk, blk, blk, alpha, &A[i * lda + r], lda, &Bt[j * ldbt + r], ldbt, 1, &C[i * ldc + j], ldc);
        }
      }
    }

    free(Bt);
    timefin = Ctimer() - timeini;
    break;

  default:
    timefin = -10;
  }
  return timefin;
}

////////////////////////////// TAREAS ///////////////////////////////////////////////////////

double MyDGEMMT(int tipo, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
  double timeini, timefin;

  int i, j, r, ldbt;
  double tmp, *Bt;

  switch (tipo)
  {
  case Normal:
    timeini = Ctimer();

#pragma omp parallel
#pragma omp single
    for (i = 0; i < m; i++)
    {
#pragma omp task firstprivate(i) private(j, r, tmp)
      for (j = 0; j < n; j++)
      {
        tmp = 0;
        for (r = 0; r < k; r++)
        {
          tmp += A[i * lda + r] * B[r * ldb + j];
        }
        C[i * ldc + j] = beta * C[i * ldc + j] + alpha * tmp;
      }
    }

    timefin = Ctimer() - timeini;
    break;

  case TransB:
    ldbt = k;
    Bt = (double *)malloc(k * n * sizeof(double));
    timeini = Ctimer();

    for (i = 0; i < k; i++)
    {
      for (j = 0; j < n; j++)
      {
        Bt[j * k + i] = B[i * n + j];
      }
    }

#pragma omp parallel
#pragma omp single
    for (i = 0; i < m; i++)
    {
#pragma omp task firstprivate(i) private(j, r, tmp)
      for (j = 0; j < n; j++)
      {
        tmp = 0.0;
        for (r = 0; r < k; r++)
        {
          tmp += A[i * lda + r] * Bt[j * ldbt + r];
        }
        C[i * ldc + j] = beta * C[i * ldc + j] + alpha * tmp;
      }
    }

    timefin = Ctimer() - timeini;
    free(Bt);

    break;

  default:
    timefin = -10;
  }
  return timefin;
}