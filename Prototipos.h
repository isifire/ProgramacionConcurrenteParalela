/* ********************************************************************** */
/*                     ESTE FICHERO NO DEBE SER MODIFICADO                */
/* ********************************************************************** */
#ifndef PRAC03_H
#define PRAC03_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#ifdef MKL
  #include <mkl.h>
#else
  #include <cblas.h>
#endif

#define Normal 1
#define TransB 2

#endif

double Ctimer(void);

double MyDGEMM(int, int, int, int, double, double*, int, double*, int, double, double*, int);
double MyDGEMMB(int, int, int, int, double, double*, int, double*, int, double, double*, int, int);
double MyDGEMMT(int, int, int, int, double, double*, int, double*, int, double, double*, int);
