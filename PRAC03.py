import ctypes
import numpy as np 

from ctypes import CDLL, c_size_t, c_double
from numpy  import linalg as LA
from random import random     
from time   import time

P1D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")
P2D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C")

IccO0 = CDLL('LIBS/PRACIccO0.so')
IccO3 = CDLL('LIBS/PRACIccO3.so')

IccO0.MyDGEMM.restype = c_double
IccO3.MyDGEMM.restype = c_double

IccO0.MyDGEMMB.restype = c_double
IccO3.MyDGEMMB.restype = c_double

IccO0.MyDGEMMT.restype = c_double
IccO3.MyDGEMMT.restype = c_double

IccO0.MyDGEMM.argtypes = [c_size_t, c_size_t, c_size_t, c_size_t, c_double, P2D, c_size_t, P2D, c_size_t, c_double, P2D, c_size_t]
IccO3.MyDGEMM.argtypes = [c_size_t, c_size_t, c_size_t, c_size_t, c_double, P2D, c_size_t, P2D, c_size_t, c_double, P2D, c_size_t]

IccO0.MyDGEMMB.argtypes = [c_size_t, c_size_t, c_size_t, c_size_t, c_double, P2D, c_size_t, P2D, c_size_t, c_double, P2D, c_size_t, c_size_t]
IccO3.MyDGEMMB.argtypes = [c_size_t, c_size_t, c_size_t, c_size_t, c_double, P2D, c_size_t, P2D, c_size_t, c_double, P2D, c_size_t, c_size_t]

IccO0.MyDGEMMT.argtypes = [c_size_t, c_size_t, c_size_t, c_size_t, c_double, P2D, c_size_t, P2D, c_size_t, c_double, P2D, c_size_t]
IccO3.MyDGEMMT.argtypes = [c_size_t, c_size_t, c_size_t, c_size_t, c_double, P2D, c_size_t, P2D, c_size_t, c_double, P2D, c_size_t]

talla = [1000, 1000]
rept  = [1,1]
alpha = 1.3
beta  = 1.7
blk   = 20
tipo  = 1  # 1 normal, 2 transpuesta de B

for i in range(0,len(talla)):
   m      = talla[i]
   n      = m + 1
   k      = m - 1
   

   A = np.random.rand(m, k).astype(np.float64)
   B = np.random.rand(k, n).astype(np.float64)
   C = np.random.rand(m, n).astype(np.float64)

   D = np.copy(C)
   secs = time()
   for j in range(rept[i]):
     D = beta*D + alpha*(A @ B)
   TIEMPO = (time()- secs)/rept[i]
   print(f"Python {m}x{n}x{k} Segundos={TIEMPO:1.5E}")

   F = np.copy(C)
   
   #G = np.copy(B)
   #R = B.transpose()


   F = np.copy(C)
   secs = time()
   for j in range(rept[i]):
     TiempC=IccO0.MyDGEMM(tipo, m, n, k, alpha, A, k, B, n, beta, F, n)
     #TiempC=IccO0.MyDGEMMT(tipo, m, n, k, alpha, A, k, B, n, beta, F, n)
     #TiempC=IccO0.MyDGEMMB(tipo, m, n, k, alpha, A, k, B, n, beta, F, n, blk)
   TIEMPO = (time()- secs)/rept[i]
   print(f"IccO0  {m}x{n}x{k} Segundos={TIEMPO:1.5E} (Segundos medidos en C={TiempC:1.5E})")
   print(f"Error entre Python y IccO0 {LA.norm(D-F, 'fro'):1.5E}")

   F = np.copy(C)
   secs = time()
   for j in range(rept[i]):
     TiempC=IccO3.MyDGEMM(tipo, m, n, k, alpha, A, k, B, n, beta, F, n)
     #TiempC=IccO3.MyDGEMMT(tipo, m, n, k, alpha, A, k, B, n, beta, F, n)
     #TiempC=IccO3.MyDGEMMB(tipo, m, n, k, alpha, A, k, B, n, beta, F,n,blk)
   TIEMPO = (time()- secs)/rept[i]
   print(f"IccO3  {m}x{n}x{k} Segundos={TIEMPO:1.5E} (Segundos medidos en C={TiempC:1.5E})")
   print(f"Error entre Python y IccO3 {LA.norm(D-F, 'fro'):1.5E}\n")
