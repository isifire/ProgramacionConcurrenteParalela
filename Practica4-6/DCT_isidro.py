import sys

from PIL import Image
import ctypes
import numpy as np 
from numpy.ctypeslib import ndpointer
from time import time



#########################################################################
#        Preparar gestión librería externa de Profesor	 (NO TOCAR)	#
#########################################################################

libProf = ctypes.cdll.LoadLibrary('./DCTProf.so')
# Preparando para el uso de "void DCT(double, double, double, double, int, int, int, double *)
# .restype  se refiere al tipo de lo que retorna la funcion. Si es void, valor "None"
# .argtypes se refiere al tipo de los argumentos de la funcion

#void TriggerDCT(double *, double *, const unsigned int, const unsigned int);
DCTProf = libProf.TriggerDCT
DCTProf.restype  = None
DCTProf.argtypes = [ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"), ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"), ctypes.c_int, ctypes.c_int]

#void TriggerIDCT(double *, double *, const unsigned int, const unsigned int);
IDCTProf = libProf.TriggerIDCT
IDCTProf.restype  = None
IDCTProf.argtypes = [ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"), ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"), ctypes.c_int, ctypes.c_int]

#void FPB(double *, const int, const unsigned int, const unsigned int);
FPBProf = libProf.FPB
FPBProf.restype  = None
FPBProf.argtypes = [ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int]

#void FPA(double *, const int, const unsigned int, const unsigned int);
FPAProf = libProf.FPA
FPAProf.restype  = None
FPAProf.argtypes = [ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int]


#########################################################################
#       Preparar gestión librería externa de Alumnx	 		#
#########################################################################
#Librería DCTAlumnx.so

libAlumnx = ctypes.cdll.LoadLibrary('./DCTAlumnx.so')

#void TriggerDCT(double *, double *, const unsigned int, const unsigned int);
DCTAlumnx = libAlumnx.TriggerDCT
DCTAlumnx.restype  = None
DCTAlumnx.argtypes = [ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"), ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"), ctypes.c_int, ctypes.c_int]

#void TriggerIDCT(double *, double *, const unsigned int, const unsigned int);
IDCTAlumnx = libAlumnx.TriggerIDCT
IDCTAlumnx.restype  = None
IDCTAlumnx.argtypes = [ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"), ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"), ctypes.c_int, ctypes.c_int]

#void FPB(double *, const int, const unsigned int, const unsigned int);
FPBAlumnx = libAlumnx.FPB
FPBAlumnx.restype  = None
FPBAlumnx.argtypes = [ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int]

#void FPA(double *, const int, const unsigned int, const unsigned int);
FPAAlumnx = libAlumnx.FPA
FPAAlumnx.restype  = None
FPAAlumnx.argtypes = [ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int]


#########################################################################
# 	Función de cálculo de error	(NO TOCAR)			#
#########################################################################
def Error(a, b):
   err=0.0
   n=len(a)
   for i in range(n):
      err += (a[i] - b[i])**2.0
   
   print(f"Error:{(err**0.5)/(n**0.5)}")
   

#########################################################################
# 	Función para leer la imagen de archivo	(NO TOCAR)		#
#########################################################################
def leer(nombre):
    im = Image.open(nombre).convert('L')
    #im.show() 
    vector = np.array(im, order='F').astype(np.double)
    return np.ndarray.flatten(vector), vector.shape[1], vector.shape[0]

#########################################################################
# 	Función para guardar la imagen a archivo (NO TOCAR)		#
#########################################################################
def grabar(vect, xres, yres, output):
    A2D=vect.astype(np.ubyte).reshape(yres,xres)
    im=Image.fromarray(A2D)
    im.save(outputfile)


#########################################################################
# 			MAIN						#
#########################################################################   
if __name__ == "__main__":
    #  Proceado de los agumentos					#
    if len(sys.argv) != 3:
        print('DCT.py <inputfile> <outputfile>')
        print("Ejemplo: test.png out.bmp")
        sys.exit(2)
    outputfile = sys.argv[2]
    inputfile = sys.argv[1]
    
    original, DimX, DimY = leer(inputfile)
    print(f'\nEjecutando {DimY}x{DimX}')

    #  Reserva de memoria de las imágenes en 1D	Prof	(NO TOCAR)	#
    dctProf = np.zeros(DimX*DimY).astype(np.double)
    idctProf = np.zeros(DimX*DimY).astype(np.double)

    #  Reserva de memoria de las imágenes en 1D	Alumnx			#
    dctAlumnx = np.zeros(DimX*DimY).astype(np.double)
    idctAlumnx = np.zeros(DimX*DimY).astype(np.double)

    #  Llamadas a las funciones de cálculo de la DCT Alumnx		#
    s = time()
    DCTAlumnx(original, dctAlumnx, DimX, DimY)
    s = time()- s
    print(f"DCTAlumnx		ha tardado {s:1.5E} segundos")   


    #  Llamadas a las funciones de cálculo de la DCT Prof (NO TOCAR)	#
    s = time()
    DCTProf(original, dctProf, DimX, DimY)
    s = time()- s
    print(f"DCTProf		ha tardado {s:1.5E} segundos")
    
    #  Llamar a la comprobación de error				#
    Error(dctAlumnx,dctProf)
    
    
    #  Llamadas a las funciones de cálculo de la IDCT Prof (NO TOCAR)	#
    s = time()
    IDCTProf(dctProf, idctProf, DimX, DimY)
    s = time()- s
    print(f"IDCTProf	ha tardado {s:1.5E} segundos")

    #  Llamadas a las funciones de cálculo de la IDCT Alumnx		#
    s = time()
    IDCTAlumnx(dctAlumnx, idctAlumnx, DimX, DimY)
    s = time()- s
    print(f"IDCTAlumnx	ha tardado {s:1.5E} segundos")    
    
    #  Llamar a la comprobación de error				#
    Error(idctAlumnx,idctProf)

    fpaProf = np.copy(dctProf)
    fpbProf = np.copy(dctProf)
    fpaiProf = np.zeros(DimX*DimY).astype(np.double)
    fpbiProf = np.zeros(DimX*DimY).astype(np.double)

    #  Llamadas a las funciones de cálculo de la FPA Prof y FPB Prof (NO TOCAR)	#
    s = time()
    FPAProf(fpaProf, 60, DimX, DimY)
    s = time()- s
    print(f"FPAProf	ha tardado {s:1.5E} segundos")

    s = time()
    FPBProf(fpbProf, 60, DimX, DimY)
    s = time()- s
    print(f"FPBProf	ha tardado {s:1.5E} segundos")

    IDCTProf(fpaProf, fpaiProf, DimX, DimY)
    IDCTProf(fpbProf, fpbiProf, DimX, DimY)

    fpaAlumnx = np.copy(dctAlumnx)
    fpbAlumnx = np.copy(dctAlumnx)
    fpaiAlumnx = np.zeros(DimX*DimY).astype(np.double)
    fpbiAlumnx = np.zeros(DimX*DimY).astype(np.double)

    #  Llamadas a las funciones de cálculo de la FPA y FPB Alumnx		#

    s = time()
    FPAAlumnx(fpaAlumnx, 60, DimX, DimY)
    s = time()- s
    print(f"FPAAlumnx ha tardado {s:1.5E} segundos")

    s = time()
    FPBAlumnx(fpbAlumnx, 60, DimX, DimY)
    s = time()- s
    print(f"FPBAlumnx ha tardado {s:1.5E} segundos")

    IDCTAlumnx(fpaAlumnx, fpaiAlumnx, DimX, DimY)
    IDCTAlumnx(fpbAlumnx, fpbiAlumnx, DimX, DimY)
    
    #  Llamar a la comprobación de error				#
    print("Para FPA: ")
    Error(fpaiAlumnx, fpaiProf)

    print("Para FPB: ")
    Error(fpbiAlumnx, fpbiProf)

    #  Grabar a archivo	matriz que se necesite				#
    grabar(idctAlumnx,DimX,DimY,outputfile)

   
