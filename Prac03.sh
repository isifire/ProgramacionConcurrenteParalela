#!/bin/sh
#
# 1.- Compilar usando fichero Makefile en el nodo destino (de ejecucion)
make

# 2.- Comprobando que se han creado las librerias (la compilacion fue correcta)

#if [ ! -x LIBS/PRACGccO0.so ]; then
#   echo "Upps, la libreria PRACGccO0.so no existe"
#   exit 1
#fi

#if [ ! -x LIBS/PRACGccO3.so ]; then
#   echo "Upps, la libreria PRACGccO3.so no existe"
#   exit 1
#fi

if [ ! -x LIBS/PRACIccO0.so ]; then
   echo "Upps, la libreria PRACIccO0.so no existe"
   exit 1
fi
if [ ! -x LIBS/PRACIccO3.so ]; then
   echo "Upps, la libreria PRACIccO3.so no existe"
   exit 1
fi

# 3.- Ejecutar el ejemplo en paralelo
echo "Paralelo"
export OMP_NUM_THREADS=4
python PRAC03.py
