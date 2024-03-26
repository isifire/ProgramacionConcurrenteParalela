#!/bin/sh
#
# 1.- Compilar usando fichero Makefile en el nodo destino (de ejecucion)
make

# 2.- Comprobando que se han creado las librerias (la compilacion fue correcta)
if [ ! -x LIBS/DCTAlumnx.so ]; then
   echo "Upps, la libreria DCTAlumnx.so no existe"
   exit 1
fi

# 3.- Ejecutar el ejemplo en secuencial
echo "Secuencial"
export OMP_NUM_THREADS=1
python DCT.py

# 4.- Ejecutar el ejemplo en paralelo
#echo ""
#echo "Paralelo"
#unset OMP_NUM_THREADS
#python DCT.py
