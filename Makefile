all: cleanall creadir PRACIccO0.so PRACIccO3.so 

flags1Slow=-Wall -O0 -fp-model=strict -qmkl=parallel -qopenmp -DMKL
flags1Fast=-Wall -O3 -fp-model=strict -qmkl=parallel -qopenmp -DMKL


creadir: 
	@mkdir LIBS
	
cleanall: clean
	@rm -rf	LIBS

clean:
	@rm -f *.o core *~
	

PRACIccO0.so: PRAC03.c
	icx -o LIBS/PRACIccO0.so -qopenmp -fPIC -shared PRAC03.c $(flags1Slow)

PRACIccO3.so: PRAC03.c
	icx -o LIBS/PRACIccO3.so -qopenmp -fPIC -shared PRAC03.c $(flags1Fast)
