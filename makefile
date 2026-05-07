NVCC=nvcc
FLAGS=-DDEBUG -O3
NVCCFLAGS=$(FLAGS) -Xcompiler -Wall
LIBS=-lm
ALWAYS_REBUILD=makefile

nbody: nbody.o compute.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LIBS)

nbody.o: nbody.c planets.h config.h vector.h compute.h $(ALWAYS_REBUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

compute.o: compute.cu config.h vector.h compute.h $(ALWAYS_REBUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f *.o nbody
