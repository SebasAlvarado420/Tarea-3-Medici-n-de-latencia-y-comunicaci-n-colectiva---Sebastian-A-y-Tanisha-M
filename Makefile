.PHONY: venv deps partA partB clean

venv:
	python3 -m venv .venv

deps:
	source .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

partA:
	mpirun -np 4 python estadisticas_mpi.py --n 1000000 --mode float --verbose

partB:
	mpirun -np 2 python latencia_mpi.py --iters 15000 --sizes 1,4096,65536,1048576 --barrier --csv results/latencias.csv

clean:
	rm -f results/latencias.csv