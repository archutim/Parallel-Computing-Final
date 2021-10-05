Compilation:
	nvcc -O3 -o gpu_tsp gpu_tsp.cu
	g++ -O3 -o seq_tsp seq_tsp.cc
Usage:
	srun -c[core_num] -p ipc21 --gres=gpu:[gpu_num] gpu_tsp [input_size] [gpu_num] [input_file]
	Ex:
		srun -c2 -p ipc21 --gres=gpu:2 gpu_tsp 16 2 data/a16

	./seq_tsp [input_size] [input_fine]
	Ex:
		./seq_tsp 442 data/pcb442