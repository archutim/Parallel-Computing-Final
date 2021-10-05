#include <iostream>
#include <pthread.h>
#include <cfloat>
#include <chrono>
#include <sys/time.h>

#define threadsPerBlock_x 16
#define threadsPerBlock_y 16

struct timeval stop, start;

typedef struct _thread_param{
	int cities;
	int num_threads;
	int thread_id;
	float* distance;
	float* thread_minimum;
	int* p_tour;
}thread_param;

void read_data(int cities, float* x_pos, float* y_pos, FILE* fp){
   	ssize_t read;
   	size_t len = 0;
   	char* line = NULL;
	char* ch = NULL;

   	for(int i = 0; i < cities; i++){
   		read = getline(&line, &len, fp);
   		if(read == -1){
   			fprintf(stderr, "Error in reading file");
      		exit(1);
   		}

   		ch = strtok(line," ");
		ch = strtok(NULL, " ");
		x_pos[i] = atof(ch);
		ch = strtok(NULL, " ");
		y_pos[i] = atof(ch);

   	}
}

void cal_dist(float* distance, float* x_pos, float* y_pos, int cities){
	float x_diff;
	float y_diff;
	for(int i = 0; i < cities; i++){
		for(int j = 0; j < cities; j++){
			x_diff = x_pos[i] - x_pos[j];
			y_diff = y_pos[i] - y_pos[j];
			distance[i * cities + j] = sqrt(x_diff * x_diff + y_diff * y_diff);
		}
	}
}

void read_files(FILE* fp, float* distance, int cities){
	float* x_pos = new float[cities];
	float* y_pos = new float[cities];
	if(fp == NULL){
      fprintf(stderr, "Can't open the file.\n");
      exit(1);
   	}
	read_data(cities, x_pos, y_pos, fp);
	cal_dist(distance, x_pos, y_pos, cities);
	delete[] x_pos;
	delete[] y_pos;
}

__global__ void gpu_two_opt(int cities, int* tour, float* distance, float* gpu_minimum, int* gpu_head_tail){
	__shared__ float thread_minimum[threadsPerBlock_x * threadsPerBlock_y];
	__shared__ int thread_head_tail[threadsPerBlock_x * threadsPerBlock_y];
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_x = blockDim.x * gridDim.x;
	int stride_y = blockDim.y * gridDim.y;
	float thread_min_diff = 0;
	int head_tail = 0;
	for(int i = x + 1; i < cities - 1; i += stride_x){
		for(int j = y + 1; j < cities; j += stride_y){
			if(i < j){
				float diff_dist = distance[tour[i] * cities + tour[j + 1]] +
									distance[tour[i - 1] * cities + tour[j]] -
									distance[tour[i - 1] * cities + tour[i]] -
									distance[tour[j] * cities + tour[j + 1]];
				if(diff_dist < thread_min_diff){
					thread_min_diff = diff_dist;
					head_tail = i * cities + j;
				}
			}
		}
	}
	thread_minimum[blockDim.x * threadIdx.y + threadIdx.x] = thread_min_diff;
	thread_head_tail[blockDim.x * threadIdx.y + threadIdx.x] = head_tail;
	__syncthreads();
	if(threadIdx.x == 0 && threadIdx.y == 0){
		float block_minimum = 0;
		int block_head_tail = 0;
		for(int i = 0; i < blockDim.x * blockDim.y; i++){
			if(thread_minimum[i] < block_minimum){
				block_minimum = thread_minimum[i];
				block_head_tail = thread_head_tail[i];
			}
		}
		gpu_minimum[blockIdx.y * gridDim.x + blockIdx.x] = block_minimum;
		gpu_head_tail[blockIdx.y * gridDim.x + blockIdx.x] = block_head_tail;
	}
	__syncthreads();
}

// function to be called by each thread
void* pthread_tsp(void* _param){
	// read pthread parameters
	thread_param* param = (thread_param *) _param;
	int cities = param->cities;
	int num_threads = param->num_threads;
	int thread_id = param->thread_id;
	float* distance = param->distance;
	float* thread_minimum = param->thread_minimum;
	int* p_tour = param->p_tour;

	// setup gpu
	cudaSetDevice(thread_id);
	dim3 threadsPerBlock(threadsPerBlock_x, threadsPerBlock_y);
	dim3 numBlocks(cities / threadsPerBlock_x + ((cities % threadsPerBlock_x) ? 1 : 0),
					cities / threadsPerBlock_y + ((cities % threadsPerBlock_y) ? 1 : 0));

	float* gpu_distance = NULL;
	cudaMalloc(&gpu_distance, cities * cities * sizeof(float));
    cudaMemcpy(gpu_distance, distance, cities * cities * sizeof(float), cudaMemcpyHostToDevice);

	int* cpu_tour = new int[cities + 1], *gpu_tour = NULL;
	cudaMalloc(&gpu_tour, (cities + 1) * sizeof(int));

	float* cpu_minimum = new float[numBlocks.x * numBlocks.y], *gpu_minimum = NULL;
	cudaMalloc(&gpu_minimum, numBlocks.x * numBlocks.y * sizeof(float));
	int* cpu_head_tail = new int[numBlocks.x * numBlocks.y], *gpu_head_tail = NULL;
	cudaMalloc(&gpu_head_tail, numBlocks.x * numBlocks.y * sizeof(int));

	// measure per pthread execution time
	gettimeofday(&start, NULL);

	for(int i = thread_id; i < cities; i += num_threads){
		// create initial solution
		for(int j = 0; j < cities; j++){
			cpu_tour[j] = (i + j) % cities;
		}
		cpu_tour[cities] = i;
		// calculate initial solution cost
		float base_dist = 0;
		for(int j = 0; j < cities; j++){
			base_dist += distance[cpu_tour[j] * cities + cpu_tour[j + 1]];
		}

		float maximum_diff = 0;
		do{
			base_dist += maximum_diff;
			maximum_diff = 0;
			// run 2-opt on device
			cudaMemcpy(gpu_tour, cpu_tour, (cities + 1) * sizeof(int), cudaMemcpyHostToDevice);
			gpu_two_opt<<<numBlocks, threadsPerBlock>>>(cities, gpu_tour, gpu_distance, gpu_minimum, gpu_head_tail);
			cudaDeviceSynchronize();
			// copy block minimums from device to host
			cudaMemcpy(cpu_minimum, gpu_minimum, numBlocks.x * numBlocks.y * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(cpu_head_tail, gpu_head_tail, numBlocks.x * numBlocks.y * sizeof(int), cudaMemcpyDeviceToHost);
			int head = 0, tail = 0;
			// find the smallest cost in block minimums
			for(int j = 0; j < numBlocks.x * numBlocks.y; j++){
				if(cpu_minimum[j] < maximum_diff){
					maximum_diff = cpu_minimum[j];
					head = cpu_head_tail[j] / cities;
					tail = cpu_head_tail[j] % cities;
				}
			}
			// reverse solution array
			while(head < tail){
				int temp = cpu_tour[head];
				cpu_tour[head] = cpu_tour[tail];
				cpu_tour[tail] = temp;
				head++;
				tail--;
			}
			// check whether get better solution
		}while(maximum_diff <= -0.1);
		
		// update global minimum
		if(base_dist < *thread_minimum){
			*thread_minimum = base_dist;
			for(int j = 0; j < cities + 1; j++){
				p_tour[j] = cpu_tour[j];
			}
		}
	}

	gettimeofday(&stop, NULL);
	printf("pthread %d : %lu microseconds\n", thread_id, (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
	// free cuda memory
	cudaFree(gpu_distance);
	cudaFree(gpu_tour);
	cudaFree(gpu_minimum);
	cudaFree(gpu_head_tail);
    return NULL;
}

int main(int argc, char* argv[])
{

	int cities = atoi(argv[1]);
	int num_threads = atoi(argv[2]);
	FILE* fp=fopen(argv[3], "r");

	//read the data from the file
	float* distance = new float[cities * cities];
	int* tour = new int[cities + 1];
	read_files(fp, distance, cities);	
	fclose(fp);

	// pthread parameters struct
	thread_param* params = new thread_param[num_threads];

	pthread_t* threads = new pthread_t[num_threads];
	// measure total execution time
	auto start = std::chrono::high_resolution_clock::now();
	// set parameters and create pthread
	for(int i = 0; i < num_threads; i++){
		params[i].cities = cities;
		params[i].num_threads = num_threads;
		params[i].thread_id = i;
		params[i].distance = distance;
		params[i].thread_minimum = new float[1];
		*(params[i].thread_minimum) = FLT_MAX;
		params[i].p_tour = new int[cities + 1];
		pthread_create(&threads[i], NULL, pthread_tsp, &params[i]);
	}
	// thread join
	for(int i = 0; i < num_threads; i++){
		pthread_join(threads[i], NULL);
	}
	// show total execution time
	auto stop = std::chrono::high_resolution_clock::now();
	std::cout << "Total time taken by function: "
         << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()
		 << " microseconds" << std::endl;
	// show solution of each thread
	for(int i = 0; i < num_threads; i++){
		printf("p_thread %d's optimal solution: %f\n", i, *(params[i].thread_minimum));
	}
	return 0;
}