#include <iostream>
#include <cmath>
#include <chrono>
#include <cfloat>
#include <cstring>

void read_data(int cities, float* x_pos, float* y_pos, FILE* fp){
   	ssize_t read;
   	size_t len = 0;
   	char* line = NULL;
	char* ch;

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

float tsp(int cities, float distance[], int optimal_tour[]){
	float minimum_dist = FLT_MAX;
	int* tour = new int[cities + 1];

	for(int i = 0; i < cities; i++){

		// Set begining tour array in this iteration
		for(int j = 0; j < cities; j++){
			tour[j] = (i + j) % cities;
		}
		tour[cities] = i;

		// Compute begining tour's total distance
		float base_dist = 0;
		for(int j = 0; j < cities; j++){
			base_dist += distance[tour[j] * cities + tour[j + 1]];
		}

		// two-opt computation
		float maximum_diff = 0;
		do{
			// update base distance by maximum_diff computed at previous round
			// maximum_diff means maximum distance could be saved in this round
			base_dist += maximum_diff;
			maximum_diff = 0;
			int index_i = 0, index_j = 0;
			// try to reverse tour array between [j] ~ [k].
			// if this reversion gets better solution , replace maximum_diff by its diff_dist
			for(int j = 1; j < cities - 1; j++){
				for(int k = j + 1; k < cities; k++){

					float diff_dist = distance[tour[j] * cities + tour[k + 1]] +
										distance[tour[j - 1] * cities + tour[k]] -
										distance[tour[j - 1] * cities + tour[j]] -
										distance[tour[k] * cities + tour[k + 1]];
					if(diff_dist < maximum_diff){
						maximum_diff = diff_dist;
						index_i = j;
						index_j = k;
					}
				}
			}

			// reverse tour with recorded index_i and indexj
			while(index_i < index_j){
				int temp = tour[index_i];
				tour[index_i] = tour[index_j];
				tour[index_j] = temp;
				index_i++;
				index_j--;
			}
			// if we can get better solution in this round, then we go throught next round
		}while(maximum_diff <= -0.1);

		if(base_dist <= minimum_dist){
			minimum_dist = base_dist;
			for(int j = 0; j < cities + 1; j++){
				optimal_tour[j] = tour[j];
			}
		}
	}
	return minimum_dist;
}

int main(int argc, char* argv[]){

	int cities = atoi(argv[1]);
	FILE* fp = fopen(argv[2], "r");

	float* distance = new float[cities * cities];
	int* tour = new int[cities + 1];
	
	read_files(fp, distance, cities);
	fclose(fp);
	
	int* app_tour = new int[cities + 1];
	auto start = std::chrono::high_resolution_clock::now();
	float app_minimum_dist = tsp(cities, distance, app_tour);
	auto stop = std::chrono::high_resolution_clock::now();
	std::cout << "Total time taken by function: "
         << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()
		 << " microseconds" << std::endl;

	printf("Approximate minimum distance = %f\n", app_minimum_dist);

}