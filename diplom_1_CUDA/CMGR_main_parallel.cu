// Итерац. МСГ с регуляризацией на  видеокарте
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <iomanip>

#include <windows.h>

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CALL( call )               \
{                                       \
cudaError_t result = call;              \
if ( cudaSuccess != result )            \
    std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else

__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
				__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}
#endif

using namespace std;

//global
const int S = 134862;

//global GPU
const int block_size = 1;
const int grid_size = 1;

const int per_thread = S / grid_size * block_size;

void clear_nevyazka() {
	//clear file with nevyazka
	ofstream nevyazka_file;
	nevyazka_file.open("nevyazka.dat", ofstream::trunc);
	nevyazka_file.close();
}

void add_in_file(double nev) {
	ofstream nevyazka;
	nevyazka.open("nevyazka.dat", ios_base::app);
	nevyazka << nev << endl;
	nevyazka.close();
}

void write_in_file(double *vec, double size, string name) {
	clock_t begin_write = clock();

	cout << "Start write in file" << endl;
	ofstream result_file;
	result_file.open(name, ofstream::trunc); //x_res_from_main.txt

	for (int i = 0; i < size; ++i) {
		result_file << setprecision(16) << vec[i] << endl;
	}

	result_file.close();

	clock_t end_write = clock();
	double write_time = double(end_write - begin_write) / CLOCKS_PER_SEC;
	cout << "Stop write in file: " << write_time << endl;
}

void create_matr(ifstream& file, double *matr, double size) {
	for (int i = 0; i < size; ++i) {
		file >> setprecision(16) >> matr[i];
	}
}

//-------------------------------------------------------------------------------------------

void show_vec(double *matr, double size) {
	for (int i = 0; i < size; ++i) {
		cout << setprecision(16) << matr[i] << "; ";
	}
	cout << endl;
}

__global__ void GPU_show_vec(double *matr, double size) {
	printf("X: ");
	for (int i = 0; i < size; ++i) {
		printf("%.16f\n", matr[i]);
	}
}

//-------------------------------------------------------------------------------------------

void mult_vec_on_num(double *vec, double num, double *res_vec) {
	for (int i = 0; i < S; ++i) {
		res_vec[i] = vec[i] * num;
	}
}

__global__ void GPU_mult_vec_on_num(double *vec, double *num, double *res_vec) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread * per_thread; i < (thread + 1) * per_thread; ++i) {
		res_vec[i] = __dmul_ru(vec[i],*num);
	}
	//printf("%.16f = %.16f * %.16f\n", res_vec[0], vec[0], *num);
}

//-------------------------------------------------------------------------------------------

//function changing original matrix A
void mult_A_on_alpha_E(double *matr, double alpha) {
	for (int i = 0; i < S; ++i) {
		if (i == 0) {
			matr[i] = matr[i] + alpha;
		}
		else if (i > 0 && i < 247) {
			matr[i * 7 + 1] = matr[i * 7 + 1] + alpha;
		}
		else if (i > 246 && i < 494) {
			matr[i * 7 + 2] = matr[i * 7 + 2] + alpha;
		}
		else if (i > 493 && i < 134368) {
			matr[i * 7 + 3] = matr[i * 7 + 3] + alpha;
		}
		else if (i > 134367 && i < 134615) {
			matr[i * 7 + 4] = matr[i * 7 + 4] + alpha;
		}
		else if (i > 134614 && i < 134861) {
			matr[i * 7 + 5] = matr[i * 7 + 5] + alpha;
		}
		else if (i == 134861) { //944034
			matr[i * 7 + 6] = matr[i * 7 + 6] + alpha;
		}
	}
}

__global__ void GPU_mult_A_on_alpha_E(double *matr, double alpha) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread * per_thread; i < (thread + 1) * per_thread; ++i) {
		if (i == 0) {
			matr[i] = __dadd_ru(matr[i], alpha);
		}
		else if (i > 0 && i < 247) {
			matr[i * 7 + 1] = __dadd_ru(matr[i * 7 + 1], alpha);
		}
		else if (i > 246 && i < 494) {
			matr[i * 7 + 2] = __dadd_ru(matr[i * 7 + 2], alpha);
		}
		else if (i > 493 && i < 134368) {
			matr[i * 7 + 3] = __dadd_ru(matr[i * 7 + 3], alpha);
		}
		else if (i > 134367 && i < 134615) {
			matr[i * 7 + 4] = __dadd_ru(matr[i * 7 + 4], alpha);
		}
		else if (i > 134614 && i < 134861) {
			matr[i * 7 + 5] = __dadd_ru(matr[i * 7 + 5], alpha);
		}
		else if (i == 134861) { //944034
			matr[i * 7 + 6] = __dadd_ru(matr[i * 7 + 6], alpha);
		}
	}
	//printf("%.14f\n", matr[1]);
}

//-------------------------------------------------------------------------------------------

void sum_vec(double *vec_one, double *vec_two, double *res_vec) {
	for (int i = 0; i < S; ++i) {
		res_vec[i] = vec_one[i] + vec_two[i];
	}
}

__global__ void GPU_sum_vec(double *vec_one, double *vec_two, double *res_vec) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread * per_thread; i < (thread + 1) * per_thread; ++i) {
		res_vec[i] = __dadd_ru(vec_one[i], vec_two[i]);
	}
}

//-------------------------------------------------------------------------------------------

void dif_vec(double *vec_one, double *vec_two, double *res_vec) {
	for (int i = 0; i < S; ++i) {
		res_vec[i] = vec_one[i] - vec_two[i];
	}
}

__global__ void GPU_dif_vec(double *vec_one, double *vec_two, double *res_vec) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread * per_thread; i < (thread + 1) * per_thread; ++i) {
		res_vec[i] = __dsub_ru(vec_one[i], vec_two[i]);
		//debug
		/*if (i > S / 2 && i < S / 2 + 10) {
			printf("%.14f = %.14f - %.14f\n", res_vec[i], vec_one[i], vec_two[i]);
		}*/
	}
}

//-------------------------------------------------------------------------------------------

void copy_matr(double *m_one, double *m_two) {
	for (int i = 0; i < S * 7; ++i) {
		m_one[i] = m_two[i];
	}
}

__global__ void GPU_copy_matr(double *m_one, double *m_two) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = (thread * per_thread) * 7; i < ((thread + 1) * per_thread) * 7; ++i) {
		m_one[i] = m_two[i];
	}
}

//-------------------------------------------------------------------------------------------

void matr_on_vec(double *matr, double *vec, double *res_vec) {
	int second_count = 0;
	int third_count = 0;
	int fourth_count = 0;
	int fifth_count = 0;
	int sixth_count = 0;
	
	for (int i = 0; i < S; ++i) {
		if (i == 0) {
			res_vec[i] = \
				matr[0] * vec[0] \
				+ matr[1] * vec[1] \
				+ matr[2] * vec[247] \
				+ matr[3] * vec[494];
		}
		else if (i > 0 && i < 247) {
			res_vec[i] = \
				matr[7 * i] * vec[second_count] \
				+ matr[7 * i + 1] * vec[second_count + 1] \
				+ matr[7 * i + 2] * vec[second_count + 2] \
				+ matr[7 * i + 3] * vec[second_count + 248] \
				+ matr[7 * i + 4] * vec[second_count + 495];

			second_count++;
		}
		else if (i > 246 && i < 494) {
			res_vec[i] = \
				matr[7 * i] * vec[third_count] \
				+ matr[7 * i + 1] * vec[third_count + 246] \
				+ matr[7 * i + 2] * vec[third_count + 247] \
				+ matr[7 * i + 3] * vec[third_count + 248] \
				+ matr[7 * i + 4] * vec[third_count + 494] \
				+ matr[7 * i + 5] * vec[third_count + 741];

			third_count++;
		}
		else if (i > 493 && i < 134368) {
			res_vec[i] = \
				matr[7 * i] * vec[fourth_count] \
				+ matr[7 * i + 1] * vec[fourth_count + 247] \
				+ matr[7 * i + 2] * vec[fourth_count + 493] \
				+ matr[7 * i + 3] * vec[fourth_count + 494] \
				+ matr[7 * i + 4] * vec[fourth_count + 495] \
				+ matr[7 * i + 5] * vec[fourth_count + 741] \
				+ matr[7 * i + 6] * vec[fourth_count + 988];
			
			fourth_count++;
		}
		else if (i > 134367 && i < 134615) {
			res_vec[i] = \
				matr[7 * i + 1] * vec[fifth_count + 133874] \
				+ matr[7 * i + 2] * vec[fifth_count + 134121] \
				+ matr[7 * i + 3] * vec[fifth_count + 134367] \
				+ matr[7 * i + 4] * vec[fifth_count + 134368] \
				+ matr[7 * i + 5] * vec[fifth_count + 134369] \
				+ matr[7 * i + 6] * vec[fifth_count + 134615];
			
			fifth_count++;
		}
		else if (i > 134614 && i < 134861) {
			res_vec[i] = \
				matr[7 * i + 2] * vec[sixth_count + 134121] \
				+ matr[7 * i + 3] * vec[sixth_count + 134368] \
				+ matr[7 * i + 4] * vec[sixth_count + 134614] \
				+ matr[7 * i + 5] * vec[sixth_count + 134615] \
				+ matr[7 * i + 6] * vec[sixth_count + 134616];
			
			sixth_count++;
		}
		else if (i == 134861) { //last_element_position = 944034
			res_vec[i] = \
				matr[7 * i + 3] * vec[134367] \
				+ matr[7 * i + 4] * vec[134614] \
				+ matr[7 * i + 5] * vec[134860] \
				+ matr[7 * i + 6] * vec[134861];
		}
	}
}

__global__ void GPU_matr_on_vec(double *matr, double *vec, double *res_vec) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	int second_count = 0;
	int third_count = 0;
	int fourth_count = 0;
	int fifth_count = 0;
	int sixth_count = 0;

	for (int i = thread * per_thread; i < (thread + 1) * per_thread; ++i) {
		if (i == 0) {
			res_vec[i] = \
				__dmul_rn(matr[0], vec[0]) \
				+ __dmul_rn(matr[1], vec[1]) \
				+ __dmul_rn(matr[2], vec[247]) \
				+ __dmul_rn(matr[3], vec[494]);
		}
		else if (i > 0 && i < 247) {
			res_vec[i] = \
				__dmul_rn(matr[7 * i],vec[second_count]) \
				+ __dmul_rn(matr[7 * i + 1], vec[second_count + 1]) \
				+ __dmul_rn(matr[7 * i + 2], vec[second_count + 2]) \
				+ __dmul_rn(matr[7 * i + 3], vec[second_count + 248]) \
				+ __dmul_rn(matr[7 * i + 4], vec[second_count + 495]);

			second_count++;
		}
		else if (i > 246 && i < 494) {
			res_vec[i] = \
				__dmul_rn(matr[7 * i], vec[third_count]) \
				+ __dmul_rn(matr[7 * i + 1], vec[third_count + 246]) \
				+ __dmul_rn(matr[7 * i + 2], vec[third_count + 247]) \
				+ __dmul_rn(matr[7 * i + 3], vec[third_count + 248]) \
				+ __dmul_rn(matr[7 * i + 4], vec[third_count + 494]) \
				+ __dmul_rn(matr[7 * i + 5], vec[third_count + 741]);

			third_count++;
		}
		else if (i > 493 && i < 134368) {
			res_vec[i] = \
				__dmul_rn(matr[7 * i], vec[fourth_count]) \
				+ __dmul_rn(matr[7 * i + 1], vec[fourth_count + 247]) \
				+ __dmul_rn(matr[7 * i + 2], vec[fourth_count + 493]) \
				+ __dmul_rn(matr[7 * i + 3], vec[fourth_count + 494]) \
				+ __dmul_rn(matr[7 * i + 4], vec[fourth_count + 495]) \
				+ __dmul_rn(matr[7 * i + 5], vec[fourth_count + 741]) \
				+ __dmul_rn(matr[7 * i + 6], vec[fourth_count + 988]);

			fourth_count++;
		}
		else if (i > 134367 && i < 134615) {
			res_vec[i] = \
				__dmul_rn(matr[7 * i + 1], vec[fifth_count + 133874]) \
				+ __dmul_rn(matr[7 * i + 2], vec[fifth_count + 134121]) \
				+ __dmul_rn(matr[7 * i + 3], vec[fifth_count + 134367]) \
				+ __dmul_rn(matr[7 * i + 4], vec[fifth_count + 134368]) \
				+ __dmul_rn(matr[7 * i + 5], vec[fifth_count + 134369]) \
				+ __dmul_rn(matr[7 * i + 6], vec[fifth_count + 134615]);

			fifth_count++;
		}
		else if (i > 134614 && i < 134861) {
			res_vec[i] = \
				__dmul_rn(matr[7 * i + 2], vec[sixth_count + 134121]) \
				+ __dmul_rn(matr[7 * i + 3], vec[sixth_count + 134368]) \
				+ __dmul_rn(matr[7 * i + 4], vec[sixth_count + 134614]) \
				+ __dmul_rn(matr[7 * i + 5], vec[sixth_count + 134615]) \
				+ __dmul_rn(matr[7 * i + 6], vec[sixth_count + 134616]);

			sixth_count++;
		}
		else if (i == 134861) { //last_element_position = 944034
			res_vec[i] = \
				__dmul_rn(matr[7 * i + 3], vec[134367]) \
				+ __dmul_rn(matr[7 * i + 4], vec[134614]) \
				+ __dmul_rn(matr[7 * i + 5], vec[134860]) \
				+ __dmul_rn(matr[7 * i + 6], vec[134861]);
		}
	}
	/*for (int i = 0; i < 1; ++i) {
		printf("res_vec[i]: %.14f =\n matr[0] %.14f * vec[0] %.14f \n + matr[1] %.14f * vec[1] %.14f \n + matr[2] %.14f * vec[247] %.14f \n + matr[3] %.14f * vec[494] %.14f \n",
			res_vec[i], matr[0], vec[0], matr[1], vec[1], matr[2], vec[247], matr[3], vec[494]);
	}*/
	//printf("res_vec[i]: %.14f =\n matr[0] %.14f * vec[0] %.14f \n + matr[1] %.14f * vec[1] %.14f \n + matr[2] %.14f * vec[247] %.14f \n + matr[3] %.14f * vec[494] %.14f \n",
	//	res_vec[0], matr[0], vec[0], matr[1], vec[1], matr[2], vec[247], matr[3], vec[494]);
}

//-------------------------------------------------------------------------------------------

double vec_on_vec(double *vec_one, double *vec_two) {
	double res = 0;
	for (int i = 0; i < S; ++i) {
		res += vec_one[i] * vec_two[i];
	}
	return res;
}

//__device__ double *res_GPU_vec_on_vec;

__global__ void GPU_vec_on_vec(double *vec_one, double *vec_two, double *res) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	double tmp_res_per_thread = 0;
	*res = 0;

	for (int i = thread * per_thread; i < (thread + 1) * per_thread; ++i) {
		tmp_res_per_thread += __dmul_rn(vec_one[i], vec_two[i]);
	}
	atomicAdd(res, tmp_res_per_thread);
	//printf("%.14f = %.14f = %.14f + %.14f\n", *res, tmp_res_per_thread, vec_one[0], vec_two[0]);
}

//-------------------------------------------------------------------------------------------

double norm_vec(double *vec) {
	double res = 0;
	for (int i = 0; i < S; ++i) {
		res += pow(vec[i], 2);
	}
	return sqrt(res);
}

//__device__ double *res_GPU_norm_vec;

__global__ void GPU_norm_vec(double *vec, double *res) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	double tmp_res_per_thread = 0;
	*res = 0;

	for (int i = thread * per_thread; i < (thread + 1) * per_thread; ++i) {
		//tmp_res_per_thread += powf(vec[i], 2);
		*res += __dmul_rn(vec[i], vec[i]); // vec[i]* vec[i];
		/*if (i > S/2 && i < S/2 + 10) {
			printf("%.14f += %.14f, ^2: %.14f\n", *res, vec[i], vec[i] * vec[i]);
		}*/
	}
	*res = __dsqrt_ru(*res);
	//atomicAdd(res, tmp_res_per_thread);
	//printf("%.14f += %.14f, ^2: %.14f\n", *res, vec[S - 1], vec[S - 1] * vec[S - 1]);
}

//-------------------------------------------------------------------------------------------

void copy_vec(double *matr_one, double *matr_two) {
	for (int i = 0; i < S; ++i) {
		matr_one[i] = matr_two[i];
	}
}

__global__ void GPU_copy_vec(double *matr_one, double *matr_two) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread * per_thread; i < (thread + 1) * per_thread; ++i) {
		matr_one[i] = matr_two[i];
	}
	//printf("%.14f\n", matr_one[0]);
}

//-------------------------------------------------------------------------------------------

void nullify(double *vec) {
	for (int i = 0; i < S; ++i) {
		vec[i] = 0;
	}
}

__global__ void GPU_nullify(double *vec) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread * per_thread; i < (thread + 1) * per_thread; ++i) {
		vec[i] = 0;
	}
}

//-------------------------------------------------------------------------------------------

//void check_stop(double *A, double *F, double *X, double result) {
//	double *stop_up = new double[1];
//	double *stop_down = new double[1];
//	double *stop_vec = new double[S];
//	double *stop_tmp = new double[S];
//
//	stop_up[0] = stop_down[0] = 0;
//
//	//host
//	/*matr_on_vec(A, X, stop_tmp);
//	dif_vec(stop_tmp, F, stop_vec);
//	stop_up = norm_vec(stop_vec);
//	stop_down = norm_vec(F);*/
//
//	//device
//	double *devA = NULL;
//	double *devF = NULL;
//	double *devX = NULL;
//	double *dev_stop_tmp = NULL;
//	double *dev_stop_vec = NULL;
//	double *dev_stop_up = NULL;
//	double *dev_stop_down = NULL;
//
//	CUDA_CALL(cudaMalloc(&devA, S * 7 * sizeof(double)));
//	CUDA_CALL(cudaMalloc(&devF, S * sizeof(double)));
//	CUDA_CALL(cudaMalloc(&devX, S * sizeof(double)));
//	CUDA_CALL(cudaMalloc(&dev_stop_tmp, S * sizeof(double)));
//	CUDA_CALL(cudaMalloc(&dev_stop_vec, S * sizeof(double)));
//	CUDA_CALL(cudaMalloc(&dev_stop_up, 1 * sizeof(double)));
//	CUDA_CALL(cudaMalloc(&dev_stop_down, 1 * sizeof(double)));
//
//	CUDA_CALL(cudaMemcpy(devA, A, S * 7 * sizeof(double), cudaMemcpyHostToDevice));
//	CUDA_CALL(cudaMemcpy(devF, F, S * sizeof(double), cudaMemcpyHostToDevice));
//	CUDA_CALL(cudaMemcpy(devX, X, S * sizeof(double), cudaMemcpyHostToDevice));
//	CUDA_CALL(cudaMemcpy(dev_stop_tmp, stop_tmp, S * sizeof(double), cudaMemcpyHostToDevice));
//	CUDA_CALL(cudaMemcpy(dev_stop_vec, stop_vec, S * sizeof(double), cudaMemcpyHostToDevice));
//	CUDA_CALL(cudaMemcpy(dev_stop_up, stop_up, 1 * sizeof(double), cudaMemcpyHostToDevice));
//	CUDA_CALL(cudaMemcpy(dev_stop_down, stop_down, 1 * sizeof(double), cudaMemcpyHostToDevice));
//
//	GPU_matr_on_vec<<<block_size, grid_size>>>(devA, devX, dev_stop_tmp);
//	CUDA_CALL(cudaDeviceSynchronize());
//
//	GPU_dif_vec<<<block_size, grid_size>>>(dev_stop_tmp, devF, dev_stop_vec);
//	CUDA_CALL(cudaDeviceSynchronize());
//
//	//CUDA_CALL(cudaMemset(res_GPU_norm_vec, 0, sizeof(double)));
//	GPU_norm_vec<<<block_size, grid_size>>>(dev_stop_vec, dev_stop_up);
//	CUDA_CALL(cudaDeviceSynchronize());
//	//CUDA_CALL(cudaMemcpyFromSymbol(&stop_up, "res_GPU_norm_vec", sizeof(double), 0, cudaMemcpyDeviceToHost));
//	CUDA_CALL(cudaMemcpy(stop_up, dev_stop_up, 1 * sizeof(double), cudaMemcpyDeviceToHost));
//	stop_up[0] = sqrt(stop_up[0]);
//
//	//CUDA_CALL(cudaMemset(res_GPU_norm_vec, 0, sizeof(double)));
//	GPU_norm_vec<<<block_size, grid_size>>>(devF, dev_stop_down);
//	CUDA_CALL(cudaDeviceSynchronize());
//	//CUDA_CALL(cudaMemcpyFromSymbol(&stop_down, "res_GPU_norm_vec", sizeof(double), 0, cudaMemcpyDeviceToHost));
//	CUDA_CALL(cudaMemcpy(stop_down, dev_stop_down, 1 * sizeof(double), cudaMemcpyDeviceToHost));
//	stop_down[0] = sqrt(stop_down[0]);
//
//	CUDA_CALL(cudaGetLastError());
//
//	cudaFree(devA);
//	cudaFree(devF);
//	cudaFree(devX);
//	cudaFree(dev_stop_tmp);
//	cudaFree(dev_stop_vec);
//	cudaFree(dev_stop_up);
//	cudaFree(dev_stop_down);
//
//	delete(stop_vec);
//	delete(stop_tmp);
//	delete(stop_up);
//	delete(stop_down);
//
//
//	result = (double)(stop_up[0] / stop_down[0]);
//}

//double check_stop2(double *A, double *F, double *X) {
//	double stop_up = 0;
//	double stop_down = 0;
//	double *stop_vec = new double[S];
//	double *stop_tmp = new double[S];
//
//	matr_on_vec(A, X, stop_tmp);
//	dif_vec(stop_tmp, F, stop_vec);
//	stop_up = norm_vec(stop_vec);
//	stop_down = norm_vec(F);
//
//	delete(stop_vec);
//	delete(stop_tmp);
//
//	return stop_up / stop_down;
//}

__global__ void GPU_ak_bk(double *up, double *down, double *ak_bk) {
	//printf("up and down: %f, %f\n", *up, *down);
	//*ak_bk = __ddiv_rn(*up, *down);
	*ak_bk = __ddiv_ru(*up, *down);
	//printf("ak_bk: %.16f\n", *ak_bk);
}

//__device__ double GPU_stop;

__global__ void GPU_check_nev(double *up, double *down, double *eps, double Eps, double *x_res) {
	*eps = __ddiv_ru(*up, *down);
	printf("Nev: %.16f %s 0.1\n\n", *eps, ((*eps < Eps) ? "<" : ">"));
	if (*eps < Eps) {
		return;
	}
}

void CGMR(double *A, double *F, clock_t begin_algo) {
	const double Eps = 0.001;

	double *x = new double[S];
	double *r = new double[S];
	double *p = new double[S];

	//coef in cycle
	double ak = 0;
	double bk = 0;

	//const coef a
	double alpha = 0;

	//tmp
	double *tmp = new double[S];
	double *rApk = new double[S];
	double *stop = new double[S];
	double *r_k1 = new double[S];
	double *x_k1 = new double[S];
	double *p_k1 = new double[S];
	double up = 0;
	double down = 0;
	
	int count_tmp = 0;

	//to device from arguments
	double *dev_A = NULL;
	double *dev_F = NULL;

	double *dev_x = NULL;
	double *dev_r = NULL;
	double *dev_p = NULL;
		
	double *dev_ak = NULL;
	double *dev_bk;

	double *dev_tmp = NULL;
	double *dev_rApk = NULL;
	double *dev_r_k1 = NULL;
	double *dev_x_k1 = NULL;
	double *dev_p_k1 = NULL;
	double *dev_up = 0;
	double *dev_down = 0;

	//for stop_deal
	double stop_up = 0;
	double stop_down = 0;
	double stop_eps = 1;
	double *stop_vec = new double[S];
	double *stop_tmp = new double[S];

	double *dev_stop_up;
	double *dev_stop_down;
	double *dev_stop_eps;
	double *dev_stop_tmp = NULL;
	double *dev_stop_vec = NULL;

	CUDA_CALL(cudaMalloc(&dev_A, S * 7 * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_F, S * sizeof(double)));

	CUDA_CALL(cudaMalloc(&dev_x, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_r, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_p, S * sizeof(double)));

	CUDA_CALL(cudaMalloc(&dev_ak, sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_bk, sizeof(double)));

	CUDA_CALL(cudaMalloc(&dev_tmp, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_rApk, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_r_k1, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_x_k1, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_p_k1, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_up, sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_down, sizeof(double)));

	CUDA_CALL(cudaMalloc(&dev_stop_up, sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_stop_down, sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_stop_eps, sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_stop_tmp, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_stop_vec, S * sizeof(double)));

	//fill all arrays with zeroes
	clock_t begin_zero_filling = clock();
	for (int i = 0; i < S; ++i) {
		x[i] = r[i] = p[i] = rApk[i] = tmp[i] = r_k1[i] = x_k1[i] = p_k1[i] = 0;
	}
	
	CUDA_CALL(cudaMemcpy(dev_A, A, S * 7 * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_F, F, S * sizeof(double), cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMemcpy(dev_x, x, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_r, r, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_p, p, S * sizeof(double), cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMemcpy(dev_ak, &ak, sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_bk, &bk, sizeof(double), cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMemcpy(dev_rApk, rApk, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_r_k1, r_k1, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_x_k1, x_k1, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_p_k1, p_k1, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_tmp, tmp, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_up, &up, sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_down, &down, sizeof(double), cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMemcpy(dev_stop_up, &stop_up, sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_stop_down, &stop_down, sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_stop_eps, &stop_eps, sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_stop_tmp, stop_tmp, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_stop_vec, stop_vec, S * sizeof(double), cudaMemcpyHostToDevice));
	
	//printf("Device Variable Copying:\t%s\n", cudaGetErrorString(cudaGetLastError()));

	clock_t end_zero_filling = clock();

	double z_time = double(end_zero_filling - begin_zero_filling) / CLOCKS_PER_SEC;
	cout << "Cuda.Malloc and Cuda.Memcpy completed in: " << z_time << endl;

	clock_t begin_make_prep = clock();

	//Подготовка перед алгоритмом
	
	//making const rA
	//rA = A + alpha*E
	
	GPU_mult_A_on_alpha_E<<<block_size, grid_size>>>(dev_A, alpha);
	CUDA_CALL(cudaDeviceSynchronize());
	
	//r = b-rA*x0	|rA*x0 = 0| |r = b|
	GPU_copy_vec<<<block_size, grid_size>>>(dev_r, dev_F);
	CUDA_CALL(cudaDeviceSynchronize());
	
	//p = r;
	GPU_copy_vec<<<block_size, grid_size>>>(dev_p, dev_r);
	CUDA_CALL(cudaDeviceSynchronize());

	clock_t end_make_prep = clock();

	double p_time = double(end_make_prep - begin_make_prep) / CLOCKS_PER_SEC;
	cout << "Before cycle preperations completed in: " << p_time << endl << endl;

	// ! УБРАТЬ ВСЕ ПЕРЕХОДЫ НА ПРОЦ С ВИДЕОКАРТЫ В ЦИКЛЕ !

	while (!(stop_eps < Eps) && count_tmp < S) {

		count_tmp++;
		cout << "Iteration: " << count_tmp << " / " << S << endl;

		//cout << "Start of iter" << endl;
		clock_t begin_CGMR = clock();

		//-------------------------------------------------------------------------------------------

		//ak = norm(r_k)^2 / (rA*p_k, p_k)
		//norm(r_k)^2
		//norm = vec_on_vec(r, r);

		GPU_vec_on_vec<<<block_size, grid_size>>>(dev_r, dev_r, dev_up);
		//CUDA_CALL(cudaDeviceSynchronize());
		
		//-------------------------------------------------------------------------------------------

		//rA*p_k
		GPU_matr_on_vec<<<block_size, grid_size>>>(dev_A, dev_p, dev_rApk);
		//CUDA_CALL(cudaDeviceSynchronize());

		//(rA*p_k, p_k)

		GPU_vec_on_vec<<<block_size, grid_size>>>(dev_rApk, dev_p, dev_down);
		//CUDA_CALL(cudaDeviceSynchronize());

		//ak = ...
		GPU_ak_bk<<<block_size, grid_size>>>(dev_up, dev_down, dev_ak);
		//CUDA_CALL(cudaDeviceSynchronize());
		//-------------------------------------------------------------------------------------------

		//r_k+1 = r_k - ak*rA*p_k
		//ak*(rA*p_k)
		
		GPU_mult_vec_on_num<<<block_size, grid_size>>>(dev_rApk, dev_ak, dev_tmp);
		//CUDA_CALL(cudaDeviceSynchronize());

		//r_k+1 = r_k - ...
		GPU_dif_vec<<<block_size, grid_size>>>(dev_r, dev_tmp, dev_r_k1);
		//CUDA_CALL(cudaDeviceSynchronize());

		//-------------------------------------------------------------------------------------------

		//nullify(tmp);
		GPU_nullify<<<block_size, grid_size>>>(dev_tmp);
		//CUDA_CALL(cudaDeviceSynchronize());

		//x_k+1 = x_k + ak*p_k
		//ak*p_k
		GPU_mult_vec_on_num<<<block_size, grid_size>>>(dev_p, dev_ak, dev_tmp);
		//CUDA_CALL(cudaDeviceSynchronize());

		//x_k+1 = x_k + ...
		GPU_sum_vec<<<block_size, grid_size>>>(dev_x, dev_tmp, dev_x_k1);
		//CUDA_CALL(cudaDeviceSynchronize());

		//-------------------------------------------------------------------------------------------

		//bk = norm(r_k+1)^2 / norm(r_k)^2
		//norm(r_k+1)^2
		GPU_vec_on_vec<<<block_size, grid_size>>>(dev_r_k1, dev_r_k1, dev_up);
		//CUDA_CALL(cudaDeviceSynchronize());
		
		//norm(r_k)^2
		GPU_vec_on_vec<<<block_size, grid_size>>>(dev_r, dev_r, dev_down);
		//CUDA_CALL(cudaDeviceSynchronize());

		//bk = ...
		GPU_ak_bk<<<1, 1>>>(dev_up, dev_down, dev_bk);
		//CUDA_CALL(cudaDeviceSynchronize());

		//-------------------------------------------------------------------------------------------

		//nullify(tmp);

		GPU_nullify<<<block_size, grid_size>>>(dev_tmp);
		//CUDA_CALL(cudaDeviceSynchronize());

		//p_k+1 = r_k+1 + bk*p_k
		//bk*p_k

		GPU_mult_vec_on_num<<<block_size, grid_size>>>(dev_p, dev_bk, dev_tmp);
		//CUDA_CALL(cudaDeviceSynchronize());

		//p_k+1 = r_k+1 + ...

		GPU_sum_vec<<<block_size, grid_size>>>(dev_r_k1, dev_tmp, dev_p_k1);
		//CUDA_CALL(cudaDeviceSynchronize());

		//-------------------------------------------------------------------------------------------
		
		//CUDA_CALL(cudaDeviceSynchronize());
		GPU_show_vec<<<block_size, grid_size>>>(dev_x_k1, 1);
		
		//-------------------------------------------------------------------------------------------

		//check_stop(A, F, x_k1, stop_eps);
		
		GPU_matr_on_vec<<<block_size, grid_size>>>(dev_A, dev_x_k1, dev_stop_tmp);
		//CUDA_CALL(cudaDeviceSynchronize());

		GPU_dif_vec<<<block_size, grid_size>>>(dev_stop_tmp, dev_F, dev_stop_vec);
		//CUDA_CALL(cudaDeviceSynchronize());
		
		GPU_norm_vec<<<block_size, grid_size>>>(dev_stop_vec, dev_stop_up);
		//CUDA_CALL(cudaDeviceSynchronize());

		GPU_norm_vec<<<block_size, grid_size>>>(dev_F, dev_stop_down);
		//CUDA_CALL(cudaDeviceSynchronize());

		GPU_check_nev<<<block_size, grid_size>>>(dev_stop_up, dev_stop_down, dev_stop_eps, Eps, dev_x_k1);
		//CUDA_CALL(cudaDeviceSynchronize());
		
		//CUDA_CALL(cudaMemcpy(&stop_eps, dev_stop_eps, sizeof(double), cudaMemcpyDeviceToHost));
		//CUDA_CALL(cudaDeviceSynchronize());
		//add_in_file(stop_eps);
		
		if (stop_eps < Eps || count_tmp > S - 1) {
			CUDA_CALL(cudaMemcpy(x_k1, dev_x_k1, S * sizeof(double), cudaMemcpyDeviceToHost));
			CUDA_CALL(cudaDeviceSynchronize());
			write_in_file(x_k1, S, "X_1.dat");
		}

		//-------------------------------------------------------------------------------------------

		//copy for next iter
		//p = p_k1;
		GPU_copy_vec<<<block_size, grid_size>>>(dev_p, dev_p_k1);
		//CUDA_CALL(cudaDeviceSynchronize());

		//x = x_k1;
		GPU_copy_vec<<<block_size, grid_size>>>(dev_x, dev_x_k1);
		//CUDA_CALL(cudaDeviceSynchronize());

		//r = r_k1;
		GPU_copy_vec<<<block_size, grid_size>>>(dev_r, dev_r_k1);
		//CUDA_CALL(cudaDeviceSynchronize());

		//clear vectors
		//nullify(stop);
		//nullify(p_k1);
		//nullify(x_k1);
		//nullify(r_k1);
		//nullify(tmp);
		//nullify(rApk);

		//GPU_nullify<<<block_size, grid_size>>>(dev_stop);
		//CUDA_CALL(cudaDeviceSynchronize());
		
		//GPU_nullify<<<block_size, grid_size>>>(dev_p_k1);
		//CUDA_CALL(cudaDeviceSynchronize());

		//GPU_nullify<<<block_size, grid_size>>>(dev_x_k1);
		//CUDA_CALL(cudaDeviceSynchronize());

		//GPU_nullify<<<block_size, grid_size>>>(dev_r_k1);
		//CUDA_CALL(cudaDeviceSynchronize());

		//GPU_nullify<<<block_size, grid_size>>>(dev_tmp);
		//CUDA_CALL(cudaDeviceSynchronize());

		//GPU_nullify<<<block_size, grid_size>>>(dev_rApk);
		//CUDA_CALL(cudaDeviceSynchronize());
		
		//-------------------------------------------------------------------------------------------

		clock_t end_CGMR = clock();

		double CGMR_time = double(end_CGMR - begin_CGMR) / CLOCKS_PER_SEC;
		cout << "Runtime of iter: " << CGMR_time << endl << endl;

		clock_t end_algo = clock();
		double algo_time = double(end_algo - begin_algo) / CLOCKS_PER_SEC;
		//cout << "Runtime of algo: " << algo_time << endl;
	}
	CUDA_CALL(cudaMemcpy(x_k1, dev_x_k1, S * sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaDeviceSynchronize());
	write_in_file(x_k1, S, "X_1.dat");

	delete(x);
	delete(r);
	delete(p);
	delete(x_k1);
	delete(r_k1);
	delete(p_k1);
	delete(tmp);
	delete(rApk);
	delete(stop);
	delete(stop_vec);
	delete(stop_tmp);

	cudaFree(dev_x);
	cudaFree(dev_r);
	cudaFree(dev_p);
	cudaFree(dev_x_k1);
	cudaFree(dev_r_k1);
	cudaFree(dev_p_k1);
	cudaFree(dev_tmp);
	cudaFree(dev_rApk);
	cudaFree(dev_A);
	cudaFree(dev_F);
	cudaFree(dev_ak);
	cudaFree(dev_bk);
	cudaFree(dev_up);
	cudaFree(dev_down);
	cudaFree(dev_stop_up);
	cudaFree(dev_stop_down);
	cudaFree(dev_stop_eps);
	cudaFree(dev_stop_tmp);
	cudaFree(dev_stop_vec);
}

int main() {
	clock_t begin_algo = clock();

	double *matr_A = new double[S * 7];
	double *matr_F = new double[S];

	ifstream A;
	ifstream F;

	clock_t begin = clock();

	A.open("A_with_01.dat");				// 134862 * 7	| result.dat
	F.open("F_new.dat");					// 134862		| F.dat

	if (A.is_open() && F.is_open()) {
		create_matr(A, matr_A, S * 7);
		create_matr(F, matr_F, S);
	}
	else {
		cout << "Error opening files" << endl;
	}

	A.close();
	F.close();

	clear_nevyazka();

	CGMR(matr_A, matr_F, begin_algo);

	clock_t end = clock();
	double time = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Runtime: " << time << endl;

	delete(matr_A);
	delete(matr_F);

	//system("pause");

	return 0;

	//На процессоре
	//last runtime: 847.723 sec.
	//last runtime: 833.092 sec. / Iteration: 29328
	//last runtime: 792.424 sec. / Iteration: 25619
	//last runtime: 2483.177 sec. / Iteration: 83403 / Eps = 0.001
	//last runtime: 2375.798 sec. / Iteration: 83403 / Eps = 0.001
	//last runtime: 1644.729 sec. / Iteration: 53317 / Eps = 0.01
	//last runtime: 740.28 sec. / Iteration: 25619 / Eps = 0.1
	//last runtime: 254.525 sec. / Iteration: 8627 / Eps = 0.1 / Alpha = 0.5
	//last runtime: ~369 sec. / Iteration: ? / Eps = 0.01 / Alpha = 0.5
	//last runtime: 496.967 sec. / Iteration: 17601 / Eps = 0.001 / Alpha = 0.5
	//last runtime: 564.746 sec. / Iteration: 18385 / Eps = 0.1 / Alpha = 0.05
	//last runtime: 438.88 sec. / Iteration: 15267 / Eps = 0.1 / Alpha = 0.1
	//last runtime: 2375.026 sec. / Iteration: 76618 / Eps = 0.00001 / Alpha = 0.05
	//last runtime: 3773.602 sec. / Iteration: 134862+ / Eps = 0.00001 / Alpha = 0.01
	//with F_new.dat
	//last runtime: 541.725 sec. / Iteration: 19328 / Eps = 0.00001 / Alpha = 0
	//last runtime: 3785.569 sec. / Iteration: 134862+ / Eps = 0.00000001 / Alpha = 0
	//with F_new.dat and A_with_minus_and_01 - не стоит использовать с минусом матрицу А
	//last runtime: 1571.473 sec. / Iteration: 43674 / Eps = 0.001 / Alpha = 0

	//! Посчитать с Eps = 10^-8 но без ограничения по количеству итераций !

	//На видеокарте
	//last runtime: 47.425 sec. / Iteration: 3 / Eps = 0.1 / Alpha = 0
}