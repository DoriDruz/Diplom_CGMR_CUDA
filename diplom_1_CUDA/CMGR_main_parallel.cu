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
const int grid_size = 14;

const int per_thread = S / grid_size * block_size;

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

void show_vec(double *matr, double size) {
	for (int i = 0; i < size; ++i) {
		cout << setprecision(16) << matr[i] << "; ";
	}
	cout << endl;
}

//-------------------------------------------------------------------------------------------

void mult_vec_on_num(double *vec, double num, double *res_vec) {
	for (int i = 0; i < S; ++i) {
		res_vec[i] = vec[i] * num;
	}
}

__global__ void GPU_mult_vec_on_num(double *vec, double num, double *res_vec) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread * per_thread; i < (thread + 1) * per_thread; ++i) {
		res_vec[i] = vec[i] * num;
	}
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

//-------------------------------------------------------------------------------------------

void sum_vec(double *vec_one, double *vec_two, double *res_vec) {
	for (int i = 0; i < S; ++i) {
		res_vec[i] = vec_one[i] + vec_two[i];
	}
}

__global__ void GPU_sum_vec(double *vec_one, double *vec_two, double *res_vec) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread * per_thread; i < (thread + 1) * per_thread; ++i) {
		res_vec[i] = vec_one[i] + vec_two[i];
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
		res_vec[i] = vec_one[i] - vec_two[i];
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

//-------------------------------------------------------------------------------------------

double vec_on_vec(double *vec_one, double *vec_two) {
	double res = 0;
	for (int i = 0; i < S; ++i) {
		res += vec_one[i] * vec_two[i];
	}
	return res;
}

//__device__ double *res_GPU_vec_on_vec; //! не забыть про host > dev / dev > host !

__global__ void GPU_vec_on_vec(double *vec_one, double *vec_two, double *res) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	double tmp_res_per_thread = 0;
	
	for (int i = thread * per_thread; i < (thread + 1) * per_thread; ++i) {
		tmp_res_per_thread += vec_one[i] * vec_two[i];
	}

	atomicAdd(res, tmp_res_per_thread);
}

//-------------------------------------------------------------------------------------------

double norm_vec(double *vec) {
	double res = 0;
	for (int i = 0; i < S; ++i) {
		res += pow(vec[i], 2);
	}
	return sqrt(res);
}

//__device__ double *res_GPU_norm_vec; //! не забыть про host > dev / dev > host ! и sqrt!

__global__ void GPU_norm_vec(double *vec, double *res) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	double tmp_res_per_thread = 0;

	for (int i = thread * per_thread; i < (thread + 1) * per_thread; ++i) {
		tmp_res_per_thread += powf(vec[i], 2);
	}
	atomicAdd(res, tmp_res_per_thread);
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

__global__ void GPU_null_one(double *vec) {
	vec = NULL;
}

//-------------------------------------------------------------------------------------------

double check_stop(double *A, double *F, double *X) {
	double *stop_up = new double[1];
	double *stop_down = new double[1];
	double *stop_vec = new double[S];
	double *stop_tmp = new double[S];

	stop_up[0] = stop_down[0] = 0;

	//host
	/*matr_on_vec(A, X, stop_tmp);
	dif_vec(stop_tmp, F, stop_vec);
	stop_up = norm_vec(stop_vec);
	stop_down = norm_vec(F);*/

	//device
	double *devA = NULL;
	double *devF = NULL;
	double *devX = NULL;
	double *dev_stop_tmp = NULL;
	double *dev_stop_vec = NULL;
	double *dev_stop_up = NULL;
	double *dev_stop_down = NULL;

	CUDA_CALL(cudaMalloc(&devA, S * 7 * sizeof(double)));
	CUDA_CALL(cudaMalloc(&devF, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&devX, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_stop_tmp, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_stop_vec, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_stop_up, 1 * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_stop_down, 1 * sizeof(double)));

	CUDA_CALL(cudaMemcpy(devA, A, S * 7 * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(devF, F, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(devX, X, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_stop_tmp, stop_tmp, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_stop_vec, stop_vec, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_stop_up, stop_up, 1 * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_stop_down, stop_down, 1 * sizeof(double), cudaMemcpyHostToDevice));

	GPU_matr_on_vec<<<block_size, grid_size>>>(devA, devX, dev_stop_tmp);
	//CUDA_CALL(cudaDeviceSynchronize());

	GPU_dif_vec<<<block_size, grid_size>>>(dev_stop_tmp, devF, dev_stop_vec);
	//CUDA_CALL(cudaDeviceSynchronize());

	//CUDA_CALL(cudaMemset(res_GPU_norm_vec, 0, sizeof(double)));
	GPU_norm_vec<<<block_size, grid_size>>>(dev_stop_vec, dev_stop_up);
	//CUDA_CALL(cudaDeviceSynchronize());
	//CUDA_CALL(cudaMemcpyFromSymbol(&stop_up, "res_GPU_norm_vec", sizeof(double), 0, cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(stop_up, dev_stop_up, 1 * sizeof(double), cudaMemcpyDeviceToHost));
	stop_up[0] = sqrt(stop_up[0]);

	//CUDA_CALL(cudaMemset(res_GPU_norm_vec, 0, sizeof(double)));
	GPU_norm_vec<<<block_size, grid_size>>>(devF, dev_stop_down);
	//CUDA_CALL(cudaDeviceSynchronize());
	//CUDA_CALL(cudaMemcpyFromSymbol(&stop_down, "res_GPU_norm_vec", sizeof(double), 0, cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(stop_down, dev_stop_down, 1 * sizeof(double), cudaMemcpyDeviceToHost));
	stop_down[0] = sqrt(stop_down[0]);

	CUDA_CALL(cudaGetLastError());

	cudaFree(devA);
	cudaFree(devF);
	cudaFree(devX);
	cudaFree(dev_stop_tmp);
	cudaFree(dev_stop_vec);
	cudaFree(dev_stop_up);
	cudaFree(dev_stop_down);

	delete(stop_vec);
	delete(stop_tmp);
	delete(stop_up);
	delete(stop_down);


	return stop_up[0] / stop_down[0];
}

void CGMR(double *A, double *F, clock_t begin_algo) {
	const double Eps = 0.1;

	//to device from arguments
	double *dev_A = NULL;
	double *dev_F = NULL;

	CUDA_CALL(cudaMalloc(&dev_A, S * 7 * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_F, S * sizeof(double)));

	CUDA_CALL(cudaMemcpy(dev_A, A, S * 7 * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_F, F, S * sizeof(double), cudaMemcpyHostToDevice));

	//double *rA = new double[S * 7];
	double *x = new double[S];
	double *r = new double[S];
	double *p = new double[S];

	double *dev_x = NULL;
	double *dev_r = NULL;
	double *dev_p = NULL;

	CUDA_CALL(cudaMalloc(&dev_x, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_r, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_p, S * sizeof(double)));

	//original A
	double *or_A = new double[S * 7];

	double *dev_or_A = NULL;

	CUDA_CALL(cudaMalloc(&dev_or_A, S * 7 * sizeof(double)));

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
	double stop_norm = 0;
	double stop_norm2 = 0;
	double *norm = new double[1];
	double *norm2 = new double[1];
	double *vov = new double[1];
	double stop_eps = 1;

	norm[0] = norm2[0] = vov[0] = 0;

	int count_tmp = 0;
	
	double *dev_tmp = NULL;
	double *dev_rApk = NULL;
	double *dev_stop = NULL;
	double *dev_r_k1 = NULL;
	double *dev_x_k1 = NULL;
	double *dev_p_k1 = NULL;
	double *dev_norm = NULL;
	double *dev_norm2 = NULL;
	double *dev_vov = NULL;

	CUDA_CALL(cudaMalloc(&dev_tmp, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_rApk, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_stop, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_r_k1, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_x_k1, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_p_k1, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_norm, 1 * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_norm2, 1 * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_vov, 1 * sizeof(double)));

	//clear file with nevyazka
	ofstream nevyazka_file;
	nevyazka_file.open("nevyazka.dat", ofstream::trunc);
	nevyazka_file.close();

	//clear file with debug
	ofstream debug_f;
	debug_f.open("debug.dat", ofstream::trunc);
	debug_f.close();

	//ЗАНУЛИТЬ or_A?

	//fill all arrays with zeroes
	clock_t begin_zero_filling = clock();
	for (int i = 0; i < S; ++i) {
		x[i] = r[i] = p[i] = 0;
		rApk[i] = stop[i] = tmp[i] = r_k1[i] = x_k1[i] = p_k1[i] = 0;
	}
	
	CUDA_CALL(cudaMemcpy(dev_x, x, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_r, r, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_p, p, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_rApk, rApk, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_stop, stop, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_r_k1, r_k1, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_x_k1, x_k1, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_p_k1, p_k1, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_tmp, tmp, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_norm, norm, 1 * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_norm2, norm2, 1 * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_vov, vov, 1 * sizeof(double), cudaMemcpyHostToDevice));

	clock_t end_zero_filling = clock();

	double z_time = double(end_zero_filling - begin_zero_filling) / CLOCKS_PER_SEC;
	cout << "Zero_filling_preperation: " << z_time << endl;

	clock_t begin_make_prep = clock();

	//Подготовка перед алгоритмом

	//copy A in or_A
	//copy_matr(or_A, A);
	
	GPU_copy_matr<<<block_size, grid_size>>>(dev_or_A, dev_A);
	//CUDA_CALL(cudaDeviceSynchronize());

	//making const rA
	//rA = A + alpha*E
	//mult_A_on_alpha_E(A, alpha);

	GPU_mult_A_on_alpha_E<<<block_size, grid_size>>>(dev_A, alpha);
	//CUDA_CALL(cudaDeviceSynchronize());

	//r = b-rA*x0	|rA*x0 = 0| |r = b|
	//copy_vec(r, F);

	GPU_copy_vec<<<block_size, grid_size>>>(dev_r, dev_F);
	//CUDA_CALL(cudaDeviceSynchronize());

	//p = r;
	//copy_vec(p, r);

	GPU_copy_vec<<<block_size, grid_size>>>(dev_p, dev_r);
	//CUDA_CALL(cudaDeviceSynchronize());

	clock_t end_make_prep = clock();

	double p_time = double(end_make_prep - begin_make_prep) / CLOCKS_PER_SEC;
	cout << "Preperation rA, r, p: " << p_time << endl << endl;

	//WHY NEVYAZKA INCREASING ? - Different data in arrays? Make zero iter out of cycle?

	while (!(stop_eps < Eps) && count_tmp < S) {

		count_tmp++;
		cout << "Iteration: " << count_tmp << " / " << S << endl;

		//cout << "Start of iter" << endl;
		clock_t begin_CGMR = clock();

		//-------------------------------------------------------------------------------------------

		//ak = norm(r_k)^2 / (rA*p_k, p_k)
		//norm(r_k)^2
		//norm = pow(norm_vec(r), 2);
		//norm = vec_on_vec(r, r);

		//CUDA_CALL(cudaMemset(res_GPU_vec_on_vec, 0, sizeof(double)));
		GPU_null_one<<<1,1>>>(dev_norm);
		GPU_vec_on_vec<<<block_size, grid_size>>>(dev_r, dev_r, dev_norm);
		//CUDA_CALL(cudaDeviceSynchronize());
		CUDA_CALL(cudaMemcpy(norm, dev_norm, 1 * sizeof(double), cudaMemcpyDeviceToHost));
		//CUDA_CALL(cudaMemcpyFromSymbol(&norm, "res_GPU_vec_on_vec", sizeof(double), 0, cudaMemcpyDeviceToHost));
		
		//-------------------------------------------------------------------------------------------

		//rA*p_k
		//matr_on_vec(A, p, rApk);

		GPU_matr_on_vec<<<block_size, grid_size>>>(dev_A, dev_p, dev_rApk);
		//CUDA_CALL(cudaDeviceSynchronize());

		//(rA*p_k, p_k)
		//vov = vec_on_vec(rApk, p);

		//CUDA_CALL(cudaMemset(res_GPU_vec_on_vec, 0, sizeof(double)));
		GPU_null_one<<<1, 1>>>(dev_vov);
		GPU_vec_on_vec<<<block_size, grid_size>>>(dev_rApk, dev_p, dev_vov);
		//CUDA_CALL(cudaDeviceSynchronize());
		CUDA_CALL(cudaMemcpy(vov, dev_vov, 1 * sizeof(double), cudaMemcpyDeviceToHost));
		//CUDA_CALL(cudaMemcpyFromSymbol(&vov, "res_GPU_vec_on_vec", sizeof(double), 0, cudaMemcpyDeviceToHost));

		//ak = ...
		ak = (isnan(norm[0] / vov[0]) ? 0 : (norm[0] / vov[0]));

		//-------------------------------------------------------------------------------------------

		//r_k+1 = r_k - ak*rA*p_k
		//ak*(rA*p_k)
		//mult_vec_on_num(rApk, ak, tmp);

		GPU_mult_vec_on_num<<<block_size, grid_size>>>(dev_rApk, ak, dev_tmp);
		//CUDA_CALL(cudaDeviceSynchronize());

		//r_k+1 = r_k - ...
		//dif_vec(r, tmp, r_k1);

		GPU_dif_vec<<<block_size, grid_size>>>(dev_r, dev_tmp, dev_r_k1);
		//CUDA_CALL(cudaDeviceSynchronize());

		//-------------------------------------------------------------------------------------------

		//nullify(tmp);

		GPU_nullify<<<block_size, grid_size>>>(dev_tmp);
		//CUDA_CALL(cudaDeviceSynchronize());

		//x_k+1 = x_k + ak*p_k
		//ak*p_k
		//mult_vec_on_num(p, ak, tmp);

		GPU_mult_vec_on_num<<<block_size, grid_size>>>(dev_p, ak, dev_tmp);
		//CUDA_CALL(cudaDeviceSynchronize());

		//x_k+1 = x_k + ...
		//sum_vec(x, tmp, x_k1);

		GPU_sum_vec<<<block_size, grid_size>>>(dev_x, dev_tmp, dev_x_k1);
		//CUDA_CALL(cudaDeviceSynchronize());

		//-------------------------------------------------------------------------------------------

		//bk = norm(r_k+1)^2 / norm(r_k)^2
		//norm(r_k+1)^2
		//norm = pow(norm_vec(r_k1), 2);
		//norm = vec_on_vec(r_k1, r_k1);

		//CUDA_CALL(cudaMemset(res_GPU_vec_on_vec, 0, sizeof(double)));
		GPU_null_one<<<1, 1>>>(dev_norm);
		GPU_vec_on_vec<<<block_size, grid_size>>>(dev_r_k1, dev_r_k1, dev_norm);
		//CUDA_CALL(cudaDeviceSynchronize());
		CUDA_CALL(cudaMemcpy(norm, dev_norm, 1 * sizeof(double), cudaMemcpyDeviceToHost));
		//CUDA_CALL(cudaMemcpyFromSymbol(&norm, "res_GPU_vec_on_vec", sizeof(double), 0, cudaMemcpyDeviceToHost));

		//norm(r_k)^2
		//norm2 = pow(norm_vec(r), 2);
		//norm2 = vec_on_vec(r, r);

		//CUDA_CALL(cudaMemset(res_GPU_vec_on_vec, 0, sizeof(double)));
		GPU_null_one<<<1, 1>>>(dev_norm2);
		GPU_vec_on_vec<<<block_size, grid_size>>>(dev_r, dev_r, dev_norm2);
		//CUDA_CALL(cudaDeviceSynchronize());
		CUDA_CALL(cudaMemcpy(norm2, dev_norm2, 1 * sizeof(double), cudaMemcpyDeviceToHost));
		//CUDA_CALL(cudaMemcpyFromSymbol(&norm2, "res_GPU_vec_on_vec", sizeof(double), 0, cudaMemcpyDeviceToHost));

		//bk = ...
		bk = (isnan(norm[0] / norm2[0]) ? 0 : (norm[0] / norm2[0]));

		//-------------------------------------------------------------------------------------------

		//nullify(tmp);

		GPU_nullify<<<block_size, grid_size>>>(dev_tmp);
		//CUDA_CALL(cudaDeviceSynchronize());

		//p_k+1 = r_k+1 + bk*p_k
		//bk*p_k
		//mult_vec_on_num(p, bk, tmp);

		GPU_mult_vec_on_num<<<block_size, grid_size>>>(dev_p, bk, dev_tmp);
		//CUDA_CALL(cudaDeviceSynchronize());

		//p_k+1 = r_k+1 + ...
		//sum_vec(r_k1, tmp, p_k1);

		GPU_sum_vec<<<block_size, grid_size>>>(dev_r_k1, dev_tmp, dev_p_k1);
		//CUDA_CALL(cudaDeviceSynchronize());

		//-------------------------------------------------------------------------------------------

		CUDA_CALL(cudaMemcpy(x_k1, dev_x_k1, S * sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaDeviceSynchronize());
		cout << "X[0]: " << endl;
		show_vec(x_k1, 1);
		
		CUDA_CALL(cudaMemcpy(A, dev_A, S * 7 * sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaDeviceSynchronize());
		stop_eps = check_stop(A, F, x_k1);

		add_in_file(stop_eps);

		cout << "Nev: " << stop_eps << " " << ((stop_eps < Eps) ? "<" : ">") << " " << Eps << endl << endl;
		if (stop_eps < Eps || count_tmp > S - 1) {
			write_in_file(x_k1, S, "X_1.dat");
		}

		//-------------------------------------------------------------------------------------------

		//copy for next iter
		//p = p_k1;
		//copy_vec(p, p_k1);
		GPU_copy_vec<<<block_size, grid_size>>>(dev_p, dev_p_k1);
		//CUDA_CALL(cudaDeviceSynchronize());

		//x = x_k1;
		//copy_vec(x, x_k1);
		GPU_copy_vec<<<block_size, grid_size>>>(dev_x, dev_x_k1);
		//CUDA_CALL(cudaDeviceSynchronize());

		//r = r_k1;
		//copy_vec(r, r_k1);
		GPU_copy_vec<<<block_size, grid_size>>>(dev_r, dev_r_k1);
		//CUDA_CALL(cudaDeviceSynchronize());

		//clear vectors
		//nullify(stop);
		//nullify(p_k1);
		//nullify(x_k1);
		//nullify(r_k1);
		//nullify(tmp);
		//nullify(rApk);

		GPU_nullify<<<block_size, grid_size>>>(dev_stop);
		//CUDA_CALL(cudaDeviceSynchronize());
		
		GPU_nullify<<<block_size, grid_size>>>(dev_p_k1);
		//CUDA_CALL(cudaDeviceSynchronize());

		GPU_nullify<<<block_size, grid_size>>>(dev_x_k1);
		//CUDA_CALL(cudaDeviceSynchronize());

		GPU_nullify<<<block_size, grid_size>>>(dev_r_k1);
		//CUDA_CALL(cudaDeviceSynchronize());

		GPU_nullify<<<block_size, grid_size>>>(dev_tmp);
		//CUDA_CALL(cudaDeviceSynchronize());

		GPU_nullify<<<block_size, grid_size>>>(dev_rApk);
		//CUDA_CALL(cudaDeviceSynchronize());

		CUDA_CALL(cudaGetLastError());

		//-------------------------------------------------------------------------------------------

		clock_t end_CGMR = clock();

		double CGMR_time = double(end_CGMR - begin_CGMR) / CLOCKS_PER_SEC;
		cout << "Runtime of iter: " << CGMR_time << endl << endl;

		clock_t end_algo = clock();
		double algo_time = double(end_algo - begin_algo) / CLOCKS_PER_SEC;
		//cout << "Runtime of algo: " << algo_time << endl;
	}
	delete(x);
	delete(r);
	delete(p);
	delete(x_k1);
	delete(r_k1);
	delete(p_k1);
	delete(tmp);
	delete(rApk);
	delete(stop);

	cudaFree(dev_x);
	cudaFree(dev_r);
	cudaFree(dev_p);
	cudaFree(dev_x_k1);
	cudaFree(dev_r_k1);
	cudaFree(dev_p_k1);
	cudaFree(dev_tmp);
	cudaFree(dev_rApk);
	cudaFree(dev_stop);
	cudaFree(dev_A);
	cudaFree(dev_F);
	cudaFree(dev_or_A);
}

void main() {
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

	CGMR(matr_A, matr_F, begin_algo);

	clock_t end = clock();
	double time = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Runtime: " << time << endl;

	delete(matr_A);
	delete(matr_F);

	system("pause");

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
}