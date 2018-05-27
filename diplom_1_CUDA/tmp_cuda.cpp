#include <iostream>
#include <fstream>
#include <string>
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


using namespace std;

__global__ void kernel_blur(unsigned char *arr, unsigned char *res_arr, int width, int height)
{
	int rad = 15;
	int n_rs = 1;
	int sum_color = 0;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	for (int dx = -rad; dx <= rad; ++dx) {
		for (int dy = -rad; dy <= rad; ++dy) {
			int nx = x + dx;
			int ny = y + dy;
			if (ny > 0 & ny < width & nx > 0 & ny < height) {
				++n_rs;

				sum_color = sum_color + arr[ny * width + nx];
			}
			else
				//cout << "Nope " << ny * fileInfoHeader.biWidth + nx <<  endl;
				continue;
		}
	}

	res_arr[y*width + x] = sum_color / n_rs;

}

int main(int argc, char *argv[])
{
	char* fileName = "C:\\Users\\George\\Documents\\Visual Studio 2015\\Projects\\bmp_blur\\bmp_blur\\work.bmp";

	// открываем файл
	FILE *f = fopen(fileName, "r+b");
	if (!f) {
		cout << "Error opening file '" << fileName << "'." << endl;
		return 0;
	}

	// заголовок изображения
	BITMAPFILEHEADER fileHeader;
	fread(&fileHeader, sizeof(BITMAPFILEHEADER), 1, f);
	cout << "bfSize = " << fileHeader.bfSize << endl;

	if (fileHeader.bfType != 0x4D42) {
		cout << "Error: '" << fileName << "' is not BMP file." << endl;
		return 0;
	}

	// информация изображения
	BITMAPINFOHEADER fileInfoHeader;
	fread(&fileInfoHeader, sizeof(BITMAPINFOHEADER), 1, f);
	cout << "biSize = " << fileInfoHeader.biSize << endl;

	//read
	int totalPixel = fileInfoHeader.biWidth * fileInfoHeader.biHeight;
	unsigned char *pixelData = new unsigned char[totalPixel * 3];
	fread(pixelData, sizeof(unsigned char), totalPixel * 3, f);

	unsigned char *pBlue = new unsigned char[totalPixel];
	unsigned char *pGreen = new unsigned char[totalPixel];
	unsigned char *pRed = new unsigned char[totalPixel];

	int bIndex = 0;
	int gIndex = 0;
	int rIndex = 0;


	for (int i = 0; i < totalPixel; ++i) {
		bIndex = i * 3;
		gIndex = i * 3 + 1;
		rIndex = i * 3 + 2;

		pBlue[i] = pixelData[bIndex];
		pGreen[i] = pixelData[gIndex];
		pRed[i] = pixelData[rIndex];
	}

	unsigned char *devB = NULL;
	unsigned char *devG = NULL;
	unsigned char *devR = NULL;

	unsigned char *devB_res = NULL;
	unsigned char *devG_res = NULL;
	unsigned char *devR_res = NULL;

	const int count_size = 16 * 16;
	int gs_width = fileInfoHeader.biWidth / 16 + (fileInfoHeader.biWidth % 16 == 0 ? 0 : 1);
	int gs_height = fileInfoHeader.biHeight / 16 + (fileInfoHeader.biHeight % 16 == 0 ? 0 : 1);

	dim3 block_size(16, 16);
	dim3 grid_size(gs_width, gs_height);

	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc(&devB, totalPixel * sizeof(unsigned char));
	cudaStatus = cudaMalloc(&devG, totalPixel * sizeof(unsigned char));
	cudaStatus = cudaMalloc(&devR, totalPixel * sizeof(unsigned char));

	cudaStatus = cudaMalloc(&devB_res, totalPixel * sizeof(unsigned char));
	cudaStatus = cudaMalloc(&devG_res, totalPixel * sizeof(unsigned char));
	cudaStatus = cudaMalloc(&devR_res, totalPixel * sizeof(unsigned char));

	CUDA_CALL(cudaMemcpy(devB, pBlue, totalPixel * sizeof(unsigned char), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(devG, pGreen, totalPixel * sizeof(unsigned char), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(devR, pRed, totalPixel * sizeof(unsigned char), cudaMemcpyHostToDevice));

	kernel_blur<<<grid_size, block_size>>>(devB, devB_res, fileInfoHeader.biWidth, fileInfoHeader.biHeight);
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());

	kernel_blur <<<grid_size, block_size >>>(devG, devG_res, fileInfoHeader.biWidth, fileInfoHeader.biHeight);
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());

	kernel_blur <<<grid_size, block_size >>>(devR, devR_res, fileInfoHeader.biWidth, fileInfoHeader.biHeight);
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());

	CUDA_CALL(cudaMemcpy(pBlue, devB_res, totalPixel * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(pGreen, devG_res, totalPixel * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(pRed, devR_res, totalPixel * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	//CUDA_CALL(cudaDeviceSynchronize());

	//new pixelData
	unsigned char *newPixelData = new unsigned char[totalPixel * 3];

	for (int i = 0; i < totalPixel; ++i) {
		bIndex = i * 3;
		gIndex = i * 3 + 1;
		rIndex = i * 3 + 2;

		newPixelData[bIndex] = pBlue[i];
		newPixelData[gIndex] = pGreen[i];
		newPixelData[rIndex] = pRed[i];
	}

	//write
	FILE *resPng;
	resPng = fopen("tmp.bmp", "w+b");

	fwrite(&fileHeader, sizeof(BITMAPFILEHEADER), 1, resPng);
	fwrite(&fileInfoHeader, sizeof(BITMAPINFOHEADER), 1, resPng);
	fwrite(newPixelData, sizeof(unsigned char), totalPixel * 3, resPng);
	fclose(resPng);

	cudaFree(devB);
	cudaFree(devG);
	cudaFree(devR);
	return 1;
}