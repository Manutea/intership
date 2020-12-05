#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "TFM.h"
#include "../Quadtree/quadtree.h"

namespace tfm {

	namespace device {

		__global__ void compute_tfm(const tfm::device::parameters tfm_param, const int size, int *samples, int *amplitudes,
			const double *delays, const short *fmc, const double *tof) {

			const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
			const int local_y = threadIdx.y;

			if (global_x >= size || samples[global_x] != -1)
				return;

			const int idX = global_x % tfm_param.columns;
			const int idY = global_x / tfm_param.columns;

			const int sizeEmitRecei = tfm_param.grids / 2;
			int tfm = 0;

			for (int idReceiver = 0; idReceiver < sizeEmitRecei; idReceiver++) {
				const double tValueEmit = tof[idX + tfm_param.columns * idY + tfm_param.rows * tfm_param.columns * local_y];
				const double tValueRece = tof[idX + tfm_param.columns * idY + tfm_param.rows * tfm_param.columns * (idReceiver + sizeEmitRecei)];

				const double tSample = tValueEmit + tValueRece + delays[idReceiver * sizeEmitRecei + local_y];
				const int idSample = static_cast<int>(tSample * tfm_param.frequency) - 1;

				if (idSample > 0 && idSample < tfm_param.samples) {
					tfm += fmc[idSample + tfm_param.samples * (idReceiver + sizeEmitRecei * local_y)];
				}
			}

			atomicAdd(&amplitudes[global_x], tfm);

			if (local_y == 0)
				samples[global_x] = true;		// le point devient un point d'échantillon tfm 
		}


		__global__ void tfm_for_granity(const tfm::device::parameters tfm_param, const int size, const double *delays, const short *fmc, const double *tof) {

			const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
			
			if (global_x >= size)
				return;
			
			const int idX = global_x % tfm_param.columns;
			const int idY = global_x / tfm_param.columns;
			
			const int sizeEmitRecei = tfm_param.grids / 2;
			int tfm = 0;
			
			for (int idEmit = 0; idEmit < sizeEmitRecei; idEmit++) {
				for (int idReceiver = 0; idReceiver < sizeEmitRecei; idReceiver++) {
					const double tValueEmit = tof[idX + tfm_param.columns * idY + tfm_param.rows * tfm_param.columns * idEmit];
					const double tValueRece = tof[idX + tfm_param.columns * idY + tfm_param.rows * tfm_param.columns * (idReceiver + sizeEmitRecei)];
			
					const double tSample = tValueEmit + tValueRece + delays[idReceiver * sizeEmitRecei + idEmit];
					const int idSample = static_cast<int>(tSample * tfm_param.frequency) - 1;
			
					if (idSample > 0 && idSample < tfm_param.samples) {
						tfm += fmc[idSample + tfm_param.samples * (idReceiver + sizeEmitRecei * idEmit)];
					}
				}
			}
		}
	}

	namespace cuda {

		void compute_tfm(quadtree::device::ressources &ressources) {
			const int size = ressources.tfm_param.rows * ressources.tfm_param.columns;
			dim3 block(1, 64, 1);

			compute_tfm << <size, block >> > (ressources.tfm_param, size,
				thrust::raw_pointer_cast(ressources.samples.data()),
				thrust::raw_pointer_cast(ressources.amplitudes.data()),
				thrust::raw_pointer_cast(ressources.delays.data()),
				thrust::raw_pointer_cast(ressources.fmc.data()),
				thrust::raw_pointer_cast(ressources.tof.data()));
		}

		float tfm_for_granity(const int pixels_per_iteration, quadtree::device::ressources &ressources) {

			const int size = ressources.tfm_param.rows * ressources.tfm_param.columns;

			int nb_kernels = size / pixels_per_iteration;
			if (size % pixels_per_iteration != 0)
				nb_kernels++;

			const int block_size = 128;
			int grid_size = pixels_per_iteration / block_size;
			if (pixels_per_iteration % block_size != 0)
				grid_size++;

			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);

			for (int ikernel = 0; ikernel < nb_kernels; ikernel++) {
				tfm_for_granity << <grid_size, block_size >> > (ressources.tfm_param, size,
					thrust::raw_pointer_cast(ressources.delays.data()),
					thrust::raw_pointer_cast(ressources.fmc.data()),
					thrust::raw_pointer_cast(ressources.tof.data()));
				cudaDeviceSynchronize();
			}
			cudaEventRecord(stop);


			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);

			return milliseconds;
		}
	}

}