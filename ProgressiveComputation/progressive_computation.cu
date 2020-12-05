#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/sort.h>
#include "progressive_computation.h"

#include <limits>
#include <fstream>


struct nodeCmp {
	__host__ __device__
		bool operator()(const quadtree::node &n1, const quadtree::node &n2) {
		return n1.score > n2.score;
	}
};
 
namespace pComputation {

	namespace device {

		__device__ int best_amp(const quadtree::node n, const int *amplitudes, const int width) {
			int best_amp = amplitudes[width * n.points_y[0] + n.points_x[0]];
			best_amp = (best_amp < 0) ? -best_amp : best_amp;
			for (short index = 1; index < 9; index++) {
				int current_amp = amplitudes[width * n.points_y[index] + n.points_x[index]];
				current_amp = (current_amp < 0) ? -current_amp : current_amp;
				best_amp = (current_amp > best_amp) ? current_amp : best_amp;
			}
			return best_amp;
		}

		__global__ void compute_score(const int width, const int depth, const int end, quadtree::node *nodes, const int *amplitudes) {
			const int global_x = blockIdx.x * blockDim.x + threadIdx.x;

			if (global_x >= end)
				return;

			if (nodes[global_x].ready_to_subdivise == false)
				return;

			const int d = depth + 1;
			nodes[global_x].score = static_cast<double>(best_amp(nodes[global_x], amplitudes, width)) / static_cast<double>(d*1*d*0.90*d*0.80*d*0.70*d*0.60*d*0.50*d*0.40*d*0.30*d*0.20*d*0.10);

			nodes[global_x].ready_to_subdivise = false;
		}
	}

	namespace cuda {

		void compute_score(const int width, const int depth, const quadtree::parameters &quad_param, quadtree::device::ressources &ressources) {
			
			int block = 128;
			int grid = quad_param.tree_size / block;
			if (quad_param.tree_size % 128 != 0)
				grid++; 
			device::compute_score << <grid, block >> > (width, depth, quad_param.tree_size,
				thrust::raw_pointer_cast(ressources.nodes.data()), 
				thrust::raw_pointer_cast(ressources.amplitudes.data()));
		}
	
		void sort(quadtree::device::ressources &ressources) {
			thrust::sort(ressources.nodes.begin(), ressources.nodes.end(), nodeCmp());
		}

		std::pair<int, float> granularity(const int nb_multiprocs, const int nb_blocks_per_mp, quadtree::device::ressources &ressources) {

			const int block_size = 128;
			const int granularity = nb_multiprocs * nb_blocks_per_mp * block_size;
			const int nb_pixels = (ressources.tfm_param.rows * ressources.tfm_param.columns)/5;

			float best_cadence = -std::numeric_limits<float>::infinity();
			int best_granularity = 0;
			float best_time = 0;

			for (int kernel_size = granularity; kernel_size < nb_pixels; kernel_size += granularity) {
				float time = tfm::cuda::tfm_for_granity(kernel_size, ressources);
				float cadence = nb_pixels / time;
				if (cadence > best_cadence) {
					best_cadence = cadence;
					best_granularity = kernel_size;
					best_time = time;
				}

			}

			std::pair<int, float> pair(best_granularity, best_cadence);
			return pair;
		}
	}

}