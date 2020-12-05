#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include "quadtree.h"

#define NODE_FLAG_INIT -100.0

namespace quadtree {

	namespace device {
		
		__device__ void make_point(int *samples, int *amplitudes, const int index) {
			samples[index] = -1;
			amplitudes[index] = 0;
		}

		__device__ quadtree::node make_node(const int width, int *samples, int *amplitudes, const int2 pts_nw, const int2 pts_se, bool *already_traced, const int depth, const int index) {

			const int center_x = pts_nw.x + (pts_se.x - pts_nw.x) / 2;
			const int center_y = pts_nw.y + (pts_se.y - pts_nw.y) / 2;
			quadtree::node n;
			n.depth = depth;

			n.index = index;
			n.ready_to_subdivise = true;

			int node_index = width * pts_nw.y + pts_nw.x;
			n.points_x[0] = pts_nw.x;
			n.points_y[0] = pts_nw.y;
			if (already_traced[node_index] == false) {
				already_traced[node_index] = true;
				make_point(samples, amplitudes, n.points_y[0] * width + n.points_x[0]);
			}

			node_index = width * pts_nw.y + center_x;
			n.points_x[1] = center_x;
			n.points_y[1] = pts_nw.y;
			if (already_traced[node_index] == false) {
				already_traced[node_index] = true;
				make_point(samples, amplitudes, n.points_y[1] * width + n.points_x[1]);
			}

			node_index = width * pts_nw.y + pts_se.x;
			n.points_x[2] = pts_se.x;
			n.points_y[2] = pts_nw.y;
			if (already_traced[node_index] == false) {
				already_traced[node_index] = true;
				make_point(samples, amplitudes, n.points_y[2] * width + n.points_x[2]);
			}

			node_index = width * center_y + pts_nw.x;
			n.points_x[3] = pts_nw.x;
			n.points_y[3] = center_y;
			if (already_traced[node_index] == false) {
				already_traced[node_index] = true;
				make_point(samples, amplitudes, n.points_y[3] * width + n.points_x[3]);
			}

			node_index = width * center_y + center_x;
			n.points_x[4] = center_x;
			n.points_y[4] = center_y;
			if (already_traced[node_index] == false) {
				already_traced[node_index] = true;
				make_point(samples, amplitudes, n.points_y[4] * width + n.points_x[4]);
			}

			node_index = width * center_y + pts_se.x;
			n.points_x[5] = pts_se.x;
			n.points_y[5] = center_y;
			if (already_traced[node_index] == false) {
				already_traced[node_index] = true;
				make_point(samples, amplitudes, n.points_y[5] * width + n.points_x[5]);
			}

			node_index = width * pts_nw.y + pts_nw.x;
			n.points_x[6] = pts_nw.x;
			n.points_y[6] = pts_se.y;
			if (already_traced[node_index] == false) {
				already_traced[node_index] = true;
				make_point(samples, amplitudes, n.points_y[6] * width + n.points_x[6]);
			}

			node_index = width * pts_se.y + center_x;
			n.points_x[7] = center_x;
			n.points_y[7] = pts_se.y;
			if (already_traced[node_index] == false) {
				already_traced[node_index] = true;
				make_point(samples, amplitudes, n.points_y[7] * width + n.points_x[7]);
			}

			node_index = width * pts_se.y + pts_se.x;
			n.points_x[8] = pts_se.x;
			n.points_y[8] = pts_se.y;
			if (already_traced[node_index] == false) {
				already_traced[node_index] = true;
				make_point(samples, amplitudes, n.points_y[8] * width + n.points_x[8]);
			}

			n.score = 0;
			return n;
		}

		__global__ void init_quadtree(const tfm::device::parameters tfm_param, int *samples, int *amplitudes, bool *already_traced, quadtree::node *nodes) {
			int local_x = threadIdx.x;

			if (local_x > 8)
				return;

			const int center_x = tfm_param.columns / 2;
			const int center_y = tfm_param.rows / 2;

			if (local_x == 0) {
				already_traced[0] = true;
				nodes[0].depth = 0;
				nodes[0].index = 0;
				nodes[0].points_x[0] = 0;
				nodes[0].points_y[0] = 0;
				make_point(samples, amplitudes, 0);
				nodes[0].score = 0;
			}

			else if (local_x == 1) {
				already_traced[center_x] = true;
				nodes[0].points_x[1] = center_x;
				nodes[0].points_y[1] = 0;
				make_point(samples, amplitudes, center_x);
			}

			else if (local_x == 2) {
				const int index = tfm_param.columns - 1;
				already_traced[index] = true;
				nodes[0].points_x[2] = tfm_param.columns - 1;
				nodes[0].points_y[2] = 0;
				make_point(samples, amplitudes, index);
			}

			else if (local_x == 3) {
				const int index = tfm_param.columns * center_y;
				already_traced[index] = true;
				nodes[0].points_x[3] = 0;
				nodes[0].points_y[3] = center_y;
				make_point(samples, amplitudes, index);
			}

			else if (local_x == 4) {
				const int index = tfm_param.columns * center_y + center_x;
				already_traced[index] = true;
				nodes[0].points_x[4] = center_x;
				nodes[0].points_y[4] = center_y;
				make_point(samples, amplitudes, index);
			}

			else if (local_x == 5) {
				const int index = tfm_param.columns * center_y + tfm_param.columns - 1;
				already_traced[index] = true;
				nodes[0].points_x[5] = tfm_param.columns - 1;
				nodes[0].points_y[5] = center_y;
				make_point(samples, amplitudes, index);
			}

			else if (local_x == 6) {
				const int index = tfm_param.columns * (tfm_param.rows - 1);
				already_traced[index] = true;
				nodes[0].points_x[6] = 0;
				nodes[0].points_y[6] = tfm_param.rows - 1;
				make_point(samples, amplitudes, index);
			}

			else if (local_x == 7) {
				const int index = tfm_param.columns * (tfm_param.rows - 1) + center_x;
				already_traced[index] = true;
				nodes[0].points_x[7] = center_x;
				nodes[0].points_y[7] = tfm_param.rows - 1;
				make_point(samples, amplitudes, index);
			}

			else {
				const int index = tfm_param.columns * (tfm_param.rows - 1) + tfm_param.columns - 1;
				already_traced[index] = true;
				nodes[0].points_x[8] = tfm_param.columns - 1;
				nodes[0].points_y[8] = tfm_param.rows - 1;
				make_point(samples, amplitudes, index);
			}
		}
	
		__global__ void firsts_subdivisions(const int width, const int depth, const int node_to_subdivise, const int total_nodes_subdivised, 
			int *samples, int *amplitudes, bool *already_traced, quadtree::node *nodes) {

			const int local_y = threadIdx.y;
			const int global_x = blockIdx.x * blockDim.x + threadIdx.x;

			if (global_x >= node_to_subdivise)
				return;

			int row = total_nodes_subdivised + 4 * global_x;
			const quadtree::node parent = nodes[row / 4];

			if (local_y == 0) {
				const int2 p1 = { parent.points_x[0], parent.points_y[0] };
				const int2 p2 = { parent.points_x[4], parent.points_y[4] };
				nodes[row] = make_node(width, samples, amplitudes, p1, p2, already_traced, depth + 1, row);
				nodes[row / 4].score = -1.0; //le supprime du tableau des noeuds a subdiviser
				nodes[row / 4].ready_to_subdivise = false;
			}

			else if (local_y == 1) {
				const int2 p1 = { parent.points_x[1], parent.points_y[1] };
				const int2 p2 = { parent.points_x[5], parent.points_y[5] };
				nodes[row + 1] = make_node(width, samples, amplitudes, p1, p2, already_traced, depth + 1, row + 1);
			}

			else if (local_y == 2) {
				const int2 p1 = { parent.points_x[3], parent.points_y[3] };
				const int2 p2 = { parent.points_x[7], parent.points_y[7] };
				nodes[row + 2] = make_node(width, samples, amplitudes, p1, p2, already_traced, depth + 1, row + 2);
			}

			else if (local_y == 3) {

				const int2 p1 = { parent.points_x[4], parent.points_y[4] };
				const int2 p2 = { parent.points_x[8], parent.points_y[8] };
				nodes[row + 3] = make_node(width, samples, amplitudes, p1, p2, already_traced, depth + 1, row + 3);
			}
		}
	
		__global__ void subdivise(const int width, const int depth, const int node_to_subdivise, const int total_nodes_subdivised,
			int *samples, int *amplitudes, bool *already_traced, quadtree::node *nodes) {

			const int local_y = threadIdx.y;
			const int global_x = blockIdx.x * blockDim.x + threadIdx.x;

			if (global_x >= node_to_subdivise)
				return;

			int row = total_nodes_subdivised + 4 * global_x;
			const quadtree::node parent = nodes[global_x];

			if (local_y == 0) {
				const int2 p1 = { parent.points_x[0], parent.points_y[0] };
				const int2 p2 = { parent.points_x[4], parent.points_y[4] };
				nodes[row] = make_node(width, samples, amplitudes, p1, p2, already_traced, depth + 1, row);
				nodes[global_x].score = -1.0; //le supprime du tableau des noeuds a subdiviser
			}

			else if (local_y == 1) {
				const int2 p1 = { parent.points_x[1], parent.points_y[1] };
				const int2 p2 = { parent.points_x[5], parent.points_y[5] };
				nodes[row + 1] = make_node(width, samples, amplitudes, p1, p2, already_traced, depth + 1, row + 1);
			}

			else if (local_y == 2) {
				const int2 p1 = { parent.points_x[3], parent.points_y[3] };
				const int2 p2 = { parent.points_x[7], parent.points_y[7] };
				nodes[row + 2] = make_node(width, samples, amplitudes, p1, p2, already_traced, depth + 1, row + 2);
			}

			else if (local_y == 3) {
				const int2 p1 = { parent.points_x[4], parent.points_y[4] };
				const int2 p2 = { parent.points_x[8], parent.points_y[8] };
				nodes[row + 3] = make_node(width, samples, amplitudes, p1, p2, already_traced, depth + 1, row + 3);
			}
		}

	}

	namespace cuda {

		void init_quadtree(quadtree::device::ressources &ressources) {
			int block = 128;
			device::init_quadtree << <1, block >> > (ressources.tfm_param,
				thrust::raw_pointer_cast(ressources.samples.data()),
				thrust::raw_pointer_cast(ressources.amplitudes.data()),
				thrust::raw_pointer_cast(ressources.already_traced.data()),
				thrust::raw_pointer_cast(ressources.nodes.data()));
		}

		void firsts_subdivisions(const int width, const int depth, const int nodes_to_subdivise, const int total_nodes_subdivised, 
			quadtree::device::ressources &ressources) {

			dim3 block(32, 4, 1);

			int grid = nodes_to_subdivise / block.x;
			if (nodes_to_subdivise % block.x != 0)
				grid++;

			device::firsts_subdivisions << <grid, block >> > (width, depth, nodes_to_subdivise, total_nodes_subdivised,
				thrust::raw_pointer_cast(ressources.samples.data()), 
				thrust::raw_pointer_cast(ressources.amplitudes.data()), 
				thrust::raw_pointer_cast(ressources.already_traced.data()),
				thrust::raw_pointer_cast(ressources.nodes.data()));
		}

		void subdivise(const int width, const int depth, const int nodes_to_subdivise, const int total_nodes_subdivised,
			quadtree::device::ressources &ressources) {
			dim3 block(32, 4, 1);
			int grid = nodes_to_subdivise / block.x;
			if (nodes_to_subdivise % block.x != 0)
				grid++;

			device::subdivise << < grid, block >> > (width, depth, nodes_to_subdivise, total_nodes_subdivised,
				thrust::raw_pointer_cast(ressources.samples.data()),
				thrust::raw_pointer_cast(ressources.amplitudes.data()),
				thrust::raw_pointer_cast(ressources.already_traced.data()),
				thrust::raw_pointer_cast(ressources.nodes.data()));
		}

		quadtree::device::ressources quadtree_ressources(const tfm::host::parameters &tfm_param, const std::vector<short> &fmc, const std::vector<double> &tof) {
			
			quadtree::device::ressources d_ressources;
			thrust::host_vector<short> host_fmc(fmc.data(), fmc.data() + fmc.size());
			thrust::host_vector<double> host_tof(tof.data(), tof.data() + tof.size());
			thrust::host_vector<double> host_delays(tfm_param.delays.data(), tfm_param.delays.data() + tfm_param.signals);

			// Copy data to device
			d_ressources.fmc = host_fmc;
			d_ressources.tof = host_tof;
			d_ressources.already_traced.resize(tfm_param.size);
			thrust::fill(d_ressources.already_traced.begin(), d_ressources.already_traced.end(), false);
			d_ressources.nodes.resize(tfm_param.size, { 0, 0, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, NODE_FLAG_INIT, false});

			// TFM points
			d_ressources.samples.resize(tfm_param.size);
			d_ressources.amplitudes.resize(tfm_param.size);

			// TFM parameters
			d_ressources.delays = host_delays;
			d_ressources.tfm_param.rows = tfm_param.rows;
			d_ressources.tfm_param.grids = tfm_param.grids;
			d_ressources.tfm_param.columns = tfm_param.columns;
			d_ressources.tfm_param.samples = tfm_param.samples;
			d_ressources.tfm_param.frequency = tfm_param.frequency;
			d_ressources.tfm_param.contributors = tfm_param.contributors;

			return std::move(d_ressources);
		}
	
		std::vector<quadtree::node> copy_nodes(const quadtree::parameters &quad_param, const quadtree::device::ressources &ressources) {
			cudaDeviceSynchronize();
			std::vector<quadtree::node> nodes(quad_param.tree_size);
			cudaMemcpy(nodes.data(), thrust::raw_pointer_cast(ressources.nodes.data()), sizeof(node) * quad_param.tree_size, cudaMemcpyDeviceToHost);
			return nodes;
		}
	}
}