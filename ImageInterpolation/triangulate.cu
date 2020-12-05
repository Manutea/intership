#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <iostream>

#include "triangulate.h"

namespace triangulate {
	namespace device {
		__device__ __managed__ int idxTri = 0;

		__device__ triangle make_triangle(const int x0, const int y0, const int x1, const int y1, const int x2, const int y2) {
			triangle t;
			t.v0x = x0;
			t.v0y = y0;

			t.v1x = x1;
			t.v1y = y1;

			t.v2x = x2;
			t.v2y = y2;
			return t;
		}

		__global__ void triangulation(const int width, const int height, const int maxNodes, quadtree::node *nodes, int *samples, triangle *triangles) {

			const int local_y = threadIdx.y;

			const int global_x = blockIdx.x * blockDim.x + threadIdx.x;

			if (global_x >= maxNodes || nodes[global_x].score < 0)
				return;

			quadtree::node currentNode = nodes[global_x];

			int2 vc = { currentNode.points_x[4], currentNode.points_y[4] };

			if (local_y == 0) {
				//Search on the North edge
				int start = currentNode.points_x[0];
				int y = currentNode.points_y[0];
				int2 v0 = { start, y };
				for (int x = start + 1; x <= currentNode.points_x[2]; x++) {
					if (samples[width * y + x]) {
						int idx = atomicAdd(&idxTri, 1);
						triangles[idx] = make_triangle(vc.x, vc.y, x, y, v0.x, v0.y);
						v0.x = x;
					}
				}
			}

			else if (local_y == 1) {
				//Search on the South edge
				int start = currentNode.points_x[6];
				int y = currentNode.points_y[6];
				int2 v0 = { start, y };

				for (int x = start + 1; x <= currentNode.points_x[8]; x++) {
					if (samples[width * v0.y + x] == 1) {
						int idx = atomicAdd(&idxTri, 1);
						triangles[idx] = make_triangle(x, y, vc.x, vc.y, v0.x, v0.y);
						v0.x = x;
					}
				}
			}

			else if (local_y == 2) {
				//Search on the West edge
				int start = currentNode.points_y[0];
				int x = currentNode.points_x[0];
				int2 v0 = { x, start };

				for (int y = start + 1; y <= currentNode.points_y[6]; y++) {
					if (samples[width * y + v0.x] == 1) {
						int idx = atomicAdd(&idxTri, 1);
						triangles[idx] = make_triangle(x, y, vc.x, vc.y, v0.x, v0.y);
						v0.y = y;
					}
				}
			}

			else {
				//Search on the East edge
				int start = currentNode.points_y[2];
				int x = currentNode.points_x[2];
				int2 v0 = { x, start };

				for (int y = start + 1; y <= currentNode.points_y[8]; y++) {
					if (samples[width * y + v0.x] == 1) {
						int idx = atomicAdd(&idxTri, 1);
						triangles[idx] = make_triangle(vc.x, vc.y, x, y, v0.x, v0.y);
						v0.y = y;
					}
				}
			}
		}
	}

	namespace cuda {

		triangulate::device::ressources triangulate_ressources(const tfm::host::parameters &tfm_param) {

			device::ressources ressources;
			ressources.triangles.resize(tfm_param.size);

			return std::move(ressources);
		}

		int triangulate(const tfm::host::parameters &tfm_param, const quadtree::parameters &quad_param, 
			quadtree::device::ressources &quadtree_ressources, device::ressources &ressources) {

			dim3 block(32, 4, 1);

			int grid = (4 * quad_param.tree_size)/(block.x * block.y);
			if ((4 * quad_param.tree_size) % 128 != 0)
				grid++;
			
			device::triangulation << <grid, block >> > (tfm_param.columns, tfm_param.rows, quad_param.tree_size,
				thrust::raw_pointer_cast(quadtree_ressources.nodes.data()),
				thrust::raw_pointer_cast(quadtree_ressources.samples.data()),
				thrust::raw_pointer_cast(ressources.triangles.data()));

			// Must be sync cause of managed memory
			cudaDeviceSynchronize();

			return device::idxTri;
		}
	}
}