#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "interpolate.h"

namespace interpolate {
	
	namespace device {
		
		__device__ float edge_function(const float2 v0, const float2 v1, const float px, const float py) {
			return (px - v0.x) * (v1.y - v0.y) - (py - v0.y) * (v1.x - v0.x);
		}

		__global__ void interpolation(const int width, const int nbr_triangles, double *image_out, const int *amplitudes, const int *sample,
			const triangulate::triangle *triangles) {
			int tid = blockIdx.x * blockDim.x + threadIdx.x;

			if (tid >= nbr_triangles)
				return;

			triangulate::triangle t = triangles[tid];
			float2 point0;
			point0.x = t.v0x + 0.5;
			point0.y = t.v0y + 0.5;

			float2 point1;
			point1.x = t.v1x + 0.5;
			point1.y = t.v1y + 0.5;

			float2 point2;
			point2.x = t.v2x + 0.5;
			point2.y = t.v2y + 0.5;

			//Calcul de la boite englobant le triangle
			short2 pointNW;
			short2 pointSE;

			pointNW.x = min(min(t.v0x, t.v1x), t.v2x);
			pointNW.y = min(min(t.v0y, t.v1y), t.v2y);

			pointSE.x = max(max(t.v0x, t.v1x), t.v2x);
			pointSE.y = max(max(t.v0y, t.v1y), t.v2y);

			float area = edge_function(point0, point1, point2.x, point2.y);

			for (short row = pointNW.y; row < pointSE.y; row++)
				for (short col = pointNW.x; col < pointSE.x; col++) {
					int index = width * row + col;

					if (sample[index] == 1) {
						image_out[index] = amplitudes[index];
					}

					else {
						float w0 = edge_function(point1, point2, col + 0.5, row + 0.5);
						float w1 = edge_function(point2, point0, col + 0.5, row + 0.5);
						float w2 = edge_function(point0, point1, col + 0.5, row + 0.5);
						if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
							w0 /= area;
							w1 /= area;
							w2 /= area;
							float amplitude = w0 * amplitudes[width * t.v0y + t.v0x] + w1 * amplitudes[width * t.v1y + t.v1x] + w2 * amplitudes[width * t.v2y + t.v2x];
							image_out[index] = static_cast<double>(amplitude);
						}
					}
				}
		}
	}
	
	namespace cuda {

		interpolate::device::ressources interpolate_ressources(const tfm::host::parameters &tfm_param) {
			interpolate::device::ressources ressources;
			ressources.image.resize(tfm_param.size);

			return std::move(ressources);
		}

		void interpolate(const int nbr_triangles, const tfm::host::parameters &tfm_param, const quadtree::device::ressources &quadtree_ressources, 
			const triangulate::device::ressources &triangulate_ressources, interpolate::device::ressources &interpolate_ressources) {

			int block = 64;
			int grid = nbr_triangles / block;
			if (nbr_triangles % 64 != 0)
				grid++;

			interpolate::device::interpolation << <grid, block>> > (tfm_param.columns, nbr_triangles,
				thrust::raw_pointer_cast(interpolate_ressources.image.data()),
				thrust::raw_pointer_cast(quadtree_ressources.amplitudes.data()),
				thrust::raw_pointer_cast(quadtree_ressources.samples.data()),
				thrust::raw_pointer_cast(triangulate_ressources.triangles.data()));
		}

		std::vector<double> copy_image(const tfm::host::parameters &tfm_param, interpolate::device::ressources &interpolate_ressources) {
			cudaDeviceSynchronize();
			std::vector<double> image(tfm_param.size);
			cudaMemcpy(image.data(), thrust::raw_pointer_cast(interpolate_ressources.image.data()), sizeof(double) * tfm_param.size, cudaMemcpyDeviceToHost);
			return image;
		}
	}
}