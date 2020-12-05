#pragma once

#include <thrust/device_vector.h>

#include "triangulate.h"
#include "../TFM/parameters.h"
#include "../Quadtree/quadtree.h"

namespace interpolate {

	namespace device {
		
		struct ressources{
			thrust::device_vector<double> image;
			ressources() = default;
			ressources(ressources&&) = default;
			ressources& operator=(ressources&&) = default;
			ressources(const ressources&) = delete;
			ressources& operator=(const ressources&) = delete;
		};
	}

	namespace cuda {
		
		std::vector<double> copy_image(const tfm::host::parameters &tfm_param, interpolate::device::ressources &interpolate_ressources);

		void interpolate(const int nbr_triangles, const tfm::host::parameters &tfm_param, const quadtree::device::ressources &quadtree_ressources,
			const triangulate::device::ressources &triangulate_ressources, interpolate::device::ressources &interpolate_ressources);

		interpolate::device::ressources interpolate_ressources(const tfm::host::parameters &tfm_param);
	}
}