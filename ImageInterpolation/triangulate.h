#pragma once

#include <thrust/device_vector.h>

#include "../TFM/parameters.h"
#include "../Quadtree/quadtree.h"

namespace triangulate {

	struct triangle {
		int v0x;
		int v0y;
		int v1x;
		int v1y;
		int v2x;
		int v2y;
	};

	namespace device {

		struct ressources
		{
			thrust::device_vector<triangle> triangles;

			ressources() = default;
			ressources(ressources&&) = default;
			ressources& operator=(ressources&&) = default;
			ressources(const ressources&) = delete;
			ressources& operator=(const ressources&) = delete;
		};
	}

	namespace cuda {
		int triangulate(const tfm::host::parameters &tfm_param, const  quadtree::parameters &quad_param, quadtree::device::ressources &quadtree_ressources, device::ressources &ressources);
	
		triangulate::device::ressources triangulate_ressources(const tfm::host::parameters &tfm_param);
	}
}