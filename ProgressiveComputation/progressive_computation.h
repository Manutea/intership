#pragma once

#include "../TFM/TFM.h"
#include "../TFM/parameters.h"
#include "../Quadtree/quadtree.h"

namespace pComputation {

	int quadtree_size(int firsts_depth_limit, int max_depth);
	int recovery(tfm::host::parameters parameters, quadtree::host::points points);

	namespace cuda {
		void sort(quadtree::device::ressources &ressources);
		std::pair<int, float> granularity(const int nb_multiprocs, const int nb_blocks_per_mp, quadtree::device::ressources &ressources);
		void compute_score(const int width, const int depth, const quadtree::parameters &quad_param, quadtree::device::ressources &ressources);
	}

}