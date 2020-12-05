#pragma once

#include "parameters.h"
#include "../Quadtree/quadtree.h"

namespace tfm {


	namespace cuda {
		void compute_tfm(quadtree::device::ressources &ressources);
		float tfm_for_granity(const int pixels_per_iteration, quadtree::device::ressources &ressources);
	}
}