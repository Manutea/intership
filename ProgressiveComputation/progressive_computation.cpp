#include "progressive_computation.h"

namespace pComputation {

	int recovery(tfm::host::parameters parameters, quadtree::host::points points) {
		int count = 0;
		for (int x = 0; x < parameters.columns; x++) {
			for (int y = 0; y < parameters.rows; y++) {
				if (points.samples[y * parameters.columns + x] == 1)
					count++;
			}
		}
		return count;
	}

	int quadtree_size(int firsts_depth_limit, int max_depth) {
		int depth;
		int nodes_to_subdivise = 1;
		int total_nodes_subdivised = nodes_to_subdivise;

		for (depth = 1; depth <= firsts_depth_limit; depth++) {
			nodes_to_subdivise *= 4;
			total_nodes_subdivised += nodes_to_subdivise;
		}

		for (int depth = firsts_depth_limit + 1; depth <= max_depth; depth++) {
			nodes_to_subdivise = 1800;
			total_nodes_subdivised += nodes_to_subdivise * 4;
		}
		return total_nodes_subdivised;
	}

}