#pragma once
#include <thrust/device_vector.h>

#include "../TFM/parameters.h"

#include <vector>
#include <array>


namespace quadtree {

	struct parameters {
		int tree_size;
		int max_depth;
	};

	struct node {
		int index;
		int depth;
		int points_x[9];
		int points_y[9];
		double score;
		bool ready_to_subdivise;
	};

	namespace host {
		struct points {
			std::vector<int> samples;
			std::vector<int> x;
			std::vector<int> y;
			std::vector<int> amplitudes;
		};
	};

	namespace device {

		struct ressources {
			thrust::device_vector<short> fmc;
			thrust::device_vector<double> tof;
			thrust::device_vector<bool> already_traced;
			thrust::device_vector<quadtree::node> nodes;

			thrust::device_vector<int> samples;
			thrust::device_vector<int> amplitudes;

			tfm::device::parameters tfm_param;
			thrust::device_vector<double> delays;

			ressources() = default;
			ressources(ressources&&) = default;
			ressources& operator=(ressources&&) = default;
			ressources(const ressources&) = delete;
			ressources& operator=(const ressources&) = delete;
		};
	};

	namespace cuda {
		void init_quadtree(quadtree::device::ressources &ressources);
		void subdivise(const int width, const int depth, const int nodes_to_subdivise, const int total_nodes_subdivised, quadtree::device::ressources &ressources);
		void firsts_subdivisions(const int width, const int depth, const int nodes_to_subdivise, const int total_nodes_subdivised, quadtree::device::ressources &ressources);

		std::vector<quadtree::node> copy_nodes(const quadtree::parameters &quad_param, const quadtree::device::ressources &ressources);
		quadtree::device::ressources quadtree_ressources(const tfm::host::parameters &tfm_param, const std::vector<short> &fmc, const std::vector<double> &tof);
	};
}