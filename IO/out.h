#pragma once

#include "bitmap_image.h"
#include "../Quadtree/quadtree.h"

namespace io {
	namespace out {

		namespace txt {
			void writeBinaryResult(std::vector<double> &tfms, const std::string path);
		}

		namespace image {
			void write_samples_image(const int width, const int height, const int max_nodes, const quadtree::node* node_to_draw, const std::string path);
			void write_quadTree_image(const int width, const int height, const int max_nodes, const quadtree::node* node_to_draw, const std::string path);
		}
	}
}