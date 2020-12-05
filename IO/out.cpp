#include "out.h"


namespace io {
	namespace out {

		namespace txt {

			void writeBinaryResult(std::vector<double> &tfms, const std::string path) {
				std::ofstream ofile(path + "result.bin", std::ios::binary);
				ofile.write(reinterpret_cast<char*>(tfms.data()), sizeof(double) * tfms.size());
			}
		}

		namespace image {

			void write_samples_image(const int width, const int height, const int max_nodes, const quadtree::node* node_to_draw, const std::string path) {
				bitmap_image image(width, height);
				image.set_all_channels(0, 0, 0);
				image_drawer draw(image);

				draw.pen_width(1);
				draw.pen_color(255, 0, 0);


				for (int index = 0; index < max_nodes; index++) {
					quadtree::node node = node_to_draw[index];

					if (node.score >= -1.0) {
						draw.plot_pixel(node.points_x[0], node.points_y[0]);
						draw.plot_pixel(node.points_x[1], node.points_y[1]);
						draw.plot_pixel(node.points_x[2], node.points_y[2]);
						draw.plot_pixel(node.points_x[3], node.points_y[3]);
						draw.plot_pixel(node.points_x[4], node.points_y[4]);
						draw.plot_pixel(node.points_x[5], node.points_y[5]);
						draw.plot_pixel(node.points_x[6], node.points_y[6]);
						draw.plot_pixel(node.points_x[7], node.points_y[7]);
						draw.plot_pixel(node.points_x[8], node.points_y[8]);
					}
				}

				image.save_image(path + "samplesQuadTreeCuda.bmp");
			}

			void write_quadTree_image(const int width, const int height, const int max_nodes, const quadtree::node* node_to_draw, const std::string path) {
				bitmap_image image(width, height);
				image.set_all_channels(0, 0, 0);
				image_drawer draw(image);

				draw.pen_width(1);
				draw.pen_color(255, 0, 0);

				for (int index = 0; index < max_nodes; index++) {
					quadtree::node node = node_to_draw[index];

					if (node.score >= -1.0) {

						int yStart = node.points_y[0];
						int yEnd = node.points_y[6];
						for (int y = yStart; y <= yEnd; y++) {
							int x = node.points_x[0];
							draw.plot_pixel(x, y);
						}

						yStart = node.points_y[0];
						yEnd = node.points_y[6];
						for (int y = yStart; y <= yEnd; y++) {
							int x = node.points_x[2];
							draw.plot_pixel(x, y);
						}

						int xStart = node.points_x[0];
						int xEnd = node.points_x[2];
						for (int x = xStart; x <= xEnd; x++) {
							int y = node.points_y[0];
							draw.plot_pixel(x, y);
						}

						xStart = node.points_x[6];
						xEnd = node.points_x[8];
						for (int x = xStart; x <= xEnd; x++) {
							int y = node.points_y[6];
							draw.plot_pixel(x, y);
						}

					}
				}
				image.save_image(path + "imageQuadTreeCuda.bmp");
			}
		}
	}
}