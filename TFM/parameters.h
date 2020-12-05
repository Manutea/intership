#pragma once
#include <thrust/device_vector.h>
#include <vector>

namespace tfm {

	namespace host {
		struct parameters {
			int size;
			int rows;
			int grids;
			int signals;
			int columns;
			int samples;
			int contributors;
			double frequency;
			std::vector<double> delays;
		};
	}

	namespace device {
		struct parameters {
			int rows;
			int grids;
			int columns;
			int samples;
			int contributors;
			double frequency;
		};
	}
}