#pragma once

#include <vector>
#include "../TFM/parameters.h"

namespace io {
	namespace in {
		tfm::host::parameters read_tfm_parameters(const std::string path);
		std::vector<short> read_fmc(const int size, const std::string path);
		std::vector<double> read_tof(const int size, const std::string path);
	}
}