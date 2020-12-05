#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <math.h>

#include "in.h"
#include "../TFM/parameters.h"


std::vector<std::string> explode(const std::string& s, const char c) {
	std::string buff{ "" };
	std::vector<std::string> v;
	for (auto n : s) {
		if (n != c) buff += n;
		else if (n == c && buff != "") { v.push_back(buff); buff = ""; }
	}
	if (buff != "") v.push_back(buff);
	return v;
}

namespace io {
	namespace in {

		tfm::host::parameters read_tfm_parameters(const std::string path) {
			tfm::host::parameters param;
			std::ifstream file(path);
			if (file.is_open()) {
				std::string line;
				while (getline(file, line)) {
					if (line.find("selection") != std::string::npos) {
						std::vector<std::string>exploded = explode(line, '=');
						std::stoi(exploded.at(1));
					}
					else if (line.find("signals") != std::string::npos) {
						std::vector<std::string>exploded = explode(line, '=');
						param.signals = std::stoi(exploded.at(1));
					}
					else if (line.find("contributors") != std::string::npos) {
						std::vector<std::string>exploded = explode(line, '=');
						param.contributors = std::stoi(exploded.at(1));
					}
					else if (line.find("samples") != std::string::npos) {
						std::vector<std::string>exploded = explode(line, '=');
						param.samples = std::stoi(exploded.at(1));
					}
					else if (line.find("grids") != std::string::npos) {
						std::vector<std::string>exploded = explode(line, '=');
						param.grids = std::stoi(exploded.at(1));
					}
					else if (line.find("rows") != std::string::npos) {
						std::vector<std::string>exploded = explode(line, '=');
						param.rows = std::stoi(exploded.at(1));
					}
					else if (line.find("columns") != std::string::npos) {
						std::vector<std::string>exploded = explode(line, '=');
						param.columns = std::stoi(exploded.at(1));
					}
					else if (line.find("frequency") != std::string::npos) {
						std::vector<std::string>exploded = explode(line, '=');

						std::string s = "0x" + exploded.at(1);
						int64_t x = std::strtoull(s.c_str(), NULL, 0);
						param.frequency = *reinterpret_cast<double *>(&x);
					}
					else if (line.find("v_delays") != std::string::npos) {
						std::vector<std::string>exploded = explode(line, '=');
						std::vector<std::string>s_delay = explode(exploded.at(1), ',');

						for (int i = 0; i < s_delay.size(); i++) {
							std::string s = "0x" + s_delay.at(i);
							int64_t x = std::strtoull(s.c_str(), NULL, 0);
							double dbl = *reinterpret_cast<double *>(&x);
							param.delays.push_back(dbl);
						}
					}
				}
				param.size = param.columns * param.rows;

				file.close();
			}
			return param;
		}

		std::vector<short> read_fmc(const int size, const std::string path) {
			std::vector<short> result(size);
			std::ifstream fin(path.c_str(), std::ios::in | std::ios::binary);
			fin.read(reinterpret_cast<char*>(result.data()), sizeof(short) * size);
			return result;
		}

		std::vector<double> read_tof(const int size, const std::string path) {
			std::vector<double> result(size);
			std::ifstream fin(path.c_str(), std::ios::in | std::ios::binary);
			fin.read(reinterpret_cast<char*>(result.data()), sizeof(double) * size);
			return result;
		}
	};
}