#include <ctime>
#include <iostream>

#include "../IO/in.h"
#include "../IO/out.h"
#include "../TFM/TFM.h"
#include "../Quadtree/quadtree.h"
#include "../ImageInterpolation/triangulate.h"
#include "../ImageInterpolation/interpolate.h"
#include "../ProgressiveComputation/progressive_computation.h"

#define MAX_DEPTH 7
#define FIRSTS_DEPTH_LIMIT 5

int main(int argc, char* argv[])
{
	if (argc < 5) {
		std::cerr << "Usage:  argc1 -> TFM parameters file path" << std::endl;
		std::cerr << "Usage:  argc2 -> tof file path" << std::endl;
		std::cerr << "Usage:  argc3 -> fmc file path" << std::endl;
		std::cerr << "Usage:  argc4 -> out file path" << std::endl;
		return 1;
	}

	// Files path
	const std::string tof_path = argv[2];
	const std::string fmc_path = argv[3];
	const std::string image_path = argv[4];
	const std::string properties_path = argv[1];

	// TFM properties
	const tfm::host::parameters tfm_param = io::in::read_tfm_parameters(properties_path);
	const std::vector<double> tof = io::in::read_tof(tfm_param.grids * tfm_param.columns * tfm_param.rows, tof_path);
	const std::vector<short> fmc = io::in::read_fmc((tfm_param.grids / 2) * (tfm_param.grids / 2) * tfm_param.samples, fmc_path);
	const int fmc_size = static_cast<int>(fmc.size());
	const int tof_size = static_cast<int>(tof.size());

	// Quadtree properties
	quadtree::host::points points;
	points.y.resize(tfm_param.rows);
	points.x.resize(tfm_param.columns);
	points.samples.resize(tfm_param.size, 0);
	points.amplitudes.resize(tfm_param.size, 0);

	const int size = pComputation::quadtree_size(FIRSTS_DEPTH_LIMIT, MAX_DEPTH);
	const quadtree::parameters quad_param = { size, MAX_DEPTH };

	// CUDA malloc && copy to device
	quadtree::device::ressources quadtree_ressources = quadtree::cuda::quadtree_ressources(tfm_param, fmc, tof);
	
	const int nb_multiprocs = 1;
	const int nb_blocks_per_mp = 1;

	//std::pair<int, float> granularity = pComputation::cuda::granularity(nb_multiprocs, nb_blocks_per_mp, quadtree_ressources);
	//
	//printf("Best cadence : %f \n", granularity.second);
	//printf("Best granularity : %d \n", granularity.first);

	std::clock_t c_start = std::clock();

	// The Adaptative algorithm
	// STEP 1 ---- Quadtree init
	// Init root && the tree
	quadtree::cuda::init_quadtree(quadtree_ressources);
	
	// The first nodes to subdivise
	int nodes_to_subdivise = 1;
	int total_nodes_subdivised = nodes_to_subdivise;
	
	for (int depth = 1; depth <= FIRSTS_DEPTH_LIMIT; depth++) {
		quadtree::cuda::firsts_subdivisions(tfm_param.columns, depth, nodes_to_subdivise, total_nodes_subdivised, quadtree_ressources);
		nodes_to_subdivise *= 4;
		total_nodes_subdivised += nodes_to_subdivise;
	}
	
	tfm::cuda::compute_tfm(quadtree_ressources);
	pComputation::cuda::compute_score(tfm_param.columns, FIRSTS_DEPTH_LIMIT-1, quad_param, quadtree_ressources);
	pComputation::cuda::sort(quadtree_ressources);
	
	// STEP 2 ---- Refining the sampling
	for (int depth = FIRSTS_DEPTH_LIMIT + 1; depth <= quad_param.max_depth; depth++) {
		nodes_to_subdivise = 712;
		printf("nodes : %d \n", nodes_to_subdivise);
		quadtree::cuda::subdivise(tfm_param.columns, depth, nodes_to_subdivise, total_nodes_subdivised, quadtree_ressources);
		tfm::cuda::compute_tfm(quadtree_ressources);
		pComputation::cuda::compute_score(tfm_param.columns, depth, quad_param, quadtree_ressources);
		pComputation::cuda::sort(quadtree_ressources);
		total_nodes_subdivised += nodes_to_subdivise * 4;
	}
	
	// STEP 3 ---- Triangulate the samples
	// Init the triangles array to the gpu
	triangulate::device::ressources triangulate_ressources = triangulate::cuda::triangulate_ressources(tfm_param);
	int nbr_triangles = triangulate::cuda::triangulate(tfm_param, quad_param, quadtree_ressources, triangulate_ressources);
	
	// STEP 4 ---- Interpolation
	interpolate::device::ressources interpolate_ressources = interpolate::cuda::interpolate_ressources(tfm_param);
	interpolate::cuda::interpolate(nbr_triangles, tfm_param, quadtree_ressources, triangulate_ressources, interpolate_ressources);
	
	std::clock_t c_end = std::clock();
	auto time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	std::cout << "CPU time used: " << time_elapsed_ms << " ms\n";
	
	// Copy result to the host
	std::vector<quadtree::node> nodes = quadtree::cuda::copy_nodes(quad_param, quadtree_ressources);
	std::vector<double> image = interpolate::cuda::copy_image(tfm_param, interpolate_ressources);
	
	// Ouput
	io::out::txt::writeBinaryResult(image, image_path);
	io::out::image::write_samples_image(tfm_param.columns, tfm_param.rows, quad_param.tree_size, nodes.data(), image_path);
	io::out::image::write_quadTree_image(tfm_param.columns, tfm_param.rows, quad_param.tree_size, nodes.data(), image_path);
}