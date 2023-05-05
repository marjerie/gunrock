#include <gunrock/algorithms/bfs.hxx>
#include <gunrock/util/performance.hxx>
#include <gunrock/io/parameters.hxx>

#include "bfs_cpu.hxx"  // Reference implementation

using namespace gunrock;
using namespace memory;

void test_bfs(int num_arguments, char** argument_array) {
  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // --
  // IO

  gunrock::io::cli::parameters_t params(num_arguments, argument_array,
                                        "Breadth First Search");

  csr_t csr;
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;

  if (params.binary) {
    csr.read_binary(params.filename);
  } else {
    csr.from_coo(mm.load(params.filename));
  }

  // Data for CSC format.
  // thrust::device_vector<vertex_t> row_indices(csr.number_of_nonzeros);
  // thrust::device_vector<edge_t> column_offsets(csr.number_of_columns + 1);

  // --
  // Build graph + metadata

  auto G =
      graph::build::from_csr<memory_space_t::device,
                             graph::view_t::csr /* | graph::view_t::csc */>(
          csr.number_of_rows,               // rows
          csr.number_of_columns,            // columns
          csr.number_of_nonzeros,           // nonzeros
          csr.row_offsets.data().get(),     // row_offsets
          csr.column_indices.data().get(),  // column_indices
          csr.nonzero_values.data().get()   // values
          // row_indices.data().get(),         // row_indices
          // column_offsets.data().get()       // column_offsets
      );

  // --
  // Params and memory allocation

  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<vertex_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);
  thrust::device_vector<int> edges_visited(1);
  int search_depth = 0;

  unsigned long fill_memory_val = 1000000000;
  size_t free_byte ;
  size_t total_byte ;
  cudaMemGetInfo( &free_byte, &total_byte );
  double free_db = (double)free_byte ;
  double total_db = (double)total_byte ;
  double used_db = total_db - free_db ;
  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n"
          ,used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
  
  thrust::device_vector<vertex_t> fill_memory(2.5*fill_memory_val);
  cudaMemGetInfo( &free_byte, &total_byte );
  free_db = (double)free_byte ;
  total_db = (double)total_byte ;
  used_db = total_db - free_db ;
  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n"
          ,used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

  // Parse sources
  std::vector<int> source_vect;
  gunrock::io::cli::parse_source_string(params.source_string, &source_vect,
                                        n_vertices, params.num_runs);
  // Parse tags
  std::vector<std::string> tag_vect;
  gunrock::io::cli::parse_tag_string(params.tag_string, &tag_vect);

  // --
  // Run problem

  std::vector<float> run_times;

  for (int i = 0; i < source_vect.size(); i++) {
    // Record run times without collecting metrics (due to overhead)
    run_times.push_back(gunrock::bfs::run(
        G, source_vect[i], false, distances.data().get(),
        predecessors.data().get(), edges_visited.data().get(), &search_depth));
  }

  // Print info for last run
  std::cout << "Source : " << source_vect.back() << "\n";
  print::head(distances, 40, "GPU distances");
  std::cout << "GPU Elapsed Time : " << run_times[params.num_runs - 1]
            << " (ms)" << std::endl;

  // --
  // CPU Run

  if (params.validate) {
    thrust::host_vector<vertex_t> h_distances(n_vertices);
    thrust::host_vector<vertex_t> h_predecessors(n_vertices);

    // Validate with last source in source vector
    float cpu_elapsed = bfs_cpu::run<csr_t, vertex_t, edge_t>(
        csr, source_vect.back(), h_distances.data(), h_predecessors.data());

    int n_errors =
        util::compare(distances.data().get(), h_distances.data(), n_vertices);
    print::head(h_distances, 40, "CPU Distances");

    std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
    std::cout << "Number of errors : " << n_errors << std::endl;
  }

  // --
  // Run performance evaluation

  if (params.collect_metrics) {
    std::vector<int> edges_visited_vect;
    std::vector<int> search_depth_vect;
    std::vector<int> nodes_visited_vect(source_vect.size());

    vertex_t n_edges = G.get_number_of_edges();

    for (int i = 0; i < source_vect.size(); i++) {
      float metrics_run_time = gunrock::bfs::run(
          G, source_vect[i], params.collect_metrics, distances.data().get(),
          predecessors.data().get(), edges_visited.data().get(), &search_depth);

      thrust::host_vector<int> h_edges_visited = edges_visited;

      edges_visited_vect.push_back(h_edges_visited[0]);
      search_depth_vect.push_back(search_depth);
    }

    fill_memory[0] = 1;

    // For BFS - the number of nodes visited is just 2 * edges_visited
    std::transform(edges_visited_vect.begin(), edges_visited_vect.end(),
                   nodes_visited_vect.begin(), [](auto& c) { return 2 * c; });

    gunrock::util::stats::get_performance_stats(
        edges_visited_vect, nodes_visited_vect, n_edges, n_vertices,
        search_depth_vect, run_times, "bfs", params.filename, "market",
        params.json_dir, params.json_file, source_vect, tag_vect, num_arguments,
        argument_array);
  }
}

int main(int argc, char** argv) {
  test_bfs(argc, argv);
}
