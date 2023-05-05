#include <gunrock/algorithms/sssp.hxx>
#include "sssp_cpu.hxx"  // Reference implementation
#include <gunrock/util/performance.hxx>
#include <gunrock/io/parameters.hxx>

using namespace gunrock;
using namespace memory;

void test_sssp(int num_arguments, char** argument_array) {
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
                                        "Single Source Shortest Path");

  csr_t csr;
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;

  if (params.binary) {
    csr.read_binary(params.filename);
  } else {
    csr.from_coo(mm.load(params.filename));
  }

  // --
  // Build graph

  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr.number_of_rows,               // rows
      csr.number_of_columns,            // columns
      csr.number_of_nonzeros,           // nonzeros
      csr.row_offsets.data().get(),     // row_offsets
      csr.column_indices.data().get(),  // column_indices
      csr.nonzero_values.data().get()   // values
  );  // supports row_indices and column_offsets (default = nullptr)

  // --
  // Params and memory allocation

  srand(time(NULL));

  vertex_t n_vertices = G.get_number_of_vertices();

  thrust::device_vector<weight_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);
  thrust::device_vector<int> edges_visited(1);
  thrust::device_vector<int> vertices_visited(1);
  int search_depth = 0;

  size_t free_byte;
  size_t total_byte;
  cudaMemGetInfo(&free_byte, &total_byte);
  double free_db = (double) free_byte;
  double total_db = (double) total_byte;
  double used_db = total_db - free_db ;
  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
      used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

  std::size_t initial_size = (G.get_number_of_edges() > G.get_number_of_vertices())
              ? G.get_number_of_edges()
              : G.get_number_of_vertices();

  size_t b_to_fill = free_db - initial_size * sizeof(int) * 1.45;
  size_t num_of_elements_to_fill = b_to_fill / sizeof(int);
  std::cout << "num_of_elements_to_fill: " << num_of_elements_to_fill << std::endl;
  thrust::device_vector<int> fill_memory(num_of_elements_to_fill, 0);

  cudaMemGetInfo(&free_byte, &total_byte);
  free_db = (double) free_byte;
  total_db = (double) total_byte;
  used_db = total_db - free_db ;
  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
      used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

  // Fill memory

  // unsigned long fill_memory_val = 1000000000;

  // size_t free_byte ;
  // size_t total_byte ;
  // cudaMemGetInfo( &free_byte, &total_byte );
  // double free_db = (double)free_byte ;
  // double total_db = (double)total_byte ;
  // double used_db = total_db - free_db ;
  // printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n"
  //     ,used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

  // thrust::device_vector<vertex_t> fill_memory(5*fill_memory_val);

  // cudaMemGetInfo( &free_byte, &total_byte );
  // free_db = (double)free_byte ;
  // total_db = (double)total_byte ;
  // used_db = total_db - free_db ;
  // printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n"
  //     ,used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

  // Parse sources
  std::vector<int> source_vect;
  gunrock::io::cli::parse_source_string(params.source_string, &source_vect,
                                        n_vertices, params.num_runs);
  // Parse tags
  std::vector<std::string> tag_vect;
  gunrock::io::cli::parse_tag_string(params.tag_string, &tag_vect);

  // --
  // GPU Run

  /// An example of how one can use std::shared_ptr to allocate memory on the
  /// GPU, using a custom deleter that automatically handles deletion of the
  /// memory.
  // std::shared_ptr<weight_t> distances(
  //     allocate<weight_t>(n_vertices * sizeof(weight_t)),
  //     deleter_t<weight_t>());
  // std::shared_ptr<vertex_t> predecessors(
  //     allocate<vertex_t>(n_vertices * sizeof(vertex_t)),
  //     deleter_t<vertex_t>());

  std::vector<float> run_times;
  for (int i = 0; i < source_vect.size(); i++) {
    // Record run times without collecting metrics (due to overhead)
    run_times.push_back(gunrock::sssp::run(
        G, source_vect[i], false, distances.data().get(),
        predecessors.data().get(), edges_visited.data().get(),
        vertices_visited.data().get(), &search_depth));
  }

  // Use memory allocated to fill
  for (int i = fill_memory.size()-20; i < fill_memory.size(); i++)
    fill_memory[i] = 1;

  print::head(distances, 40, "GPU distances");
  std::cout << "GPU Elapsed Time : " << run_times[params.num_runs - 1]
            << " (ms)" << std::endl;

  // --
  // CPU Run

  if (params.validate) {
    thrust::host_vector<weight_t> h_distances(n_vertices);
    thrust::host_vector<vertex_t> h_predecessors(n_vertices);

    float cpu_elapsed = sssp_cpu::run<csr_t, vertex_t, edge_t, weight_t>(
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
    std::vector<int> nodes_visited_vect;

    vertex_t n_edges = G.get_number_of_edges();

    for (int i = 0; i < source_vect.size(); i++) {
      float metrics_run_time = gunrock::sssp::run(
          G, source_vect[i], params.collect_metrics, distances.data().get(),
          predecessors.data().get(), edges_visited.data().get(),
          vertices_visited.data().get(), &search_depth);

      thrust::host_vector<int> h_edges_visited = edges_visited;
      thrust::host_vector<int> h_vertices_visited = vertices_visited;

      edges_visited_vect.push_back(h_edges_visited[0]);
      nodes_visited_vect.push_back(h_vertices_visited[0]);
      search_depth_vect.push_back(search_depth);
    }

    gunrock::util::stats::get_performance_stats(
        edges_visited_vect, nodes_visited_vect, n_edges, n_vertices,
        search_depth_vect, run_times, "sssp", params.filename, "market",
        params.json_dir, params.json_file, source_vect, tag_vect, num_arguments,
        argument_array);
  }
}

int main(int argc, char** argv) {
  test_sssp(argc, argv);
}
