#include <gunrock/algorithms/spgemm_try.hxx>

using namespace gunrock;
using namespace memory;

void test_spmv(int num_arguments, char** argument_array) {
  if (num_arguments != 4) {
    std::cerr << "usage: ./bin/<program-name> a.mtx b.mtx" << std::endl;
    exit(1);
  }

  // --
  // Define types
  // Specify the types that will be used for
  // - vertex ids (vertex_t)
  // - edge offsets (edge_t)
  // - edge weights (weight_t)

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  // Filename to be read
  std::string filename_a = argument_array[1];
  constexpr memory_space_t space = memory_space_t::device;

  /// Load the matrix-market dataset into csr format.
  /// See `format` to see other supported formats.
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  using csr_t = format::csr_t<space, vertex_t, edge_t, weight_t>;
  csr_t a_csr;
  a_csr.from_coo(mm.load(filename_a));

  auto A = graph::build::from_csr<space, graph::view_t::csr>(
      a_csr.number_of_rows, a_csr.number_of_columns, a_csr.number_of_nonzeros,
      a_csr.row_offsets.data().get(), a_csr.column_indices.data().get(),
      a_csr.nonzero_values.data().get());

  std::string filename_b = argument_array[2];
  csr_t b_csr;
  b_csr.from_coo(mm.load(filename_b));

  thrust::device_vector<vertex_t> row_indices(b_csr.number_of_nonzeros);
  thrust::device_vector<edge_t> column_offsets(b_csr.number_of_columns + 1);

  /// For now, we are using the transpose of CSR-matrix A as the second operand
  /// for our spgemm.
  auto B_csr = graph::build::from_csr<space, graph::view_t::csr>(
      b_csr.number_of_rows, b_csr.number_of_columns, b_csr.number_of_nonzeros,
      b_csr.row_offsets.data().get(), b_csr.column_indices.data().get(),
      b_csr.nonzero_values.data().get());

  auto B = graph::build::from_csr<space, graph::view_t::csc>(
      b_csr.number_of_rows, b_csr.number_of_columns, b_csr.number_of_nonzeros,
      b_csr.row_offsets.data().get(), b_csr.column_indices.data().get(),
      b_csr.nonzero_values.data().get(),
      row_indices.data().get(),         
      column_offsets.data().get());

// printf("A\n");
//     printf("row_offsets[:%lu] = ", a_csr.row_offsets.size());
//     for (int i = 0; i < a_csr.row_offsets.size(); i++)
//       std::cout <<  a_csr.row_offsets[i] << ' ';
//     printf("\n");
//     printf("column_indices[:%lu] = ", a_csr.column_indices.size());
//     for (int i = 0; i < a_csr.column_indices.size(); i++)
//       std::cout <<  a_csr.column_indices[i] << ' ';
//     printf("\n");
//     printf("nonzero_values[:%lu] = ", a_csr.nonzero_values.size());
//     for (int i = 0; i < a_csr.nonzero_values.size(); i++)
//       std::cout <<  a_csr.nonzero_values[i] << ' ';
//     printf("\n");

//     printf("B\n");
//     printf("row_offsets[:%lu] = ", b_csr.row_offsets.size());
//     for (int i = 0; i < b_csr.row_offsets.size(); i++)
//       std::cout <<  b_csr.row_offsets[i] << ' ';
//     printf("\n");
//     printf("column_indices[:%lu] = ", b_csr.column_indices.size());
//     for (int i = 0; i < b_csr.column_indices.size(); i++)
//       std::cout <<  b_csr.column_indices[i] << ' ';
//     printf("\n");
//     printf("nonzero_values[:%lu] = ", b_csr.nonzero_values.size());
//     for (int i = 0; i < b_csr.nonzero_values.size(); i++)
//       std::cout <<  b_csr.nonzero_values[i] << ' ';
//     printf("\n");


  /// Let's use CSR representation
  csr_t C;

  size_t mb_to_fill = atoi(argument_array[3]);
  std::cout << "mb_to_fill: " << mb_to_fill << std::endl;
  size_t num_of_elements_to_fill = mb_to_fill * 1024 * 1024 / sizeof(int);
  thrust::device_vector<int> fill_memory(num_of_elements_to_fill, 0);

  size_t free_byte;
  size_t total_byte;
  cudaMemGetInfo(&free_byte, &total_byte);
  double free_db = (double) free_byte;
  double total_db = (double) total_byte;
  double used_db = total_db - free_db ;
  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
      used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

  cudaStream_t streamk;
  cudaStreamCreate(&streamk);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    // Handle the error.
    std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
  } else std::cout << "cudaStreamCreate success!" << '\n';
  unsigned long long streamId;
  cudaStreamGetId(streamk, &streamId);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    // Handle the error.
    std::cout << "cudaStreamGetId Error: " << cudaGetErrorString(err) << std::endl;
  }

  cudaDeviceSynchronize();

  std::cout << "cudaStream: " << streamId << '\n';

  std::shared_ptr<gcuda::multi_context_t> context = 
      std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0, streamk));

  // --
  // GPU Run
  float gpu_elapsed = gunrock::spgemm_try::run(A, B_csr, B, C, context);

  for (int i = fill_memory.size()-20; i < fill_memory.size(); i++)
      fill_memory[i] = 1;

  std::cout << "Number of rows: " << C.number_of_rows << std::endl;
  std::cout << "Number of columns: " << C.number_of_columns << std::endl;
  std::cout << "Number of nonzeros: " << C.number_of_nonzeros << std::endl;

  print::head(C.row_offsets, 10, "row_offsets");
  print::head(C.column_indices, 10, "column_indices");
  print::head(C.nonzero_values, 10, "nonzero_values");

  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
}

// Main method, wrapping test function
int main(int argc, char** argv) {
  test_spmv(argc, argv);
}