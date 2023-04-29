/**
 * @file spgemm.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse-Matrix-Matrix multiplication.
 * @date 2022-01-04
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <gunrock/algorithms/algorithms.hxx>

// Thrust includes (scan, reduce)
#include <thrust/reduce.h>
#include <thrust/scan.h>

namespace gunrock {
namespace spgemm_try {

template <typename graph_type, typename graph_t>
struct param_t {
  graph_type& A;
  graph_type& B_csr;
  graph_t& B;
  param_t(graph_type& _A, graph_type& _B_csr, graph_t& _B) : A(_A), B_csr(_B_csr), B(_B) {}
};

template <typename csr_t>
struct result_t {
  csr_t& C;
  result_t(csr_t& _C) : C(_C) {}
};

template <typename graph_type, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_type> {
  using edge_t = typename graph_type::edge_type;
  using vertex_t = typename graph_type::vertex_type;
  using weight_t = typename graph_type::weight_type;

  param_type& param;
  result_type& result;

  problem_t(graph_type& A,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::problem_t<graph_type>(A, _context),
        param(_param),
        result(_result) {}

  thrust::device_vector<edge_t> estimated_nz_per_row;
  thrust::host_vector<edge_t> nnz;

  void init() override {
    auto& A = this->param.A;
    // auto& C = this->result.C;

    estimated_nz_per_row.resize(A.get_number_of_vertices());
    nnz.resize(1);
  }

  void reset() override {
    auto policy = this->context->get_context(0)->execution_policy();
    thrust::fill(policy, estimated_nz_per_row.begin(),
                 estimated_nz_per_row.end(), 0);

    // Reset NNZ.
    nnz[0] = 0;

    // Reset sparse-matrix C.
    auto& C = this->result.C;
    C.row_offsets.clear();
    C.column_indices.clear();
    C.nonzero_values.clear();
  }
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context,
            enactor_properties_t _properties = enactor_properties_t())
      : gunrock::enactor_t<problem_t>(_problem, _context, _properties) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;

  void loop(gcuda::multi_context_t& context) override {
    auto E = this->get_enactor();
    auto P = this->get_problem();

    auto policy = this->context->get_context(0)->execution_policy();

    auto& A = P->param.A;
    auto& B_csr = P->param.B_csr;
    auto& B = P->param.B;
    auto& C = P->result.C;

    auto& row_offsets = C.row_offsets;
    auto& column_indices = C.column_indices;
    auto& nonzero_values = C.nonzero_values;

    auto& estimated_nz_per_row = P->estimated_nz_per_row;
    auto estimated_nz_ptr = estimated_nz_per_row.data().get();

    // Resize row-offsets.
    // row_offsets.resize(A.get_number_of_vertices() + 1);

    thrust::device_vector<edge_t, thrust::virtual_allocator<edge_t>> row_offsets_vm_d;
    auto& row_offsets_vm = row_offsets_vm_d;
    row_offsets_vm.resize(A.get_number_of_vertices() + 1);

    thrust::fill(policy, estimated_nz_per_row.begin(),
                 estimated_nz_per_row.end(), 0);

    /// Step 1. Count the upperbound of number of nonzeros per row of C.
    auto upperbound_nonzeros =
        [=] __host__ __device__(vertex_t const& m,  // ... source (row index)
                                vertex_t const& k,  // neighbor (column index)
                                edge_t const& nz_idx,  // edge (row ↦ column)
                                weight_t const& nz     // weight (nonzero).
                                ) -> bool {
      // Compute number of nonzeros of the sparse-matrix C for each row.
      math::atomic::add(&(estimated_nz_ptr[m]), B_csr.get_number_of_neighbors(k));
      return false;
    };

    operators::advance::execute<operators::load_balance_t::block_mapped,
                                operators::advance_direction_t::forward,
                                operators::advance_io_type_t::graph,
                                operators::advance_io_type_t::none>(
        A, E, upperbound_nonzeros, context);

    // printf("estimated nnz per row:\n");
    // for (auto it: P->estimated_nz_per_row) 
    //   std::cout << it << '\n';

    // printf("---------------------------------------------------\n");

    // printf("estimated_nz_per_row[:%lu] = ", P->estimated_nz_per_row.size());
    // for (int i = 0; i < P->estimated_nz_per_row.size(); i++)
    //   std::cout <<  P->estimated_nz_per_row[i] << ' ';
    // printf("\n");

    /// Step X. Calculate upperbound of C's row-offsets.
    thrust::exclusive_scan(policy, P->estimated_nz_per_row.begin(),
                           P->estimated_nz_per_row.end() + 1, row_offsets_vm.begin(),
                           edge_t(0), thrust::plus<edge_t>());
    
    // printf("row_offsets[:%lu] = ", row_offsets.size());
    // for (int i = 0; i < row_offsets.size(); i++)
    //   std::cout << row_offsets[i] << ' ';
    // printf("\n");

    /// Step X. Calculate the upperbound of total number of nonzeros in the
    /// sparse-matrix C.
    thrust::copy(row_offsets_vm.begin() + A.get_number_of_vertices(),
                 row_offsets_vm.begin() + A.get_number_of_vertices() + 1,
                 P->nnz.begin());

    edge_t estimated_nzs = P->nnz[0];

    // std::cout << "estimated nzs : " << estimated_nzs << '\n';

    /// Step . Allocate upperbound memory for C's values and column indices.
    // column_indices.resize(estimated_nzs, -1);
    // nonzero_values.resize(estimated_nzs, weight_t(0));

    // edge_t* row_off = row_offsets.data().get();
    // vertex_t* col_ind = column_indices.data().get();
    // weight_t* nz_vals = nonzero_values.data().get();

    // thrust::device_vector<weight_t, thrust::virtual_allocator<weight_t>> nz_vals_vm(estimated_nzs, weight_t(0));
    // thrust::device_vector<weight_t> nz_vals_vm(estimated_nzs, weight_t(0));

      thrust::device_vector<weight_t, thrust::virtual_allocator<weight_t>> nz_vals_vm_d;
      auto& nz_vals_vm = nz_vals_vm_d;
      nz_vals_vm.resize(estimated_nzs, weight_t(0));
      weight_t* nz_vals_vm_ptr = nz_vals_vm.data().get();

      thrust::device_vector<vertex_t, thrust::virtual_allocator<vertex_t>> column_indices_vm_d;
      auto& column_indices_vm = column_indices_vm_d;
      column_indices_vm.resize(estimated_nzs, -1);
      vertex_t* column_indices_vm_ptr = column_indices_vm.data().get();

      edge_t* row_offsets_vm_ptr = row_offsets_vm.data().get();

      size_t initial_size = 5242880;
      //initial_size = 524288*5;
      size_t size_per_thread = initial_size / A.get_number_of_vertices();

      thrust::device_vector<weight_t, thrust::virtual_allocator<weight_t>> nz_vals;
      // thrust::device_vector<weight_t> nz_vals;
      nz_vals.resize(initial_size, weight_t(0));
      volatile weight_t* nz_vals_ptr = nz_vals.data().get();

      std::cout << "nz_vals.data().get()" << nz_vals.data().get() << '\n';

      // thrust::device_vector<int> num_of_extra_chunks(1, 0);
      // int* num_of_extra_chunks_ptr = num_of_extra_chunks.data().get();

    // weight_t* nz_vals_vm_ptr = nz_vals_vm.data().get();

    // int dis_val = min(10, estimated_nzs);
    // printf("nz_vals_vm[:10] = ");
    // for (int i = 0; i < dis_val; i++)
    //   std::cout << nz_vals_vm[i] << ' ';
    // printf("\n");

    // printf("nonzero_values[:10] = ");
    // for (int i = 0; i < dis_val; i++)
    //   std::cout << nonzero_values[i] << ' ';
    // printf("\n");

    // std::cout << "min float : " << std::numeric_limits<float>::min() << std::endl;

    // // to store ptrs to vm row vectors
    // thrust::device_vector<float *> vm_vector_ptrs(A.get_number_of_vertices());

    // // vm vector for each row
    // for (int i = 0; i < A.get_number_of_vertices(); i++) {
    //   thrust::device_vector<float, thrust::virtual_allocator<float>> P->estimated_nz_per_row[i]
    // }

    /// Step X. Calculate C's column indices and values.

    // size_t free_byte;
    // size_t total_byte;
    // cudaMemGetInfo(&free_byte, &total_byte);
    // double free_db = (double) free_byte;
    // double total_db = (double) total_byte;
    // double used_db = total_db - free_db ;
    // printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
    //     used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
    
    // size_t num_of_elements_to_fill = free_db / 1.001 / sizeof(int);
    // thrust::device_vector<int> fill_memory(num_of_elements_to_fill, 0);

    // cudaMemGetInfo(&free_byte, &total_byte);
    // free_db = (double) free_byte;
    // total_db = (double) total_byte;
    // used_db = total_db - free_db ;
    // printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
    //     used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

    auto naive_spgemm = [=] __host__ __device__(vertex_t const& row) -> bool {
      // Get the number of nonzeros in row of sparse-matrix A.
      auto a_offset = A.get_starting_edge(row);
      auto a_nnz = A.get_number_of_neighbors(row);
      auto c_offset = thread::load(&row_offsets_vm_ptr[row]);
      auto n = 0;
      bool increment = false;

      // iterate over the columns of B.
      for (edge_t b_col = 0; b_col < B.get_number_of_vertices(); ++b_col) {
        // Get the number of nonzeros in column of sparse-matrix B.
        auto b_offset = B.get_starting_edge(b_col);
        auto b_nnz = B.get_number_of_neighbors(b_col);
        auto b_nz_idx = b_offset;
        auto a_nz_idx = a_offset;

        // For the row in A, multiple with corresponding element in B.
        while ((a_nz_idx < (a_offset + a_nnz)) && (b_nz_idx < (b_offset + b_nnz))) {
          auto a_col = A.get_destination_vertex(a_nz_idx);
          auto b_row = B.get_source_vertex(b_nz_idx);

          //  Multiply if the column of A equals row of B.
          if (a_col == b_row) {
              auto a_nz = A.get_edge_weight(a_nz_idx);
              auto b_nz = B.get_edge_weight(b_nz_idx);

              // Calculate  C's nonzero index.
              std::size_t c_nz_idx = c_offset + n;
              assert(c_nz_idx < estimated_nzs);

              // Assign column index.
              thread::store(&column_indices_vm_ptr[c_nz_idx], n);
              // thread::store(&col_ind[c_nz_idx], n);

              // Accumulate the nonzero value.
              nz_vals_vm_ptr[c_nz_idx] += a_nz * b_nz;
              // nz_vals[c_nz_idx] += a_nz * b_nz;

              a_nz_idx++;
              b_nz_idx++;

              increment = true;
          }
          else if (a_col < b_row) a_nz_idx++;
          else b_nz_idx++;
        }
        // a non zero element was stored in C, so we increment n
        if (increment) {
          n++;
          increment = false;
        }
      }
      return false;
    };

    operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
        A, naive_spgemm, context);

    cuInit(0);

    int pi;
    cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, 0);

    if (!pi) {
        throw std::runtime_error{"Device cannot use host pointer for registered memory."};
    }

    // volatile size_t* cpu_new_size, *new_size;
    volatile int* cpu_extra_chunk_size, *extra_chunk_size, *h_data, *real_chunk_size, *h_real;
    volatile bool* vector_ready, *h_ready, *cpu_vector_ready;
    int actual_extra_chunk_size = 0;

    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      // Handle the error.
      std::cout << "cudaSetDeviceFlags Error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaHostAlloc((void **)&h_data, sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc((void **)&h_ready, sizeof(bool), cudaHostAllocMapped);
    cudaHostAlloc((void **)&h_real, sizeof(int), cudaHostAllocMapped);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      // Handle the error.
      std::cout << "cudaHostAlloc Error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaHostGetDevicePointer((int **)&extra_chunk_size, (int *)h_data, 0);
    cudaHostGetDevicePointer((bool **)&vector_ready, (bool *)h_ready, 0);
    cudaHostGetDevicePointer((int **)&real_chunk_size, (int *)h_real, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      // Handle the error.
      std::cout << "cudaHostGetDevicePointer Error: " << cudaGetErrorString(err) << std::endl;
    }
    *h_data = 0;
    *extra_chunk_size = 0;
    *vector_ready = true;
    *real_chunk_size = 0;

    // gets new size from kernel
    int *atomic_extra_chunk_size;
    cudaMalloc((int **)&atomic_extra_chunk_size, sizeof(int));
    cudaMemset((void *)atomic_extra_chunk_size, 0x0, sizeof(int));

    auto dynamic_spgemm = [=] __host__ __device__(vertex_t const& row) -> bool {
      // Get the number of nonzeros in row of sparse-matrix A.
      auto a_offset = A.get_starting_edge(row);
      auto a_nnz = A.get_number_of_neighbors(row);
      auto c_offset = row * size_per_thread;
      auto n = 0;
      int num_of_extra_chunks = 0;
      bool increment = false;

      // iterate over the columns of B.
      for (edge_t b_col = 0; b_col < B.get_number_of_vertices(); ++b_col) {
        // Get the number of nonzeros in column of sparse-matrix B.
        auto b_offset = B.get_starting_edge(b_col);
        auto b_nnz = B.get_number_of_neighbors(b_col);
        auto b_nz_idx = b_offset;
        auto a_nz_idx = a_offset;

        //if (row == 0) printf("for loop: %d %d\n", num_of_extra_chunks, *extra_chunk_size);

        // For the row in A, multiple with corresponding element in B.
        while ((a_nz_idx < (a_offset + a_nnz)) && (b_nz_idx < (b_offset + b_nnz))) {
          auto a_col = A.get_destination_vertex(a_nz_idx);
          auto b_row = B.get_source_vertex(b_nz_idx);

          //  Multiply if the column of A equals row of B.
          if (a_col == b_row) {
              auto a_nz = A.get_edge_weight(a_nz_idx);
              auto b_nz = B.get_edge_weight(b_nz_idx);

              // if (n == size_per_thread/4) {
              //   if (*real_chunk_size <= num_of_extra_chunks) {
              //     // printf("stuck here... ");
              //     int old = math::atomic::cas((int*) atomic_extra_chunk_size, num_of_extra_chunks, 
              //         num_of_extra_chunks + 1);
              //     *extra_chunk_size = *atomic_extra_chunk_size;
              //     //while (*real_chunk_size <= num_of_extra_chunks);
              //   }
              // }

              if (n >= size_per_thread) {
                n = 0;
                if (*real_chunk_size <= num_of_extra_chunks) {
                  // printf("stuck here... ");
                  int old = math::atomic::cas((int*) atomic_extra_chunk_size, num_of_extra_chunks, 
                      num_of_extra_chunks + 1);
                  *extra_chunk_size = *atomic_extra_chunk_size;
                  while (*real_chunk_size <= num_of_extra_chunks);
                }
              
                while (*real_chunk_size <= num_of_extra_chunks);
                num_of_extra_chunks++;
              }

              // Calculate  C's nonzero index.
              std::size_t c_nz_idx = c_offset + initial_size * num_of_extra_chunks + n;
              //assert(c_nz_idx < estimated_nzs);

              // Assign column index.
              // thread::store(&column_indices_vm_ptr[c_nz_idx], n);
              // thread::store(&col_ind[c_nz_idx], n);

              // Accumulate the nonzero value.
              // assert(*real_chunk_size >= num_of_extra_chunks);
              if (*real_chunk_size < num_of_extra_chunks) printf("%d %d\n", *real_chunk_size, num_of_extra_chunks);
              assert(c_nz_idx < (initial_size * (num_of_extra_chunks + 1)));
              // nz_vals_ptr[c_nz_idx] += a_nz * b_nz;

              a_nz_idx++;
              b_nz_idx++;

              increment = true;
          }
          else if (a_col < b_row) a_nz_idx++;
          else b_nz_idx++;
        }
        // a non zero element was stored in C, so we increment n
        if (increment) {
          n++;
          increment = false;
        }
      }
      return false;
    };

    operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
        A, dynamic_spgemm, context);

    // unsigned long long streamId;
    // cudaStreamGetId(context.get_stream(), &streamId);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //   // Handle the error.
    //   std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    // }

    // std::cout << "cudaStreamQuery: " << cudaStreamQuery(context.get_stream()) << '\n';
    // std::cout << "cudaStream: " << streamId << '\n';
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //   // Handle the error.
    //   std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    // }

    std::cout << "cudaStreamQuery: " << context.get_stream_status() << '\n';
    std::cout << "cudaStream: " << context.get_stream_id() << '\n';
    
    // if (cudaStreamQuery(context.get_stream()) == cudaSuccess) printf("uhhh....\n");
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //   // Handle the error.
    //   std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    // }

    if (context.get_stream_status() == cudaSuccess) printf("uhhh....\n");

    // while (cudaStreamQuery(context.get_stream()) != cudaSuccess) {
    while (context.get_stream_status() == cudaSuccess) {
      //printf("yay here... ");
      // cpu_extra_chunk_size = extra_chunk_size;
      int chunk_size = *extra_chunk_size;
      cpu_vector_ready = vector_ready;
      // if (*cpu_extra_chunk_size > actual_extra_chunk_size) {
      if (chunk_size > actual_extra_chunk_size) {
        // printf("yay here... %d\n", *cpu_extra_chunk_size);
        printf("yay here... %d\n", chunk_size);
        *cpu_vector_ready = false;
        // nz_vals.resize((*(cpu_extra_chunk_size) + 1) * initial_size);
        nz_vals.resize((chunk_size + 1) * initial_size);
        std::cout << "done resizing!\n";
        // actual_extra_chunk_size = *cpu_extra_chunk_size;
        nz_vals_ptr = nz_vals.data().get();
        actual_extra_chunk_size = chunk_size;
        *real_chunk_size = actual_extra_chunk_size;
        *cpu_vector_ready = true;
        std::cout << "actual_extra_chunk_size: " << actual_extra_chunk_size << '\n';
        std::cout << "nz_vals_ptr: " << nz_vals_ptr << '\n';
        std::cout << "nz_vals.data().get()" << nz_vals.data().get() << '\n';
        //nz_vals_ptr = nz_vals.data().get();
      }
    }

    std::cout << "after.. cudaStreamQuery: " << context.get_stream_status() << '\n';

    //context.synchronize_stream();  

    // printf("nz_vals_vm[:%lu] = ", nz_vals_vm.size());
    // for (auto it: nz_vals_vm) 
    //   std::cout << it << ' ';
    // printf("\n");

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      // Handle the error.
      std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    }

    // printf("---------------------------------------------------\n");


    // int dis_val = min(30, estimated_nzs);

    // printf("nz_vals_vm[:10] = ");
    // for (int i = 0; i < dis_val; i++)
    //   std::cout << nz_vals_vm[i] << ' ';
    // printf("\n");

    // printf("nonzero_values[:10] = ");
    // for (int i = 0; i < dis_val; i++)
    //   std::cout << nonzero_values[i] << ' ';
    // printf("\n---------------------------------------------------\n");

    /// Step X. Fix-up, i.e., remove overestimated nonzeros and rellocate the
    /// storage as necessary.
    auto real_nonzeros = [=] __host__ __device__(vertex_t const& row) -> void {
      edge_t overestimated_nzs = 0;
      // For all nonzeros within the row of C.
      for (auto nz = row_offsets_vm_ptr[row]; nz < row_offsets_vm_ptr[row + 1]; ++nz) {
        // Find the invalid column indices and zero-values, they represent
        // overestimated nonzeros.
        // if (col_ind[nz] == -1)
        if (column_indices_vm_ptr[nz] == -1)
          overestimated_nzs += 1;
      }
      // Remove overestimated nonzeros.
      estimated_nz_ptr[row] -= overestimated_nzs;
    };

    operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
        A, real_nonzeros, context);
    
    // cudaDeviceSynchronize();

    // printf("real_nonzeros kernel done\n");

    thrust::exclusive_scan(policy, P->estimated_nz_per_row.begin(),
                           P->estimated_nz_per_row.end(), row_offsets_vm.begin(),
                           edge_t(0), thrust::plus<edge_t>());

    thrust::copy(row_offsets_vm.begin() + A.get_number_of_vertices() - 1,
                 row_offsets_vm.begin() + A.get_number_of_vertices(),
                 row_offsets_vm.begin() + A.get_number_of_vertices());

    /// Step X. Calculate the upperbound of total number of nonzeros in the
    /// sparse-matrix C.
    thrust::copy(row_offsets_vm.begin() + A.get_number_of_vertices(),
                 row_offsets_vm.begin() + A.get_number_of_vertices() + 1,
                 P->nnz.begin());


    auto itcVM = thrust::copy_if(
        policy, column_indices_vm.begin(), column_indices_vm.end(),
        column_indices_vm.begin(),
        [] __device__(const vertex_t& x) -> bool { return x != -1; });

    // auto itc = thrust::copy_if(
    //     policy, column_indices.begin(), column_indices.end(),
    //     column_indices.begin(),
    //     [] __device__(const vertex_t& x) -> bool { return x != -1; });


    // cudaDeviceSynchronize();

    // printf("copy_if on column_indices done\n");

    auto itVM = thrust::copy_if(policy, nz_vals_vm.begin(),
                               nz_vals_vm.end(), nz_vals_vm.begin(),
                               [] __device__(const weight_t& nz) -> bool {
                                 return nz != weight_t(0);
                               });

    // auto itv = thrust::copy_if(policy, nonzero_values.begin(),
    //                            nonzero_values.end(), nonzero_values.begin(),
    //                            [] __device__(const weight_t& nz) -> bool {
    //                              return nz != weight_t(0);
    //                            }); 

    // cudaDeviceSynchronize();

    // printf("copy_if on nz_vals_vm done\n");    

    // printf("AFTER vm result vector C:\n");
    // for (auto it: nz_vals_vm) 
    //   std::cout << it << '\n';

    auto nz_nnz = thrust::distance(nz_vals_vm.begin(), itVM);
    // auto nz_nnz = thrust::distance(nonzero_values.begin(), itv);

    // for (int i = 0; i < nz_nnz; i++) {
    //   assert(nz_vals_vm[i] == nonzero_values[i]);
    // }

    // int display_num = min(20, (int) nz_nnz);

    // for (int i = 0; i < 20; i++)
    //   fill_memory[i] = 1;

    // printf("nz_vals_vm[:%d] = ", display_num);
    // for (int i = 0; i < display_num; i++)
    //   std::cout << nz_vals_vm[i] << ' ';
    // printf("\n");

    // printf("nonzero_values[:%d] = ", display_num);
    // for (int i = 0; i < display_num; i++)
    //   std::cout << nonzero_values[i] << ' ';
    // printf("\n");

    // auto idx_nnz = thrust::distance(column_indices.begin(), itc);
    // auto nz_nnz = thrust::distance(nonzero_values.begin(), itv);

    // std::cout << "idx_nnz ? nz_nnz : " << idx_nnz << " ? " << nz_nnz
    //           << std::endl;

    /// Step X. Make sure C is set.
    C.number_of_rows = A.get_number_of_vertices();
    C.number_of_columns = B.get_number_of_vertices();
    C.number_of_nonzeros = P->nnz[0];
  }

  /**
   * @brief SpGEMM converges within one iteration.
   *
   * @param context The context of the execution (unused).
   * @return true returns true after one iteration.
   */
  virtual bool is_converged(gcuda::multi_context_t& context) {
    return this->iteration == 0 ? false : true;
  }
};  // struct enactor_t

template <typename graph_type, typename graph_t, typename csr_t>
float run(graph_type& A, graph_type& B_csr,
          graph_t& B,
          csr_t& C,
          std::shared_ptr<gcuda::multi_context_t> context  // Context
) {
  using param_type = param_t<graph_type, graph_t>;
  using result_type = result_t<csr_t>;

  param_type param(A, B_csr, B);
  result_type result(C);

  using problem_type = problem_t<graph_type, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(A, param, result, context);
  problem.init();
  problem.reset();

  // Disable internal-frontiers:
  enactor_properties_t props;
  props.self_manage_frontiers = true;

  enactor_type enactor(&problem, context, props);
  return enactor.enact();
}

}  // namespace spgemm
}  // namespace gunrock