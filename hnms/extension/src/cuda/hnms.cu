#ifdef WITH_CUDA
#include "hnms.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>

#include <ctime>
#include <chrono>
using namespace std::chrono;


#define CONF_TO_INT_MULT 1000000
#define CONF_TO_INT_ADD 100000
#define CONF_TO_INT(x) (long long)((x) * CONF_TO_INT_MULT) + CONF_TO_INT_ADD

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)
const int CUDA_NUM_THREADS = 512;

int const threadsPerBlock = sizeof(unsigned long long) * 8;

inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


template <typename T>
__global__ void hnms_max_conf_kernel(long long nthreads,
        T* box_confs,
        int64_t* cell_indices,
        int64_t* cell_max_confs) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
      unsigned long long conf = CONF_TO_INT(box_confs[i]);
      unsigned long long cell = cell_indices[i];
      unsigned long long * cell_max = (unsigned long long*)(cell_max_confs + cell);
      // long long type is not supported for atomiMax
      atomicMax(cell_max, conf);
  }
}

template <typename T>
__global__ void hnms_max_idx_kernel(long long nthreads,
        T* box_confs,
        int64_t* cell_indices,
        int64_t* cell_max_confs) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
      unsigned long long conf = CONF_TO_INT(box_confs[i]);
      auto cell = cell_indices[i];
      unsigned long long* cell_max = (unsigned long long*)(cell_max_confs + cell);
      // no implementation to take long long, but unsigned long long
      atomicCAS(cell_max, conf, (unsigned long long)i);
  }
}

template <typename T>
__global__ void hash_rects_kernel(int64_t nthreads,
        T* dets,
        T w0, T h0, T alpha,
        T bx, T by,
        T alpha_ratio,
        int64_t* out) {
    CUDA_1D_KERNEL_LOOP(idx_box, nthreads) {
        auto log_w0 = log(w0);
        auto log_h0 = log(h0);
        auto log_alpha = log(alpha);

        auto curr_det = dets + idx_box * 4;
        auto x = curr_det[0];
        auto y = curr_det[1];
        auto w = curr_det[2];
        auto h = curr_det[3];
        auto w0_alpha = w0 * alpha_ratio;
        auto h0_alpha = h0 * alpha_ratio;

        auto i = round((log_w0 - log(w)) / log_alpha);
        auto j = round((log_h0 - log(h)) / log_alpha);
        auto di = w0_alpha / pow(alpha, i);
        auto dj = h0_alpha / pow(alpha, j);

        int64_t qx, qy;
        qx = round(x / di - bx);
        qy = round(y / dj - by);
        auto curr_out  = out + 4 * idx_box;
        curr_out[0] = qx;
        curr_out[1] = qy;
        curr_out[2] = i;
        curr_out[3] = j;
    }
}

at::Tensor hash_rects_cuda(const at::Tensor& dets,
               float w0,
               float h0,
               float alpha,
               float bx,
               float by) {
    auto num_box = dets.size(0);
    auto alpha_ratio = (1. - alpha) / (1. + alpha);

    auto result = at::zeros({long(num_box), 4},
            dets.options().dtype(at::kLong));

    AT_DISPATCH_FLOATING_TYPES(dets.type(), "HASH_RECTS", [&] {
            hash_rects_kernel<scalar_t><<<GET_BLOCKS(num_box), CUDA_NUM_THREADS>>>(num_box,
                    dets.data<scalar_t>(),
                    (scalar_t)w0, (scalar_t)h0, (scalar_t)alpha,
                    (scalar_t)bx, (scalar_t)by,
                    alpha_ratio,
                    result.data<int64_t>());
            });
    return result;
}

__global__ void map_code(int num_box,
        int64_t* codes,
        int64_t* codes_as_one) {
    CUDA_1D_KERNEL_LOOP(idx_box, num_box) {
        auto curr_code = codes + 4 * idx_box;
        auto curr_mapped = codes_as_one + idx_box;
        *curr_mapped = curr_code[0] +
            curr_code[1] * 10000 +
            curr_code[2] * 100000000 +
            curr_code[3] * 1000000000000;
    }
}

at::Tensor get_best_idx_each_code(
        at::Tensor codes,
        const at::Tensor& scores) {
    auto num_box = codes.size(0);
    auto codes_as_one = at::zeros({long(num_box)},
            codes.options().dtype(at::kLong));
    map_code<<<GET_BLOCKS(num_box), CUDA_NUM_THREADS>>>(num_box,
            codes.data<int64_t>(),
            codes_as_one.data<int64_t>());
    THCudaCheck(cudaGetLastError());

    auto unique_result = at::unique_dim(codes_as_one, 0, // dim
            false, true);

    at::Tensor reverse_index = std::get<1>(unique_result);
    auto count = std::get<0>(unique_result).size(0);

    auto result = at::zeros({long(count)},
            codes.options().dtype(at::kLong));

    // get the maximum confidence score for each code with the atomic operation
    // of atomicMax.
    AT_DISPATCH_FLOATING_TYPES(scores.type(), "HNMS_MAX_IDX_KERNEL", [&] {
        hnms_max_conf_kernel<scalar_t><<<GET_BLOCKS(num_box), CUDA_NUM_THREADS>>>(
                num_box,
                scores.data<scalar_t>(),
                reverse_index.data<int64_t>(),
                result.data<int64_t>());
            });
    THCudaCheck(cudaGetLastError());

    AT_DISPATCH_FLOATING_TYPES(scores.type(), "HNMS_MAX_IDX_KERNEL", [&] {
            hnms_max_idx_kernel<scalar_t><<<GET_BLOCKS(num_box), CUDA_NUM_THREADS>>>(
                    num_box,
                    scores.data<scalar_t>(),
                    reverse_index.data_ptr<int64_t>(),
                    result.data<int64_t>());
            // NULL,
            });
    return result;
}

at::Tensor hnms_cuda(const at::Tensor& dets,
               const at::Tensor& scores,
               float w0,
               float h0,
               float alpha,
               float bx,
               float by
               ) {
    AT_ASSERTM(dets.type().is_cuda(), "dets must be a CUDA tensor");
    AT_ASSERTM(scores.type().is_cuda(), "scores must be a CUDA tensor");
    AT_ASSERTM(dets.type() == scores.type(), "dets should have the same type as scores");
    if (dets.numel() == 0) {
      return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
    }

    auto codes = hash_rects_cuda(dets, w0, h0, alpha, bx, by);
    auto result = get_best_idx_each_code(codes, scores);
    return result;
}

#endif
