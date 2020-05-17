#pragma once
#include <torch/extension.h>

#ifdef WITH_CUDA
at::Tensor hnms_cuda(const at::Tensor& dets,
               const at::Tensor& scores,
               float w0,
               float h0,
               float alpha,
               float bx,
               float by
               );
#endif


at::Tensor hnms_cpu(const at::Tensor& dets,
               const at::Tensor& scores,
               float w0,
               float h0,
               float alpha,
               float bx,
               float by
               );


at::Tensor hnms(const at::Tensor& dets,
               const at::Tensor& scores,
               float w0,
               float h0,
               float alpha,
               float bx,
               float by
               );
