#include "hnms.h"
#include <tgmath.h>
#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>
#include <map>
#include <vector>


at::Tensor hash_rects(const at::Tensor& dets,
               float w0,
               float h0,
               float alpha,
               float bx,
               float by) {
    auto log_w0 = log(w0);
    auto log_h0 = log(h0);
    auto log_alpha = log(alpha);

    // map the rects to the code
    auto x = dets.select(1, 0).contiguous();
    auto y = dets.select(1, 1).contiguous();
    auto w = dets.select(1, 2).contiguous();
    auto h = dets.select(1, 3).contiguous();
    auto alpha_ratio = (1. - alpha) / (1. + alpha);
    auto w0_alpha = w0 * alpha_ratio;
    auto h0_alpha = h0 * alpha_ratio;

    auto i = at::round((log_w0 - at::log(w)) / log_alpha);
    auto j = at::round((log_h0 - at::log(h)) / log_alpha);

    auto di = w0_alpha / at::pow(alpha, i);
    auto dj = h0_alpha / at::pow(alpha, j);

    at::Tensor qx, qy;
    qx = at::round(x / di - bx);
    qy = at::round(y / dj - by);
    auto result = at::stack({qx, qy, i, j}, 1);
    return at::_cast_Long(result).contiguous();
}

typedef long TCode;
TCode get_code(const long* p_code) {
    return p_code[0] + p_code[1] * 10000 +
        p_code[2] * 100000000 + p_code[3] * 1000000000000;
}

at::Tensor get_best_score_each_code(
        at::Tensor codes,
        const at::Tensor& scores) {
    std::map<TCode, long> code_to_idx;

    auto p_code = codes.data<long>();
    auto p_score = scores.data<float>();

    auto ndets = codes.size(0);
    for (auto i = 0; i < ndets; i++) {
        auto code = get_code(p_code);
        if (code_to_idx.count(code) == 0) {
            code_to_idx[code] = i;
        } else {
            auto &pre_idx = code_to_idx[code];
            if (p_score[pre_idx] < p_score[i]) {
                pre_idx = i;
            }
        }
        p_code += 4;
    }

    at::Tensor result = at::ones({long(code_to_idx.size())},
            scores.options().dtype(at::kLong).device(at::kCPU));
    auto p = result.data<long>();
    int idx = 0;
    for (auto i = code_to_idx.begin(); i != code_to_idx.end(); i++) {
        p[idx++] = i->second;
    }

    return result;
}

at::Tensor hnms_cpu(const at::Tensor& dets,
               const at::Tensor& scores,
               float w0,
               float h0,
               float alpha,
               float bx,
               float by
               ) {
    AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
    AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
    AT_ASSERTM(dets.type() == scores.type(), "dets should have the same type as scores");
    if (dets.numel() == 0) {
      return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
    }

    auto codes = hash_rects(dets, w0, h0, alpha, bx, by);

    auto result = get_best_score_each_code(codes, scores);

    return result;
}

at::Tensor hnms(const at::Tensor& dets,
               const at::Tensor& scores,
               float w0,
               float h0,
               float alpha,
               float bx,
               float by
               ) {
  if (dets.type().is_cuda()) {
#ifdef WITH_CUDA
      if (dets.numel() == 0)
          return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
      return hnms_cuda(dets, scores, w0, h0, alpha, bx, by);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  return hnms_cpu(dets, scores, w0, h0, alpha, bx, by);
}

