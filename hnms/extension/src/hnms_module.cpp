#include "hnms.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hnms", &hnms, "HNMS");
}

