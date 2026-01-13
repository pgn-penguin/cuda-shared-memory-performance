#include <torch/extension.h>
#include <vector>

// 宣告共享記憶體CUDA前向傳播函數
std::vector<torch::Tensor> custom_conv_shared_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int groups);

// 包裝函數實現
std::vector<torch::Tensor> custom_conv_shared_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int groups) {
    return custom_conv_shared_cuda_forward(input, weight, stride, padding, groups);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &custom_conv_shared_forward, 
          "Custom convolution forward with shared memory (CUDA)",
          py::arg("input"),
          py::arg("weight"), 
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("groups") = 1);
}