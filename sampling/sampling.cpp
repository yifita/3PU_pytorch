#include <torch/extension.h>
#include <iostream>

// CUDA forward declarations

at::Tensor furthest_sampling_cuda_forward(
    int b, int n, int m,
    at::Tensor input,
    at::Tensor temp,
    at::Tensor idx);

at::Tensor gather_points_cuda_forward(int b, int c, int n, int npoints,
          at::Tensor points, at::Tensor idx,
          at::Tensor out);

at::Tensor gather_points_cuda_backward(int b, int c, int n, int npoints,
               at::Tensor grad_out, at::Tensor idx, at::Tensor grad_points);

// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor furthest_sampling_forward(
    int b, int n, int m,
    at::Tensor input,
    at::Tensor temp,
    at::Tensor idx) {
  CHECK_INPUT(input);
  CHECK_INPUT(temp);
  return furthest_sampling_cuda_forward(b, n, m, input, temp, idx);
}

at::Tensor gather_points_forward(int b, int c, int n, int npoints,
        at::Tensor points_tensor,
        at::Tensor idx_tensor,
        at::Tensor out_tensor) {
    CHECK_INPUT(points_tensor);
    CHECK_INPUT(idx_tensor);
    return gather_points_cuda_forward(b, c, n, npoints, points_tensor, idx_tensor, out_tensor);
}

at::Tensor gather_points_backward(int b, int c, int n, int npoints,
        at::Tensor grad_out_tensor,
        at::Tensor idx_tensor,
        at::Tensor grad_points_tensor) {
    return gather_points_cuda_backward(b, c, n, npoints, grad_out_tensor, idx_tensor, grad_points_tensor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("furthest_sampling", &furthest_sampling_forward, "furthest point sampling (no gradient)");
  m.def("gather_forward", &gather_points_forward, "gather npoints points along an axis");
  m.def("gather_backward", &gather_points_backward, "gather npoints points along an axis backward");
}