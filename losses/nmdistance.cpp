#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
std::vector<at::Tensor> NmDistanceKernelLauncher(
    int b,int n, at::Tensor xyz,
    int m, at::Tensor xyz2, at::Tensor result, at::Tensor result_i, at::Tensor result2, at::Tensor result2_i);

std::vector<at::Tensor> NmDistanceGradKernelLauncher(int b,int n, at::Tensor xyz1,
    int m, at::Tensor xyz2,
    at::Tensor grad_dist1, at::Tensor idx1,
    at::Tensor grad_dist2, at::Tensor idx2,
    bool requires_grad_1, bool requires_grad_2,
    at::Tensor grad_xyz1, at::Tensor grad_xyz2);

// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> nmdistance_forward(
    int b,int n, at::Tensor xyz,
    int m, at::Tensor xyz2, at::Tensor result, at::Tensor result_i,
    at::Tensor result2, at::Tensor result2_i) {
  CHECK_INPUT(xyz);
  CHECK_INPUT(xyz2);
  return NmDistanceKernelLauncher(b, n, xyz, m, xyz2, result, result_i, result2, result2_i);
}

std::vector<at::Tensor> nmdistance_backward(int b,int n, at::Tensor xyz1,
    int m, at::Tensor xyz2,
    at::Tensor grad_dist1, at::Tensor idx1,
    at::Tensor grad_dist2, at::Tensor idx2,
    bool requires_grad_1, bool requires_grad_2,
    at::Tensor grad_xyz1, at::Tensor grad_xyz2) {
  return NmDistanceGradKernelLauncher(b, n, xyz1, m, xyz2, grad_dist1, idx1, grad_dist2, idx2,
    requires_grad_1, requires_grad_2, grad_xyz1, grad_xyz2);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nmdistance_forward", &nmdistance_forward, "point-to-point distance forward");
  m.def("nmdistance_backward", &nmdistance_backward, "point-to-point distance backward");
}