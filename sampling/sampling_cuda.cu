#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "cuda_utils.h"


__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
             int idx1, int idx2) {

    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

// input: points(b, c, n) idx(b, m)
// output: out(b, c, m)
template <typename scalar_t>
__global__ void gather_points_forward_kernel(int b, int c, int n, int m,
                     const scalar_t *__restrict__ points,
                     const int *__restrict__ idx,
                     scalar_t *__restrict__ out) {
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        for (int l = blockIdx.y; l < c; l += gridDim.y) {
            for (int j = threadIdx.x; j < m; j += blockDim.x) {
            int a = idx[i * m + j];
            out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
            }
        }
    }
}

at::Tensor gather_points_cuda_forward(int b, int c, int n, int npoints,
                  at::Tensor points, at::Tensor idx,
                  at::Tensor out) {

    cudaError_t err;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.type(), "gather_points_cuda_forward", ([&] {
            gather_points_forward_kernel<scalar_t><<<dim3(b, c, 1), opt_n_threads(npoints)>>>(
            b, c, n, npoints,
            points.data<scalar_t>(),
            idx.data<int32_t>(),
            out.data<scalar_t>());
        }));

    err = cudaGetLastError();
    if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
    }
    return out;
}

// input: grad_out(b, c, m) idx(b, m)
// output: grad_points(b, c, n)
template <typename scalar_t>
__global__ void gather_points_backward_kernel(int b, int c, int n, int m,
                      scalar_t *__restrict__ grad_out,
                      const int *__restrict__ idx,
                      scalar_t *__restrict__ grad_points) {
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        for (int l = blockIdx.y; l < c; l += gridDim.y) {
            for (int j = threadIdx.x; j < m; j += blockDim.x) {
            int a = idx[i * m + j];
            atomicAdd(grad_points + (i * c + l) * n + a,
                  grad_out[(i * c + l) * m + j]);
            }
        }
    }
}


at::Tensor gather_points_cuda_backward(int b, int c, int n, int npoints,
                       at::Tensor grad_out, at::Tensor idx, at::Tensor grad_points) {
    cudaError_t err;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_out.type(), "gather_points_cuda_backward", ([&] {
        gather_points_backward_kernel<scalar_t><<<dim3(b, c, 1), opt_n_threads(npoints)>>>(
            b, c, n, npoints,
            grad_out.data<scalar_t>(),
            idx.data<int32_t>(),
            grad_points.data<scalar_t>());
      }));

    err = cudaGetLastError();
    if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
    }
    return grad_points;
}


template <unsigned int block_size>
__global__ void furthest_point_sampling_forward_kernel(int b, int n, int m,
    const float * __restrict__ input, float * __restrict__ temp, int * __restrict__ idx) {
    // temp: (nxb) the closest distance from each of the n points to the existing set
    if (m <= 0) return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];
    const unsigned int buffer_size = block_size;
    __shared__ float buf[block_size*3];
    for (int i=blockIdx.x; i<b; i+=gridDim.x){
        int old=0;
        // first out of sought m points is point0
        if (threadIdx.x==0) idx[i*m+0]=old;
        // fill buffer in the shared memory with input *once* for faster read
        for (int j=threadIdx.x;j<min(buffer_size,n)*3;j+=blockDim.x){
          buf[j]=input[i*n*3+j];
        }
        __syncthreads();
        // iteratively add m points
        for (int j=1; j<m; j++){
              int besti=0;
              float best=-1;
              // position of the last point
              float x1=input[i*n*3+old*3+0];
              float y1=input[i*n*3+old*3+1];
              float z1=input[i*n*3+old*3+2];
              // Neither do i understand this loop
              for (int k=threadIdx.x;k<n;k+=blockDim.x){
                float td=temp[blockIdx.x*n+k];
                float x2,y2,z2;
                // if buffer not filled, set new point an input point
                if (k<buffer_size){
                  x2=buf[k*3+0];
                  y2=buf[k*3+1];
                  z2=buf[k*3+2];
                }else{
                  x2=input[i*n*3+k*3+0];
                  y2=input[i*n*3+k*3+1];
                  z2=input[i*n*3+k*3+2];
                }
                float d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
                float d2=min(d,td);
                if (d2!=td)
                  temp[blockIdx.x*n+k]=d2;
                if (d2>best){
                  best=d2;
                  besti=k;
                }
              }
              dists[threadIdx.x]=best;
              dists_i[threadIdx.x]=besti;
              // u from 0~log2(block_size)
              for (int u=0;(1<<u)<blockDim.x;u++){
                __syncthreads();
                // maximize pairwise between the current thread and
                // the sibling thread in the binary-tree
                if (threadIdx.x<(blockDim.x>>(u+1))){
                  int i1=(threadIdx.x*2)<<u;
                  int i2=(threadIdx.x*2+1)<<u;
                  if (dists[i1]<dists[i2]){
                    dists[i1]=dists[i2];
                    dists_i[i1]=dists_i[i2];
                  }
                }
              }
              __syncthreads();
              old=dists_i[0];
              if (threadIdx.x==0)
                idx[i*m+j]=old;
            }
          }
        }

at::Tensor furthest_sampling_cuda_forward(int b, int n, int m,
    at::Tensor input, at::Tensor temp, at::Tensor idx) {

    unsigned int n_threads = opt_n_threads(n);
    unsigned int n_blocks = min(32, (n*b + n_threads/2)/n_threads);
    switch (n_threads) {
      case 512:
      furthest_point_sampling_forward_kernel<512><<<n_blocks, n_threads>>>(
          b, n, m, input.data<float>(),
          temp.data<float>(),
          idx.data<int32_t>());
      break;
      case 256:
      furthest_point_sampling_forward_kernel<256><<<n_blocks, n_threads>>>(
          b, n, m,
          input.data<float>(),
          temp.data<float>(),
          idx.data<int32_t>());
      break;
      case 128:
      furthest_point_sampling_forward_kernel<128><<<n_blocks, n_threads>>>(
          b, n, m,
          input.data<float>(),
          temp.data<float>(),
          idx.data<int32_t>());
      break;
      case 64:
      furthest_point_sampling_forward_kernel<64><<<n_blocks, n_threads>>>(
          b, n, m,
          input.data<float>(),
          temp.data<float>(),
          idx.data<int32_t>());
      break;
      case 32:
      furthest_point_sampling_forward_kernel<32><<<n_blocks, n_threads>>>(
          b, n, m,
          input.data<float>(),
          temp.data<float>(),
          idx.data<int32_t>());
      break;
      case 16:
      furthest_point_sampling_forward_kernel<16><<<n_blocks, n_threads>>>(
          b, n, m,
          input.data<float>(),
          temp.data<float>(),
          idx.data<int32_t>());
      break;
      case 8:
      furthest_point_sampling_forward_kernel<8><<<n_blocks, n_threads>>>(
          b, n, m,
          input.data<float>(),
          temp.data<float>(),
          idx.data<int32_t>());
      break;
      case 4:
      furthest_point_sampling_forward_kernel<4><<<n_blocks, n_threads>>>(
          b, n, m,
          input.data<float>(),
          temp.data<float>(),
          idx.data<int32_t>());
      break;
      case 2:
      furthest_point_sampling_forward_kernel<2><<<n_blocks, n_threads>>>(
          b, n, m,
          input.data<float>(),
          temp.data<float>(),
          idx.data<int32_t>());
      break;
      case 1:
      furthest_point_sampling_forward_kernel<1><<<n_blocks, n_threads>>>(
          b, n, m,
          input.data<float>(),
          temp.data<float>(),
          idx.data<int32_t>());
      break;
      default:
      furthest_point_sampling_forward_kernel<512><<<n_blocks, n_threads>>>(
          b, n, m,
          input.data<float>(),
          temp.data<float>(),
          idx.data<int32_t>());
      }

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
    }
    return idx;
}

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
template <typename scalar_t>
__global__ void query_ball_point_kernel(int b, int n, int m, float radius,
                                        int nsample,
                                        const scalar_t *__restrict__ new_xyz,
                                        const scalar_t *__restrict__ xyz,
                                        int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    scalar_t new_x = new_xyz[j * 3 + 0];
    scalar_t new_y = new_xyz[j * 3 + 1];
    scalar_t new_z = new_xyz[j * 3 + 2];
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      scalar_t x = xyz[k * 3 + 0];
      scalar_t y = xyz[k * 3 + 1];
      scalar_t z = xyz[k * 3 + 2];
      scalar_t d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
          }
        }
        idx[j * nsample + cnt] = k;
        ++cnt;
      }
    }
  }
}

at::Tensor ball_query_cuda_forward(int b, int n, int m, float radius,
                                     int nsample, at::Tensor query,
                                     at::Tensor xyz, at::Tensor idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(xyz.type(), "query_ball_point_kernel", ([&]() {
    query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(b, n, m, radius, nsample, 
      query.data<scalar_t>(), xyz.data<scalar_t>(), idx.data<int32_t>());
		  }));
  CUDA_CHECK_ERRORS();
  return idx;
}