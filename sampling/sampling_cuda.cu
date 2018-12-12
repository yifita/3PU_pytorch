#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

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
    if (m <= 0)
    return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];
    // 3kB
    const int BufferSize=3072;
    // 3kB*3*4 =  36kB
    __shared__ float buf[BufferSize*3];
    for (int i=blockIdx.x; i<b; i+=gridDim.x){
        int old=0;
        // first out of sought m points is point0
        if (threadIdx.x==0) idxs[i*m+0]=old;
        // fill buffer in the shared memory with input for faster read
        for (int j=threadIdx.x;j<min(BufferSize,n)*3;j+=blockDim.x){
          buf[j]=input[i*n*3+j];
        }
        __syncthreads();
    int batch_index = blockIdx.x;
    input += batch_index * n * 3;
    temp += batch_index * n;
    idx += batch_index * m;

    int tid = threadIdx.x;
    const int stride = block_size;

    int old = 0;
    if (threadIdx.x == 0)
    idx[0] = old;

    __syncthreads();
    // each block process a batch
    // iteratively add m points
    for (int j = 1; j < m; j++) {
        int besti = 0;
        float best = -1;
        // last added point
        float x1 = input[old * 3 + 0];
        float y1 = input[old * 3 + 1];
        float z1 = input[old * 3 + 2];
        // all threads loop through all n points
        // so each thread starts from tid in the current block
        // and goes to the next block
        for (int k = tid; k < n; k += stride) {
            float x2, y2, z2;
            x2 = input[k * 3 + 0];
            y2 = input[k * 3 + 1];
            z2 = input[k * 3 + 2];
            float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
            if (mag <= 1e-3) continue;

            float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
            // update the distance to the existing set
            float d2 = min(d, temp[k]);
            temp[k] = d2;
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();
        // max reduction over n points
        // reduce threads logarithmly, maximize pairwise current thread (tid-th thread)
        // with (tid+x)-th thread.
        if (block_size >= 512) {
            if (tid < 256) {
            __update(dists, dists_i, tid, tid + 256);
            }
            __syncthreads();
        }
        if (block_size >= 256) {
            if (tid < 128) {
            __update(dists, dists_i, tid, tid + 128);
            }
            __syncthreads();
        }
        if (block_size >= 128) {
            if (tid < 64) {
            __update(dists, dists_i, tid, tid + 64);
            }
            __syncthreads();
        }
        if (block_size >= 64) {
            if (tid < 32) {
            __update(dists, dists_i, tid, tid + 32);
            }
            __syncthreads();
        }
        if (block_size >= 32) {
            if (tid < 16) {
            __update(dists, dists_i, tid, tid + 16);
            }
            __syncthreads();
        }
        if (block_size >= 16) {
            if (tid < 8) {
            __update(dists, dists_i, tid, tid + 8);
            }
            __syncthreads();
        }
        if (block_size >= 8) {
            if (tid < 4) {
            __update(dists, dists_i, tid, tid + 4);
            }
            __syncthreads();
        }
        if (block_size >= 4) {
            if (tid < 2) {
            __update(dists, dists_i, tid, tid + 2);
            }
            __syncthreads();
        }
        if (block_size >= 2) {
            if (tid < 1) {
            __update(dists, dists_i, tid, tid + 1);
            }
            __syncthreads();
        }

        old = dists_i[0];
        if (tid == 0)
            idx[j] = old;
    }
}


at::Tensor furthest_sampling_cuda_forward(int b, int n, int m,
    at::Tensor input, at::Tensor temp, at::Tensor idx) {

    unsigned int n_threads = opt_n_threads(n);
    unsigned int n_blocks = (n*b + n_threads/2)/n_threads;
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