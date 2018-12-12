#include <vector>
#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_utils.h"


// closest point from xyz (b, n, 3) to xyz2 (b, m, 3)
template <unsigned int batch>
__global__ void NmDistanceKernel(int b, int n,
		const float * xyz, int m, const float * xyz2, float *result, int *result_i){
	// buffer to cache xyz2 4bytes*3*2^9 = 6kB shared
	__shared__ float buf[batch*3];
	// in total, loop through all b point clouds
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		// loop through the points in current point cloud in xyz2
		for (int k2=0; k2<m; k2+=batch){
			// fill the buffer with chunks of xyz2 values
			int end_k=min(m, k2+batch)-k2;
			for (int j=threadIdx.x; j<end_k*3; j+=blockDim.x){
				buf[j]=xyz2[(i*m+k2)*3+j];
			}
			__syncthreads();
			// loop through all n points in xyz
			for (int j=threadIdx.x; j<n; j+=blockDim.x){
				// current point in xyz
				float x1=xyz[(i*n+j)*3+0];
				float y1=xyz[(i*n+j)*3+1];
				float z1=xyz[(i*n+j)*3+2];
				int best_i=0;
				float best=0;
				// end_k&3 (0~3). end_ka largest multiple of 4
				int end_ka=end_k-(end_k&3);
				// loop through all buffered (xyz2)
				// find shortest distance from a point in xyz -> buffered xyz2
				// process four points in a thread
				if (end_ka==batch){
					for (int k=0;k<batch;k+=4){
						{
							float x2=buf[k*3+0]-x1;
							float y2=buf[k*3+1]-y1;
							float z2=buf[k*3+2]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (k==0 || d<best){
								best=d;
								best_i=k+k2;
							}
						}
						{
							float x2=buf[k*3+3]-x1;
							float y2=buf[k*3+4]-y1;
							float z2=buf[k*3+5]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+1;
							}
						}
						{
							float x2=buf[k*3+6]-x1;
							float y2=buf[k*3+7]-y1;
							float z2=buf[k*3+8]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+2;
							}
						}
						{
							float x2=buf[k*3+9]-x1;
							float y2=buf[k*3+10]-y1;
							float z2=buf[k*3+11]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+3;
							}
						}
					}
				}else{
					for (int k=0;k<end_ka;k+=4){
						{
							float x2=buf[k*3+0]-x1;
							float y2=buf[k*3+1]-y1;
							float z2=buf[k*3+2]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (k==0 || d<best){
								best=d;
								best_i=k+k2;
							}
						}
						{
							float x2=buf[k*3+3]-x1;
							float y2=buf[k*3+4]-y1;
							float z2=buf[k*3+5]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+1;
							}
						}
						{
							float x2=buf[k*3+6]-x1;
							float y2=buf[k*3+7]-y1;
							float z2=buf[k*3+8]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+2;
							}
						}
						{
							float x2=buf[k*3+9]-x1;
							float y2=buf[k*3+10]-y1;
							float z2=buf[k*3+11]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+3;
							}
						}
					}
				}
				for (int k=end_ka;k<end_k;k++){
					float x2=buf[k*3+0]-x1;
					float y2=buf[k*3+1]-y1;
					float z2=buf[k*3+2]-z1;
					float d=x2*x2+y2*y2+z2*z2;
					if (k==0 || d<best){
						best=d;
						best_i=k+k2;
					}
				}
				if (k2==0 || result[(i*n+j)]>best){
					result[(i*n+j)]=best;
					result_i[(i*n+j)]=best_i;
				}
			}
			__syncthreads();
		}
	}
}

std::vector<at::Tensor> NmDistanceKernelLauncher(int b,int n, at::Tensor xyz,
		int m, at::Tensor xyz2, at::Tensor result, at::Tensor result_i, at::Tensor result2, at::Tensor result2_i){
	// bxn
	unsigned int n_threads, n_blocks;
	n_threads = opt_n_threads(n);
	n_blocks = min(32, (n*b + n_threads/2)/n_threads);
	switch (n_threads) {
		case 512:
		NmDistanceKernel<512><<<n_blocks, n_threads>>>(b,n,xyz.data<float>(),
			m,xyz2.data<float>(),result.data<float>(),result_i.data<int32_t>());
		break;
		case 256:
		NmDistanceKernel<256><<<n_blocks, n_threads>>>(b,n,xyz.data<float>(),
			m,xyz2.data<float>(),result.data<float>(),result_i.data<int32_t>());
		break;
		case 128:
		NmDistanceKernel<128><<<n_blocks, n_threads>>>(b,n,xyz.data<float>(),
			m,xyz2.data<float>(),result.data<float>(),result_i.data<int32_t>());
		break;
		case 64:
		NmDistanceKernel<64><<<n_blocks, n_threads>>>(b,n,xyz.data<float>(),
			m,xyz2.data<float>(),result.data<float>(),result_i.data<int32_t>());
		break;
		case 32:
		NmDistanceKernel<32><<<n_blocks, n_threads>>>(b,n,xyz.data<float>(),
			m,xyz2.data<float>(),result.data<float>(),result_i.data<int32_t>());
		break;
		case 16:
		NmDistanceKernel<16><<<n_blocks, n_threads>>>(b,n,xyz.data<float>(),
			m,xyz2.data<float>(),result.data<float>(),result_i.data<int32_t>());
		break;
		case 8:
		NmDistanceKernel<8><<<n_blocks, n_threads>>>(b,n,xyz.data<float>(),
			m,xyz2.data<float>(),result.data<float>(),result_i.data<int32_t>());
		break;
		case 4:
		NmDistanceKernel<4><<<n_blocks, n_threads>>>(b,n,xyz.data<float>(),
			m,xyz2.data<float>(),result.data<float>(),result_i.data<int32_t>());
		break;
		case 2:
		NmDistanceKernel<2><<<n_blocks, n_threads>>>(b,n,xyz.data<float>(),
			m,xyz2.data<float>(),result.data<float>(),result_i.data<int32_t>());
		break;
		case 1:
		NmDistanceKernel<1><<<n_blocks, n_threads>>>(b,n,xyz.data<float>(),
			m,xyz2.data<float>(),result.data<float>(),result_i.data<int32_t>());
		break;
		default:
		NmDistanceKernel<512><<<n_blocks, n_threads>>>(b,n,xyz.data<float>(),
			m,xyz2.data<float>(),result.data<float>(),result_i.data<int32_t>());
		break;
	}

	// bxm
	n_threads = opt_n_threads(m);
	n_blocks = min(32, (m*b + n_threads/2)/n_threads);
	switch (n_threads){
		case 512:
		NmDistanceKernel<512><<<n_blocks, n_threads>>>(b,m,xyz2.data<float>(),
			n,xyz.data<float>(),result2.data<float>(),result2_i.data<int32_t>());
		break;
		case 256:
		NmDistanceKernel<256><<<n_blocks, n_threads>>>(b,m,xyz2.data<float>(),
			n,xyz.data<float>(),result2.data<float>(),result2_i.data<int32_t>());
		break;
		case 128:
		NmDistanceKernel<128><<<n_blocks, n_threads>>>(b,m,xyz2.data<float>(),
			n,xyz.data<float>(),result2.data<float>(),result2_i.data<int32_t>());
		break;
		case 64:
		NmDistanceKernel<64><<<n_blocks, n_threads>>>(b,m,xyz2.data<float>(),
			n,xyz.data<float>(),result2.data<float>(),result2_i.data<int32_t>());
		break;
		case 32:
		NmDistanceKernel<32><<<n_blocks, n_threads>>>(b,m,xyz2.data<float>(),
			n,xyz.data<float>(),result2.data<float>(),result2_i.data<int32_t>());
		break;
		case 16:
		NmDistanceKernel<16><<<n_blocks, n_threads>>>(b,m,xyz2.data<float>(),
			n,xyz.data<float>(),result2.data<float>(),result2_i.data<int32_t>());
		break;
		case 8:
		NmDistanceKernel<8><<<n_blocks, n_threads>>>(b,m,xyz2.data<float>(),
			n,xyz.data<float>(),result2.data<float>(),result2_i.data<int32_t>());
		break;
		case 4:
		NmDistanceKernel<4><<<n_blocks, n_threads>>>(b,m,xyz2.data<float>(),
			n,xyz.data<float>(),result2.data<float>(),result2_i.data<int32_t>());
		break;
		case 2:
		NmDistanceKernel<2><<<n_blocks, n_threads>>>(b,m,xyz2.data<float>(),
			n,xyz.data<float>(),result2.data<float>(),result2_i.data<int32_t>());
		break;
		case 1:
		NmDistanceKernel<1><<<n_blocks, n_threads>>>(b,m,xyz2.data<float>(),
			n,xyz.data<float>(),result2.data<float>(),result2_i.data<int32_t>());
		break;
		default:
		NmDistanceKernel<512><<<n_blocks, n_threads>>>(b,m,xyz2.data<float>(),
			n,xyz.data<float>(),result2.data<float>(),result2_i.data<int32_t>());
		break;
	}

	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd updateOutput: %s\n", cudaGetErrorString(err));
	    exit(-1);
	  }
	return {result, result_i, result2, result2_i};
}

// template std::vector<at::Tensor> NmDistanceKernelLauncher<double>(int b,int n, at::Tensor xyz,
// 		int m, at::Tensor xyz2, at::Tensor result, at::Tensor result_i, at::Tensor result2, at::Tensor result2_i);
// template std::vector<at::Tensor> NmDistanceKernelLauncher<float>(int b,int n, at::Tensor xyz,
// 		int m, at::Tensor xyz2, at::Tensor result, at::Tensor result_i, at::Tensor result2, at::Tensor result2_i);

__global__ void NmDistanceGradKernel(int b, int n, const float * xyz1,
		int m, const float * xyz2, const float * grad_dist1, const int * idx1, float * grad_xyz1, float * grad_xyz2){
	for (int i=blockIdx.x; i<b; i+=gridDim.x){
		for (int j=threadIdx.x; j<n; j+=blockDim.x){
			// j-th point in xyz1 is mapped to j2-th point in xyz2
			float x1=xyz1[(i*n+j)*3+0];
			float y1=xyz1[(i*n+j)*3+1];
			float z1=xyz1[(i*n+j)*3+2];
			int j2=idx1[i*n+j];
			float x2=xyz2[(i*m+j2)*3+0];
			float y2=xyz2[(i*m+j2)*3+1];
			float z2=xyz2[(i*m+j2)*3+2];
			float g=grad_dist1[i*n+j]*2;
			atomicAdd(grad_xyz1+(i*n+j)*3+0, g*(x1-x2));
			atomicAdd(grad_xyz1+(i*n+j)*3+1, g*(y1-y2));
			atomicAdd(grad_xyz1+(i*n+j)*3+2, g*(z1-z2));
			atomicAdd(grad_xyz2+(i*m+j2)*3+0, -(g*(x1-x2)));
			atomicAdd(grad_xyz2+(i*m+j2)*3+1, -(g*(y1-y2)));
			atomicAdd(grad_xyz2+(i*m+j2)*3+2, -(g*(z1-z2)));
		}
	}
}

__global__ void NmDistanceGrad1Kernel(int b, int n, const float * xyz1,
		int m, const float * xyz2, const float * grad_dist1, const int * idx1, float * grad_xyz1){
	for (int i=blockIdx.x; i<b; i+=gridDim.x){
		for (int j=threadIdx.x; j<n; j+=blockDim.x){
			// j-th point in xyz1 is mapped to j2-th point in xyz2
			float x1=xyz1[(i*n+j)*3+0];
			float y1=xyz1[(i*n+j)*3+1];
			float z1=xyz1[(i*n+j)*3+2];
			int j2=idx1[i*n+j];
			float x2=xyz2[(i*m+j2)*3+0];
			float y2=xyz2[(i*m+j2)*3+1];
			float z2=xyz2[(i*m+j2)*3+2];
			float g=grad_dist1[i*n+j]*2;
			atomicAdd(grad_xyz1+(i*n+j)*3+0, g*(x1-x2));
			atomicAdd(grad_xyz1+(i*n+j)*3+1, g*(y1-y2));
			atomicAdd(grad_xyz1+(i*n+j)*3+2, g*(z1-z2));
		}
	}
}

__global__ void NmDistanceGrad2Kernel(int b, int n, const float * xyz1,
		int m, const float * xyz2, const float * grad_dist1, const int * idx1, float * grad_xyz2){
	for (int i=blockIdx.x; i<b; i+=gridDim.x){
		for (int j=threadIdx.x; j<n; j+=blockDim.x){
			// j-th point in xyz1 is mapped to j2-th point in xyz2
			float x1=xyz1[(i*n+j)*3+0];
			float y1=xyz1[(i*n+j)*3+1];
			float z1=xyz1[(i*n+j)*3+2];
			int j2=idx1[i*n+j];
			float x2=xyz2[(i*m+j2)*3+0];
			float y2=xyz2[(i*m+j2)*3+1];
			float z2=xyz2[(i*m+j2)*3+2];
			float g=grad_dist1[i*n+j]*2;
			atomicAdd(grad_xyz2+(i*m+j2)*3+0, -(g*(x1-x2)));
			atomicAdd(grad_xyz2+(i*m+j2)*3+1, -(g*(y1-y2)));
			atomicAdd(grad_xyz2+(i*m+j2)*3+2, -(g*(z1-z2)));
		}
	}
}

std::vector<at::Tensor> NmDistanceGradKernelLauncher(int b,int n, at::Tensor xyz1,
		int m, at::Tensor xyz2,
		at::Tensor grad_dist1, at::Tensor idx1,
		at::Tensor grad_dist2, at::Tensor idx2,
		bool requires_grad_1, bool requires_grad_2,
		at::Tensor grad_xyz1, at::Tensor grad_xyz2) {
	std::vector<at::Tensor> v;
	unsigned int n_threads, n_blocks;
	if (!requires_grad_2)
	{
		n_threads = opt_n_threads(n);
		n_blocks = min(32, (n*b + n_threads/2)/n_threads);
		NmDistanceGrad1Kernel<<<n_blocks, n_threads>>>(b,n,xyz1.data<float>(),
			m,xyz2.data<float>(),grad_dist1.data<float>(),idx1.data<int32_t>(),grad_xyz1.data<float>());
		n_threads = opt_n_threads(m);
		n_blocks = min(32, (m*b + n_threads/2)/n_threads);
		NmDistanceGrad2Kernel<<<n_blocks, n_threads>>>(b,m,xyz2.data<float>(),
			n,xyz1.data<float>(),grad_dist2.data<float>(),idx2.data<int32_t>(),grad_xyz1.data<float>());
		v = {grad_xyz1};
	}
	if (!requires_grad_1)
	{
		n_threads = opt_n_threads(m);
		n_blocks = min(32, (m*b + n_threads/2)/n_threads);
		NmDistanceGrad2Kernel<<<n_blocks, n_threads>>>(b,n,xyz1.data<float>(),
			m,xyz2.data<float>(),grad_dist1.data<float>(),idx2.data<int32_t>(),grad_xyz1.data<float>());
		n_threads = opt_n_threads(n);
		n_blocks = min(32, (n*b + n_threads/2)/n_threads);
		NmDistanceGrad1Kernel<<<n_blocks, n_threads>>>(b,m,xyz2.data<float>(),
			n,xyz1.data<float>(),grad_dist2.data<float>(),idx1.data<int32_t>(),grad_xyz2.data<float>());
		v = {grad_xyz1};
	}
	if (requires_grad_1 && requires_grad_2)
	{
		n_threads = opt_n_threads(n);
		n_blocks = min(32, (n*b + n_threads/2)/n_threads);
		NmDistanceGradKernel<<<n_blocks, n_threads>>>(b,n,xyz1.data<float>(),
			m,xyz2.data<float>(),grad_dist1.data<float>(),idx1.data<int32_t>(),grad_xyz1.data<float>(), grad_xyz2.data<float>());

		n_threads = opt_n_threads(m);
		n_blocks = min(32, (m*b + n_threads/2)/n_threads);
		NmDistanceGradKernel<<<n_blocks, n_threads>>>(b,m,xyz2.data<float>(),
			n,xyz1.data<float>(),grad_dist2.data<float>(),idx2.data<int32_t>(),grad_xyz2.data<float>(),grad_xyz1.data<float>());
		v = {grad_xyz1, grad_xyz2};
	}

	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd get grad: %s\n", cudaGetErrorString(err));
	    exit(-1);
	  }
	return v;
}

