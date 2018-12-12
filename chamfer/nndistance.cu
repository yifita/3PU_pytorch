#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_utils.h"

// closest point from xyz (b, n, 3) to xyz2 (b, m, 3)
__global__ void NmDistanceKernel(int b, int n,
		const float * xyz, int m, const float * xyz2, float *result, int *result_i){
	// block size 512
	const int batch=512;
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
			for (int j=threadIdx.x+blockIdx.y*blockDim.x; j<n; j+=blockDim.x*gridDim.y){
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


	NmDistanceKernel<<<dim3(32,16,1),512>>>(b,n,xyz.data<float>(),m,xyz2.data<float>(),result.data<float>,result_i.data<int32_t>());
	NmDistanceKernel<<<dim3(32,16,1),512>>>(b,m,xyz2.data<float>(),n,xyz,result2.data<float>(),result2_i.data<int>());

	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd updateOutput: %s\n", cudaGetErrorString(err));
	    //THError("aborting");
	    exit(-1);
	  }
	return {result, result_i, result2, result2_i};
}

__global__ void NmDistanceGradKernel(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,float * grad_xyz1,float * grad_xyz2){
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
			float x1=xyz1[(i*n+j)*3+0];
			float y1=xyz1[(i*n+j)*3+1];
			float z1=xyz1[(i*n+j)*3+2];
			int j2=idx1[i*n+j];
			float x2=xyz2[(i*m+j2)*3+0];
			float y2=xyz2[(i*m+j2)*3+1];
			float z2=xyz2[(i*m+j2)*3+2];
			float g=grad_dist1[i*n+j]*2;
			atomicAdd(&(grad_xyz1[(i*n+j)*3+0]),g*(x1-x2));
			atomicAdd(&(grad_xyz1[(i*n+j)*3+1]),g*(y1-y2));
			atomicAdd(&(grad_xyz1[(i*n+j)*3+2]),g*(z1-z2));
			atomicAdd(&(grad_xyz2[(i*m+j2)*3+0]),-(g*(x1-x2)));
			atomicAdd(&(grad_xyz2[(i*m+j2)*3+1]),-(g*(y1-y2)));
			atomicAdd(&(grad_xyz2[(i*m+j2)*3+2]),-(g*(z1-z2)));
		}
	}
}
int NmDistanceGradKernelLauncher(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,const float * grad_dist2,const int * idx2,float * grad_xyz1,float * grad_xyz2, cudaStream_t stream){
	cudaMemset(grad_xyz1,0,b*n*3*4);
	cudaMemset(grad_xyz2,0,b*m*3*4);
	NmDistanceGradKernel<<<dim3(1,16,1),256>>>(b,n,xyz1,m,xyz2,grad_dist1,idx1,grad_xyz1,grad_xyz2);
	NmDistanceGradKernel<<<dim3(1,16,1),256>>>(b,m,xyz2,n,xyz1,grad_dist2,idx2,grad_xyz2,grad_xyz1);

	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd get grad: %s\n", cudaGetErrorString(err));
	    //THError("aborting");
	    return 0;
	  }
	  return 1;

}

