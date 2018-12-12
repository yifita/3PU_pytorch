import faiss


GPU_RES = faiss.StandardGpuResources()
GPU_RES.setTempMemoryFraction(0.1)
