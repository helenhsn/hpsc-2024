#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

__global__ void fill_bucket(int maxTid, int Narray, int *bucket, int *keys)
{
  // initializing shared mem within block
  extern __shared__ int bucket[];
  __syncthreads();
  temp[threadIdx.x] = 0;
  __syncthreads();

  // filling the bucket -> each thread can process several elements in the key array
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < maxTid)
  {
    int el = keys[tid];
    atomicAdd(&temp[el], 1);
  }

  atomicAdd(&bucket[threadIdx.x], &temp[threadIdx.x]); 
}


std::vector<int> bucket_sort_CPU(std::vector<int> key, int n, int range)
{
  std::vector<int> bucket(range);
  std::vector<int> output(n);

  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<key.size(); i++) {
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      output[j++] = i;
    }
  }

  return output;
}

void print_vec(std::vector<int> vec)
{
  for (int i=0; i<vec.size(); i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}


int main() {
  int n = 50;
  int range = 5;

  // initializing all the buffers
  std::vector<int> key(n);
  std::vector<int> bucket(range); 

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;

  }
  printf("\n");


  // CPU
  auto start = std::chrono::steady_clock::now();

  std::vector<int> outputCPU = bucket_sort_CPU(key, n, range);

  auto end = std::chrono::steady_clock::now();

  float CPU = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  // CPU verification
  print_vec(outputCPU);

  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));

  start = std::chrono::steady_clock::now();
  
  fill_bucket<<<n/range, range, range*sizeof(int)>>>();
  cudaDeviceSynchronize();

  print_vec(bucket);
  // fill_keys<<<>>>();
  // cudaDeviceSynchronize();

  float GPU = (std::chrono::steady_clock::now() - start).count();

  printf("Time taken by CPU: %f VS GPU: %f",  CPU, GPU);

}
