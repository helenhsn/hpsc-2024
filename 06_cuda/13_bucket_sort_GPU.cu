#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

void print_vec(int *vec, int size);

__global__ void fill_bucket(int maxTid, int *bucket, int *keys)
{

  // initializing shared mem within block
  extern __shared__ int temp[];
  temp[threadIdx.x] = 0;
  __syncthreads();
  // printf("ok, blockIdx=%i, tid = %i && threadIDx.X = %i\n", blockIdx.x, tid, threadIdx.x);

  // filling the bucket -> each thread can process several elements in the key array

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < maxTid)
  {
    atomicAdd(temp +  keys[tid], 1);
  }
  __syncthreads();


  // the number of threads per block is equal to the range of the bucket
  atomicAdd(bucket + threadIdx.x, temp[threadIdx.x]); 
}


// prefix sum algorithm to fill the offset buffer (sum of bucket values)
__global__ void fill_offset(int range, int *bucket, int *offset, int *key)
{
  extern __shared__ int temp[];
  int loc_tid = threadIdx.x;
  temp[loc_tid] = loc_tid>0 ? bucket[loc_tid - 1] : 0;
  __syncthreads();

  int outbuff = 0;

  for (int k=1; k<range; k<<=1)
  {
    // swap in and out buffers
    outbuff = 1 - outbuff;
    int outTID = (1 - outbuff)*range+loc_tid;
    int res = (loc_tid >= k) ? 1 : 0;

    temp[outbuff*range+loc_tid] = temp[outTID] + res * temp[outTID-k];
    __syncthreads();
  }

  offset[loc_tid] = temp[outbuff*range+loc_tid];
}

__global__ void fill_key(int maxTid, int *bucket, int *offset, int *key)
{

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < maxTid)
  {
    int nb_occurences = bucket[tid];
    int off = offset[tid];

    for (int i=0; i<nb_occurences; i++)
    {
      key[i+off] = tid;
    }
  }
}


void bucket_sort_CPU(int *key, int *output, int n, int range)
{
  std::vector<int> bucket(range);

  auto start = std::chrono::high_resolution_clock::now();

  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }

  std::chrono::duration<double> task1 =  std::chrono::high_resolution_clock::now() - start;
  printf("\nFILLING BUCKET TASK executed on CPU in = %f\n", task1.count());

  start = std::chrono::high_resolution_clock::now();

  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      output[j++] = i;
    }
  }

  std::chrono::duration<double> task2 = std::chrono::high_resolution_clock::now() - start;
  printf("\nSORTING KEYS WITH BUCKET TASK executed on CPU in = %f", task2.count());


}

void print_vec(int *vec, int size)
{
  for (int i=0; i<size; i++) {
    printf("%d ",vec[i]);
  }
  printf("\n");
}





int main() {
  int n = 10000000;

  // BEWARE !!! 
  // range can only go up to 128 bc of shared memory capacity in fill_offset prefix scan function...
  int range = 1024;

  // initializing all the buffers
  int *key;
  int *bucket; 
  int *offset; 

  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&offset, range*sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
  }
  printf("Key array filled! \n");


  // CPU

  std::vector<int> outputCPU(n);


  auto start = std::chrono::high_resolution_clock::now();

  bucket_sort_CPU(key, outputCPU.data(), n, range);

  // printf("\nFILLING BUCKET TASK executed on GPU in = %f", task1.count());
  std::chrono::duration<double>  CPU = (std::chrono::high_resolution_clock::now() - start);


  start = std::chrono::high_resolution_clock::now();

  
  fill_bucket<<<(n+range-1)/range, range, range*sizeof(int)>>>(n, bucket, key);
  cudaDeviceSynchronize();
  
  std::chrono::duration<double> task1 = std::chrono::high_resolution_clock::now() - start;
  printf("\nFILLING BUCKET TASK executed on GPU in = %f", task1.count());
  
  auto start2 = std::chrono::high_resolution_clock::now();
  
  fill_offset<<<1, range, 2*range*sizeof(int)>>>(range, bucket, offset, key);
  cudaDeviceSynchronize();


  fill_key<<<(range+31)/32, 32>>>(range, bucket, offset, key);
  cudaDeviceSynchronize();

  std::chrono::duration<double> task2 = std::chrono::high_resolution_clock::now() - start2;
  printf("\nSORTING KEYS WITH BUCKET TASK executed on GPU in = %f", task2.count());
  // printf("\noutput gpu = \n bucket SORTED = ");
  // print_vec(bucket, range);
  // printf("\n");
  // printf("\noutput gpu = \n offset SORTED = ");
  // print_vec(offset, range);
  // printf("\n");
  // printf("\noutput gpu = \n KEY SORTED = ");
  // print_vec(key, n);
  // printf("\n");

  std::chrono::duration<double>  GPU = (std::chrono::high_resolution_clock::now() - start);

  printf("\n\n >> Time taken for array of size = %i with range = %i (SECONDS)  \n >> CPU: %f  \n >> GPU: %f (speedup by a factor x%f)", n, range,  CPU.count(), GPU.count(), CPU.count()/GPU.count());

}
