#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>



/**
 * 
 * CUDA VERSION OF THE BUCKET SORT WITH CUDA STREAMS
 * Not worth it because gpu is already filled when running the fill bucket kernel...
 * 
*/

#define NUM_STREAMS 10

void print_vec(int *vec, int size);

__global__ void fill_bucket(int maxTid, int *bucket, int *keys, int stream_offset)
{

  // initializing shared mem within block
  extern __shared__ int temp[];
  temp[threadIdx.x] = 0;
  __syncthreads();
  // printf("loctid=%i", threadIdx.x);
  // printf("ok, blockIdx=%i, tid = %i && threadIDx.X = %i\n", blockIdx.x, tid, threadIdx.x);

  // filling the bucket -> each thread can process several elements in the key array

  int tid = threadIdx.x + blockIdx.x*blockDim.x + stream_offset;
  // printf("\ntid=%i", tid);

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
  for (int k=1; k<range; k=k<<1)
  {
    // swap in and out buffers
    outbuff = 1 - outbuff;
    int outTID = (1 - outbuff)*range+loc_tid;
    int res = (loc_tid >= k) ? temp[outTID-k] : 0;

    temp[outbuff*range+loc_tid] = temp[outTID] + res;
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
  int n = 268435456;

  // the bigger the range, the more the faster the CUDA version compared to the sequential version.
  // but cannot increase it too much because of hardware limitations.
  int range = 1024;

  // initializing all the buffers
  int *h_key = (int *) calloc(n, sizeof(int));
  int *d_key;
  int *d_bucket;
  int *h_bucket = (int *) calloc(range, sizeof(int)); 
  int *d_offset; 
  int *h_offset = (int *) calloc(range, sizeof(int)); 




  for (int i=0; i<n; i++) {
    h_key[i] = rand() % range;
  }
  printf("Key array filled! \n");

  cudaMalloc(&d_key, n*sizeof(int));
  cudaMemcpy(d_key, h_key, sizeof(int)*n, cudaMemcpyHostToDevice);

  cudaMallocManaged(&d_bucket, range*sizeof(int));
  cudaMallocManaged(&d_offset, range*sizeof(int));

  // CPU

  std::vector<int> outputCPU(n);
  auto start = std::chrono::high_resolution_clock::now();
  bucket_sort_CPU(h_key, outputCPU.data(), n, range);
  std::chrono::duration<double>  CPU = (std::chrono::high_resolution_clock::now() - start);


  // GPU



  /*
  STEP 1: build the bucket array. this step is basically a histogram initialization.
  I decided to use cuda streams because the cuda algorithm with only shared atomics
  is still not very efficient compared to the CPU/sequential version (~2 to 3 times faster
  on my own GPU = RTX 3050)

  this step causes the n & range parameters values to be quite limited 
  as the former needs to be divisible by the latter in order to cover the whole range of thread IDs.
  */
  start = std::chrono::high_resolution_clock::now();
 
  cudaStream_t streams[NUM_STREAMS];
 
  int streamSize = n / NUM_STREAMS;
  int streamSizeBytes = streamSize*sizeof(int);
  int blockSize = range;
  int gridSize = (streamSize)/blockSize;
  

  for (int i=0; i<NUM_STREAMS; ++i)
  {
    cudaStreamCreate(&streams[i]);
  }

  for (int i=0; i<NUM_STREAMS; ++i)
  {
    int offsetStream = i*streamSize;
    cudaMemcpyAsync(
      d_key + offsetStream, 
      h_key + offsetStream, 
      streamSizeBytes, 
      cudaMemcpyHostToDevice, 
      streams[i]);
  }
  for (int i=0; i<NUM_STREAMS; ++i)
  {
    int offsetStream = i*streamSize;
    fill_bucket<<<gridSize, blockSize, blockSize*sizeof(int), streams[i]>>>(n, d_bucket, d_key, offsetStream);
  }
  for (int i=0; i<NUM_STREAMS; ++i)
  {
    int offsetStream = i*streamSize;
    cudaMemcpyAsync(
      h_key + offsetStream, 
      d_key + offsetStream, 
      streamSizeBytes, 
      cudaMemcpyDeviceToHost, 
      streams[i]);
  }
  cudaDeviceSynchronize();

  std::chrono::duration<double> task1 = std::chrono::high_resolution_clock::now() - start;
  printf("\nFILLING BUCKET TASK executed on GPU in = %f", task1.count());
  


  /*
  STEP 2: sort the key array according to the bucket array. an offset array
  is built in the first kernel using prefix scan sum pattern.
  */
  auto start2 = std::chrono::high_resolution_clock::now();
  
  fill_offset<<<1, range, 2*range*sizeof(int)>>>(range, d_bucket, d_offset, d_key);
  cudaDeviceSynchronize();


  fill_key<<<(range+31)/32, 32>>>(range, d_bucket, d_offset, d_key);
  cudaDeviceSynchronize();

  std::chrono::duration<double> task2 = std::chrono::high_resolution_clock::now() - start2;
  printf("\nSORTING KEYS WITH BUCKET TASK executed on GPU in = %f", task2.count());

  std::chrono::duration<double>  GPU = (std::chrono::high_resolution_clock::now() - start);
  printf("\n\n >> Time taken for array of size = %i with range = %i (SECONDS)  \n >> CPU: %f  \n >> GPU: %f (speedup by a factor x%f)", n, range,  CPU.count(), GPU.count(), CPU.count()/GPU.count());



  /*
  PRINTING TO CONFIRM PROPER SORTING OF THE KEY ARRAY
  */
  cudaMemcpy(h_key, d_key, sizeof(int)*n, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_bucket, d_bucket, sizeof(int)*range, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_offset, d_offset, sizeof(int)*range, cudaMemcpyDeviceToHost);

  // printf("\noutput gpu = \n bucket SORTED = ");
  // print_vec(h_bucket, range);
  // printf("\n");
  // printf("\noutput gpu = \n offset SORTED = ");
  // print_vec(h_offset, range);
  // printf("\n");
  // printf("\noutput gpu = \n KEY SORTED = \n");
  // print_vec(h_key, n);
  // printf("\n");



  cudaFree(d_offset);
  cudaFree(d_bucket);
  cudaFree(d_key);
  free(h_bucket);
  free(h_offset);
  free(h_key);
}
