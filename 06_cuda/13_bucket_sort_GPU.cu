#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

void print_vec(int *vec, int size);

__global__ void fill_bucket(int maxTid, int *bucket, int *keys)
{

  // initializing shared mem within block
  extern __shared__ int temp[];
  __syncthreads();
  temp[threadIdx.x] = 0;
  __syncthreads();
  // printf("ok, blockIdx=%i, tid = %i && threadIDx.X = %i\n", blockIdx.x, tid, threadIdx.x);

  // filling the bucket -> each thread can process several elements in the key array

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < maxTid)
  {
    int el = keys[tid];
    int old = atomicAdd(&temp[el], 1);
  }
  __syncthreads();


  // the number of threads per block is equal to the range of the bucket
  atomicAdd(&bucket[threadIdx.x], temp[threadIdx.x]); 
}


// prefix sum algorithm to fill the offset buffer (sum of bucket values)
__global__ void fill_offset(int n, int *bucket, int *offset)
{
  extern __shared__ int temp[];
  int loc_tid = threadIdx.x;
  temp[loc_tid] = loc_tid>0 ? bucket[loc_tid - 1] : 0;
  __syncthreads();

  // printf("tid = %i && temp = %i\n", loc_tid, temp[loc_tid]);

  int outbuff = 0;

  for (int k=1; k<n; k<<=1)
  {
    // swap in and out buffers
    outbuff = 1 - outbuff;

    int inTID = outbuff*n+loc_tid;
    int outTID = (1 - outbuff)*n+loc_tid;
    if (loc_tid >= k)
    {
      temp[inTID] = temp[outTID] + temp[outTID-k];
      // printf("k = %i intTID %i  & value = %i , old = %i // outTID %i & value = %i // loc tid exec= %i\n",k, inTID, temp[inTID], old, outTID-k, temp[outTID-k], loc_tid);

    }
    else // simple copy
    {
      temp[inTID] = temp[outTID];
      // printf("kkk = %i intTID %i  & value = %i // outTID %i & value = %i // loc tid exec= %i\n",k, inTID, temp[inTID], outTID, temp[(1 - outbuff)*n+loc_tid], loc_tid);
    }
    __syncthreads();
    // if (loc_tid==0) printf("\n");
  }

  offset[loc_tid] = temp[outbuff*n+loc_tid];

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

  // auto start = std::chrono::steady_clock::now();

  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }

  // printf("\nTASK 1 CPU = %f", (float) (std::chrono::steady_clock::now() - start).count());

  // start = std::chrono::steady_clock::now();

  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      output[j++] = i;
    }
  }

  // printf("\nTASK 2 CPU = %f", (float) (std::chrono::steady_clock::now() - start).count());


}

void print_vec(int *vec, int size)
{
  for (int i=0; i<size; i++) {
    printf("%d ",vec[i]);
  }
  printf("\n");
}





int main() {
  const int n = 500000;


  // BEWARE !!! 
  // range can only go up to 128 bc of shared memory capacity in fill_offset prefix scan function...
  const int range = 128;

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
  printf("\n");


  // CPU

  int outputCPU[n];


  auto start = std::chrono::steady_clock::now();

  bucket_sort_CPU(key, outputCPU, n, range);


  float CPU = (std::chrono::steady_clock::now() - start).count();


  start = std::chrono::steady_clock::now();

  
  fill_bucket<<<(n+range-1)/range, range, range*sizeof(int)>>>(n, bucket, key);
  cudaDeviceSynchronize();

  // printf("\nTASK 1 GPU = %f", (float) (std::chrono::steady_clock::now() - start).count());


  // start = std::chrono::steady_clock::now();
  
  fill_offset<<<1, range, 2*range>>>(range, bucket, offset);
  cudaDeviceSynchronize();


  fill_key<<<(range+31)/32, 32>>>(range, bucket, offset, key);
  cudaDeviceSynchronize();

  // printf("\nTASK 2 GPU = %f", (float) (std::chrono::steady_clock::now() - start).count());


  // printf("\noutput gpu = \n KEY SORTED = ");
  // print_vec(key, n);
  // printf("\n");

  float GPU = (std::chrono::steady_clock::now() - start).count();

  printf("Time taken: for array of size = %i with range = %i  \n >> CPU: %f  \n >> GPU: %f", n, range,  CPU, GPU);

}
