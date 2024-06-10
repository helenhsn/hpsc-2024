#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <iostream>
#include <chrono>


using namespace std;
typedef vector<vector<float>> matrix;

__host__ void cudaErrorCheck(cudaError_t err)
{
    if (err != cudaSuccess)
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl << std::flush;
}

__global__ void updateUV_SM(
  int N,
  int nx,
  int ny,
  float dt,
  float dx,
  float dy,
  float dx2,
  float dy2,
  float nu,
  float rho,
  float *u,
  float *v,
  float *p
)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int tid = j*nx+i;
  if (i < nx && j <ny)
  {
    extern __shared__ float u_v_n[];
    // load u_n and v_n into shared mem into on 1D array
    u_v_n[tid] = u[tid];
    u_v_n[tid+N] = v[tid];
    float newU = 0.0;
    float newV = 0.0;
    __syncthreads();

    
    if (i > 0 && j > 0 && i< nx-1 && j <ny-1)
    {
      float u_ij = u_v_n[tid];
      float v_ij = u_v_n[tid+N];

      int jp1_i = (j+1)*nx+i;
      float v_jp1_i = u_v_n[jp1_i+N];
      float u_jp1_i = u_v_n[jp1_i];

      int jm1_i = (j-1)*nx+i;
      float v_jm1_i = u_v_n[jm1_i+N];
      float u_jm1_i = u_v_n[jm1_i];

      int j_ip1 = j*nx+i+1;
      float u_j_ip1 = u_v_n[j_ip1];
      float v_j_ip1 = u_v_n[j_ip1+N];

      int j_im1 = j*nx+i-1;
      float u_j_im1 = u_v_n[j_im1];
      float v_j_im1 = u_v_n[j_im1+N];

      newU = u_ij - u_ij * dt / dx * (u_ij - u_j_im1)
                    - u_ij * dt / dy * (u_ij - u_jm1_i)
                    - dt / (2 * rho * dx) * (p[j*nx+i+1] - p[j*nx+i-1])
                    + nu * dt / dx2 * (u_j_ip1 - 2 * u_ij + u_j_im1)
                    + nu * dt / dy2 * (u_jp1_i - 2 * u_ij + u_jm1_i);

      newV = v_ij - v_ij * dt / dx * (v_ij - v_j_im1)
                    - v_ij * dt / dy * (v_ij - v_jm1_i)
                    - dt / (2 * rho * dy) * (p[(j+1)*nx+i] - p[(j-1)*nx+i])
                    + nu * dt / dx2 * (v_j_ip1 - 2 * v_ij + v_j_im1)
                    + nu * dt / dy2 * (v_jp1_i - 2 * v_ij + v_jm1_i);
    }
    __syncthreads();

    // UPDATE BOUNDARIES. Lines first and then columns.

    // u[:,-1]=0.0 v[:,-1]=0.0
    if (i==(nx-1))
    {
      newU = 0.0;
      newV = 0.0;
    }
    __syncthreads();
    
    // u[:,0]=0.0 v[:,0]=0.0

    if (i==0)
    {
      newU = 0.0;
      newV = 0.0;
    }
    __syncthreads();

    //  u[0, :]  = 0 , v[0, :]  = 0
    if (j==0)
    {
      newU = 0.0;
      newV = 0.0;
    }
    __syncthreads();
    //  u[-1, :] = 1 , v[-1, :]  = 0
    if (j==(ny-1))
    {
      newU = 1.0;
      newV = 0.0;
    }
    __syncthreads();

    // Final step: write into GM
    u[tid] = newU;
    v[tid] = newV;
  }
}

__global__ void updateP(
  int N, 
  int nit,
  int nx, 
  int ny,
  float dx2,
  float dy2,
  float *p,
  float *b
  )
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int tid = j*nx+i;
  if (i < nx && j <ny)
  {
    extern __shared__ float tempP[];
    // 2 copies of p: tempP[0..N-1] is p and tempP[N..N*N-1] is p_n
    tempP[tid] = p[tid];
    int tid_pn=tid+N;
    tempP[tid_pn] = p[tid];
    __syncthreads();

    float b_ij_c = 0.0;
    float sum2_dx_dy = 0.0;
    if (i>0 && j>0 && i < (nx-1) && j <(ny-1))
    {
      b_ij_c = b[tid] * dx2 * dy2;
      sum2_dx_dy =  2*(dx2 + dy2);
    }
  
    
    for (int it=0; it<nit; it++)
    {
      tempP[tid_pn]=tempP[tid];
      __syncthreads();

      if (i>0 && j>0 && i < (nx-1) && j <(ny-1))
      {
        // updating p[j, i]
        // if (j==4 && i==2) printf("res1=%.10f, res2 = %.10f, bc =%.10f", dy2*(tempP[tid_pn+1] + tempP[tid_pn-1]), dx2*(tempP[tid_pn+nx] + tempP[tid_pn-nx]), b_ij_c);
        tempP[tid] = (dy2*(tempP[tid_pn+1] + tempP[tid_pn-1]) + dx2*(tempP[tid_pn+nx] + tempP[tid_pn-nx]) - b_ij_c)/sum2_dx_dy;

      }
      __syncthreads();

      // updating limits -> threads divergence!!
      // array of indices which contains, for each index, the index of change
       // p[:, -1] = p[:, -2]
      if (i==(nx-1))
      {
        // if (tid-1 >= maxTid || (tid-1) < 0) {printf("\n nx=%i i=%i j=%i tid = %i, tid = %i, tid-1 = %i maxTid=%i", nx, i, j, tid, j*nx+i, tid-1, maxTid);}

        tempP[tid] = tempP[tid-1];
      }
      __syncthreads();
      
      
      // p[0, :] = p[1, :]
      if (j==0)
      {
        tempP[tid] = tempP[tid+nx];
      }
      __syncthreads();

     

      //p[:, 0] = p[:, 1]
      if (i==0)
      {
        tempP[tid] = tempP[tid+1];
      }
      __syncthreads();

      //  p[-1, :] = 0
      if (j==(ny-1))
      {
        tempP[tid] = 0.0;
      }
      __syncthreads();
    }

    // end: copy shared array 'p' into global memory array 'p'
    p[tid] = tempP[tid];
  }
}

__global__ void computeB(
  int nx, 
  int ny, 
  float rho,
  float dt,
  float dx,
  float dy,
  float *u, 
  float *v,
  float *b
  )
{
   int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;

  if (i > 0 && j > 0 && i< nx-1 && j <ny-1)
  {
    int tid = j*nx+i;
    int jp1_i = tid+nx;
    float v_jp1_i = v[jp1_i];
    float u_jp1_i = u[jp1_i];

    int jm1_i = tid-nx;
    float v_jm1_i = v[jm1_i];
    float u_jm1_i = u[jm1_i];

    int j_ip1 = tid+1;
    float u_j_ip1 = u[j_ip1];
    float v_j_ip1 = v[j_ip1];

    int j_im1 = tid-1;
    float u_j_im1 = u[j_im1];
    float v_j_im1 = v[j_im1];

    // b[j, i]
    float u_j_di = (u_j_ip1 - u_j_im1) / (2 * dx);
    float v_j_di = (v_jp1_i - v_jm1_i) / (2 * dy);
    b[tid] = rho * (
      1 / dt * ((u_j_ip1 - u_j_im1) / (2 * dx) + (v_jp1_i - v_jm1_i) / (2 * dy)) 
      - u_j_di*u_j_di 
      - ((u_jp1_i - u_jm1_i) * (v_j_ip1 - v_j_im1) / (2 * dy * dx)) 
      - v_j_di*v_j_di
    );
  }
}


__global__ void init_arrays(
  int maxTid, 
  int nx, 
  float *u, 
  float *v, 
  float *p
  )
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int tid = j*nx+i;
  if (tid < maxTid)
  {
    u[tid] = 0.0;
    v[tid] = 0.0;
    p[tid] = 0.0;
  }
}

int main() {
  auto start = std::chrono::high_resolution_clock::now();
  float upP = 0.0;


  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dx2 = dx*dx;
  double dy = 2. / (ny - 1);
  double dy2 = dy*dy;
  double dt = .01;
  double rho = 1.;
  double nu = .02;

  int nbElements = nx*ny;

  float *u;
  float *v;
  float *p;
  float *b;
  float *un;
  float *vn;

  cudaErrorCheck(cudaMallocManaged(&u, sizeof(float)*nx*ny));
  cudaErrorCheck(cudaMallocManaged(&v, sizeof(float)*nx*ny));
  cudaErrorCheck(cudaMallocManaged(&b, sizeof(float)*nx*ny));
  cudaErrorCheck(cudaMallocManaged(&p, sizeof(float)*nx*ny));
  cudaErrorCheck(cudaMallocManaged(&un, sizeof(float)*nx*ny));
  cudaErrorCheck(cudaMallocManaged(&vn, sizeof(float)*nx*ny));

  dim3 dimGrid((nx+31)/32, (ny+31)/32, 1);
  dim3 dimBlock(32, 32, 1);

  // initializing u, v, p to 0.0
  init_arrays<<<dimGrid, dimBlock>>>(nbElements, nx, u, v, p);


  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");
  for (int n=0; n<nt; n++) {

    computeB<<< dim3((nx-2+31)/32,(ny-2+31)/32,1), dim3(32, 32, 1)>>>(
      nx,
      ny,
      rho,
      dt,
      dx,
      dy,
      u,
      v,
      b
    );

    cudaErrorCheck(cudaDeviceSynchronize());

    auto t = std::chrono::high_resolution_clock::now();

    updateP<<<dimGrid, dimBlock, 2*nbElements*sizeof(float)>>>(
      nbElements,
      nit,
      nx,
      ny,
      dx2,
      dy2,
      p,
      b
    );
    cudaErrorCheck(cudaDeviceSynchronize());

    std::chrono::duration<double> tot = std::chrono::high_resolution_clock::now() - t;
    upP+= tot.count();


    updateUV_SM<<<dimGrid, dimBlock, 2*nbElements*sizeof(float)>>>(
      nbElements,
      nx,
      ny,
      dt,
      dx,
      dy,
      dx2,
      dy2,
      nu,
      rho,
      u,
      v,
      p
    );
    cudaErrorCheck(cudaDeviceSynchronize());

    if (n%10==0) {
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          ufile << u[j*nx+i] << " ";
      ufile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          vfile << v[j*nx+i] << " ";
      vfile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          pfile << p[j*nx+i] << " ";
      pfile << "\n";
    }
  }
  std::chrono::duration<double> total = std::chrono::high_resolution_clock::now() - start;
    printf("\nSIMULATION executed on GPU in = %f\n", total.count());
    printf("\n update of P in = %f\n", upP);

  ufile.close();
  vfile.close();
  pfile.close();
}
