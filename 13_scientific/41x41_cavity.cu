#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <iostream>
#include <chrono>

/**
 * NAVIER STOKES COMPUTATIONAL MODULE
 * using GLOBAL MEMORY ONLY
*/

using namespace std;
typedef vector<vector<float>> matrix;

__host__ void cudaErrorCheck(cudaError_t err)
{
    if (err != cudaSuccess)
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl << std::flush;
}

__global__ void updateP_Cols(
  int nx,
  int offsetLeft,
  float *p
)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if (tid < nx)
  {
    p[tid] = p[tid+nx];
    p[tid+offsetLeft] = 0.0;
  }
}

__global__ void updateP_Lines(
  int maxTid,
  int nx,
  int offsetBottom,
  float *p
)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  if (j < maxTid)
  { 
    int column = j*nx;
    p[column+offsetBottom] = p[column+offsetBottom-1]; // p[:, -1] = p[:, -2]
    p[column] = p[column+1]; // p[:, 0] = p[:, 1]
  }
}

__global__ void updateUV_RLBounds(
  int nx,
  int offsetLeft,
  float *u,
  float *v
)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if (tid < nx)
  {
    u[tid] = 0.0;
    v[tid] = 0.0;
    u[tid+offsetLeft] = 1.0;
    v[tid+offsetLeft] = 0.0;
  }
}

__global__ void updateUV_TBBounds(
  int ny,
  int nx,
  int offsetBottom,
  float *u,
  float *v
)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  if (j < ny)
  { 
    int column = j*nx;
    u[column] = 0.0;
    v[column] = 0.0;
    u[column+offsetBottom] = 0.0;
    v[column+offsetBottom] = 0.0;
  }
}

__global__ void updateUV(
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
  float *un,
  float *vn,
  float *p

)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;

  if (i > 0 && j > 0 && i< nx-1 && j <ny-1)
  {
    int tid = j*nx+i;

    float u_ij = un[tid];
    float v_ij = vn[tid];

    int jp1_i = (j+1)*nx+i;
    float v_jp1_i = vn[jp1_i];
    float u_jp1_i = un[jp1_i];

    int jm1_i = (j-1)*nx+i;
    float v_jm1_i = vn[jm1_i];
    float u_jm1_i = un[jm1_i];

    int j_ip1 = j*nx+i+1;
    float u_j_ip1 = un[j_ip1];
    float v_j_ip1 = vn[j_ip1];

    int j_im1 = j*nx+i-1;
    float u_j_im1 = un[j_im1];
    float v_j_im1 = vn[j_im1];

    u[tid] = u_ij - u_ij * dt / dx * (u_ij - u_j_im1)
                  - u_ij * dt / dy * (u_ij - u_jm1_i)
                  - dt / (2 * rho * dx) * (p[j*nx+i+1] - p[j*nx+i-1])
                  + nu * dt / dx2 * (u_j_ip1 - 2 * u_ij + u_j_im1)
                  + nu * dt / dy2 * (u_jp1_i - 2 * u_ij + u_jm1_i);

    v[tid] = v_ij - v_ij * dt / dx * (v_ij - v_j_im1)
                  - v_ij * dt / dy * (v_ij - v_jm1_i)
                  - dt / (2 * rho * dy) * (p[(j+1)*nx+i] - p[(j-1)*nx+i])
                  + nu * dt / dx2 * (v_j_ip1 - 2 * v_ij + v_j_im1)
                  + nu * dt / dy2 * (v_jp1_i - 2 * v_ij + v_jm1_i);

  }
}

__global__ void copyUnVn(
  int maxTid,
  int nx,
  float *un,
  float *u,
  float *vn,
  float *v
)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int tid = j*nx+i;

  if (tid < maxTid)
  {
    un[tid] = u[tid];
    vn[tid] = v[tid];
  }
}

__global__ void copyPN(
  int N, 
  int nx, 
  float *p,
  float *pn
)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int tid = j*nx+i;
  if (tid<N)
  {
    //copy
    pn[tid] = p[tid];
  }
}
__global__ void updateP(
  int nx, 
  int ny,
  float dx2,
  float dy2,
  float *p,
  float *pn,
  float *b
  )
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int tid = j*nx+i;
  if (i>0 && j>0 && i < (nx-1) && j <(ny-1))
  {
    p[tid] = (dy2*(pn[tid+1] + pn[tid-1]) + dx2*(pn[tid+nx] + pn[tid-nx]) - b[tid] * dx2 * dy2)/(2*(dx2 + dy2));

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
  printf("NAVIER STOKES FOR A 41x41 GRID -------->> EXECUTE: 41x41_plot.py // COMPARISON: 41x41_cavity.py\n");

  auto start = std::chrono::high_resolution_clock::now();


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
  float *pn;
  float *b;
  float *un;
  float *vn;

  cudaErrorCheck(cudaMallocManaged(&u, sizeof(float)*nx*ny));
  cudaErrorCheck(cudaMallocManaged(&v, sizeof(float)*nx*ny));
  cudaErrorCheck(cudaMallocManaged(&b, sizeof(float)*nx*ny));
  cudaErrorCheck(cudaMallocManaged(&p, sizeof(float)*nx*ny));
  cudaErrorCheck(cudaMallocManaged(&pn, sizeof(float)*nx*ny));
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

    for (int it =0; it<nit; ++it)
    {

      copyPN<<<dimGrid, dimBlock, 2*nbElements*sizeof(float)>>>(
        nbElements,
        nx,
        p,
        pn
      );
      cudaErrorCheck(cudaDeviceSynchronize());

      updateP<<<dimGrid, dimBlock, 2*nbElements*sizeof(float)>>>(
        nx,
        ny,
        dx2,
        dy2,
        p,
        pn,
        b
      );
      cudaErrorCheck(cudaDeviceSynchronize());

      updateP_Lines<<<dim3((ny+31)/32, 1, 1), dim3(32, 1, 1)>>>(
      ny,
      nx,
      ny-1,
      p);
      cudaErrorCheck(cudaDeviceSynchronize());

      updateP_Cols<<<dim3((nx+31)/32, 1, 1), dim3(32, 1, 1)>>>(
        nx,
        nx*(ny-1),
        p
      );
      cudaErrorCheck(cudaDeviceSynchronize());
    }


    copyUnVn<<<dimGrid, dimBlock>>>(
      nbElements,
      nx,
      un,
      u,
      vn,
      v
    );
    cudaErrorCheck(cudaDeviceSynchronize());


    updateUV<<<dimGrid, dimBlock>>>(
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
      un,
      vn,
      p
    );
    cudaErrorCheck(cudaDeviceSynchronize());

   

    updateUV_TBBounds<<<dim3((ny+31)/32, 1, 1), dim3(32, 1, 1)>>>(
      ny,
      nx,
      ny-1,
      u,
      v
    );
    cudaErrorCheck(cudaDeviceSynchronize());

    updateUV_RLBounds<<<dim3((nx+31)/32, 1, 1), dim3(32, 1, 1)>>>(
      nx,
      nx*(ny-1),
      u,
      v
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

  ufile.close();
  vfile.close();
  pfile.close();
}
