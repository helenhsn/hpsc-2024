#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <iostream>
#include <chrono>

using namespace std;
typedef vector<vector<float>> matrix;

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

  matrix u(ny,vector<float>(nx));
  matrix v(ny,vector<float>(nx));
  matrix p(ny,vector<float>(nx));
  matrix b(ny,vector<float>(nx));
  matrix un(ny,vector<float>(nx));
  matrix vn(ny,vector<float>(nx));
  matrix pn(ny,vector<float>(nx));
  
  
  for (int j=0; j<ny; j++) {
    for (int i=0; i<nx; i++) {
      u[j][i] = 0;
      v[j][i] = 0;
      p[j][i] = 0;
      b[j][i] = 0;
    }
  }
  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");
  for (int n=0; n<nt; n++) {

    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
        float tempU = (u[j][i+1] - u[j][i-1]) / (2 * dx);
        float tempU_sq = tempU*tempU;
        float tempV = (v[j+1][i] - v[j-1][i]) / (2 * dy);
        float tempV_sq = tempV*tempV;
        b[j][i] = rho * (
                  1 / dt * (tempU+ tempV) 
                  - tempU_sq
                  - ((u[j+1][i] - u[j-1][i])*(v[j][i+1] - v[j][i-1]) / (2 * dx * dy)) 
                  - tempV_sq
        );
      }
    }
    for (int it=0; it<nit; it++) {

      // copy p into pn
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
	        pn[j][i] = p[j][i];


      for (int j=1; j<ny-1; j++) {
        for (int i=1; i<nx-1; i++) {
	        p[j][i] = (
                    dy2 * (pn[j][i+1] + pn[j][i-1]) 
                    + dx2 * (pn[j+1][i] + pn[j-1][i]) 
                    - b[j][i] * dx2 * dy2/ (2 * (dx2 + dy2))
                    );
	      }
      }
      for (int j=0; j<ny; j++) {
        // Compute p[j][0] and p[j][nx-1]
        p[j][nx-1] = p[j][nx-2]; 
        p[j][0] = p[j][1]; 
      }
      for (int i=0; i<nx; i++) {
	      // Compute p[0][i] and p[ny-1][i]
        p[0][i] = p[1][i];
        p[ny-1][i] = 0;  
      }
    }
    for (int j=0; j<ny; j++) {
      for (int i=0; i<nx; i++) {
        un[j][i] = u[j][i];
	      vn[j][i] = v[j][i];
      }
    }
    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
	      // Compute u[j][i] and v[j][i]
        u[j][i] = un[j][i] 
                    - un[j][i] * dt / dx * (un[j][i] - un[j][i-1])
                    - vn[j][i] * dt / dy * (un[j][i] - un[j-1][i])
                    - dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1])
                    + nu * dt / dx2 * (un[j][i+1] - 2 * un[j][i] + un[j][i-1])
                    + nu * dt / dy2 * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);

          v[j][i] = vn[j][i] 
                    - un[j][i] * dt / dx * (vn[j][i] - vn[j][i-1])
                    - vn[j][i] * dt / dy * (vn[j][i] - vn[j-1][i])
                    - dt / (2 * rho * dy) * (p[j+1][i] - p[j-1][i])
                    + nu * dt / dx2 * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1])
                    + nu * dt / dy2 * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
      
      }
    }
    // updating bounds

    for (int j = 0; j < ny; ++j) {
        u[j][0] = 0;
        u[j][nx - 1] = 0;
        v[j][0] = 0;
        v[j][nx - 1] = 0;
    }
    for (int i = 0; i < nx; ++i) {
        u[0][i] = 0;
        u[ny-1][i] = 1;
        v[0][i] = 0;
        v[ny-1][i] = 0;
    }
    if (n % 10 == 0) {
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          ufile << u[j][i] << " ";
      ufile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          vfile << v[j][i] << " ";
      vfile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          pfile << p[j][i] << " ";
      pfile << "\n";
    }
  }
  std::chrono::duration<double> total = std::chrono::high_resolution_clock::now() - start;
  printf("\nSIMULATION executed on CPU (C++) in = %f\n", total.count());

  ufile.close();
  vfile.close();
  pfile.close();
}
