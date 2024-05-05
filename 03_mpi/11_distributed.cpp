#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

struct Body {
  double x, y, m, fx, fy;
};

void CPU_NONPAR(const int N, int rank, Body **init_ptr, Body **ibody_ptr) {
    Body jbody[N];

    Body *init = (*init_ptr);
    Body *ibody = (*ibody_ptr);

    srand48(0);

    for(int i=0; i<N; i++) {
      ibody[i].x = jbody[i].x = init[i].x=drand48();
      ibody[i].y = jbody[i].y = init[i].y= drand48();
      ibody[i].m = jbody[i].m = init[i].m= drand48();
      ibody[i].fx = jbody[i].fx = ibody[i].fy = jbody[i].fy = init[i].fy = init[i].fy= 0;
      // printf("%i: ibody = %g %g \n", id, ibody[i].x, ibody[i].y);


    }

    for(int i=0; i<N; i++) {
      for(int j=0; j<N; j++) {
        double rx = ibody[i].x - jbody[j].x;
        double ry = ibody[i].y - jbody[j].y;
        double r = std::sqrt(rx * rx + ry * ry);
        if (r > 1e-15) {
          ibody[i].fx -= rx * jbody[j].m / (r * r * r);
          ibody[i].fy -= ry * jbody[j].m / (r * r * r);
        }
      }
    }

  // waiting for every rank to initialize its (same) array of initial values <!>
  MPI_Barrier(MPI_COMM_WORLD);
}


void CPU_PAR(int argc, char** argv, const int N, int size, int rank, Body *init, Body **ibody_ptr)
{
  
  Body jbody[N/size];
  Body *ibody = (* ibody_ptr) ;
  for(int i=0; i<N/size; i++) {
    int id = rank*N/size+i;
    ibody[i].x = jbody[i].x =init[id].x;
    ibody[i].y = jbody[i].y = init[id].y;
    ibody[i].m = jbody[i].m = init[id].m;
    ibody[i].fx = jbody[i].fx = ibody[i].fy = jbody[i].fy = init[id].fx;
    // printf("%i: ibody = %g %g \n", id, ibody[i].x, ibody[i].y);
  }
  int recv_from = (rank + 1) % size;
  int send_to = (rank - 1 + size) % size;

  MPI_Datatype MPI_BODY;
  MPI_Type_contiguous(5, MPI_DOUBLE, &MPI_BODY);
  MPI_Type_commit(&MPI_BODY);

  MPI_Win win; 
  MPI_Win_create(jbody, N/size*sizeof(Body), sizeof(Body), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
  MPI_Win_fence(0, win);
  
  for(int irank=0; irank<size; irank++) {   
    
    MPI_Win_fence(0, win);
    MPI_Put(jbody, N/size, MPI_BODY, send_to, 0, N/size, MPI_BODY, win);
    MPI_Win_fence(0, win);

    for(int i=0; i<N/size; i++) {
      for(int j=0; j<N/size; j++) {
          double rx = ibody[i].x - jbody[j].x;
          double ry = ibody[i].y - jbody[j].y;
          double r = std::sqrt(rx * rx + ry * ry);
          if (r > 1e-15) {
            ibody[i].fx -= rx * jbody[j].m / (r * r * r);
            ibody[i].fy -= ry * jbody[j].m / (r * r * r);
          }
    }
   }
  }
  MPI_Win_free(&win);

}


int main(int argc, char** argv) {
  const int N = 20;
  MPI_Init(&argc, &argv);

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
  
  
  /* same array of values shared between the CPU_NONPAR and CPU_PAR functions*/
  Body *init_array = (Body *) calloc(N, sizeof(Body));

  /* output arrays*/
  Body *ibody_nonpar = (Body *) calloc(N, sizeof(Body));
  Body *ibody_par = (Body *) calloc(N/size, sizeof(Body));
  
  /*
  CPU_NONPAR = every rank runs the same n-body algorithm on the same array of initial values
  this equals to the traditionnal sequential algorithm but run 4 times because of mpi ranks.
  */
  CPU_NONPAR(N, rank, (Body **) &init_array, (Body **) &ibody_nonpar);

  /*
  CPU_PAR = every rank runs the same n-body algorithm on an N/size slice of the array of initial values
  contains the homework for the 9th of May (uses MPI_Window).
  */
  CPU_PAR(argc, argv, N, size, rank, init_array, (Body **) &ibody_par);

  /* Comparing results for coherency*/
  for(int irank=0; irank<size; irank++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(irank==rank) {
      for(int i=0; i<N/size; i++) {
        int global_id = i+rank*N/size;
        printf("%d: CPU NON PAR = %g %g || CPU PAR = %g %g\n",global_id, ibody_nonpar[global_id].fx,ibody_nonpar[global_id].fy, ibody_par[i].fx,ibody_par[i].fy);
      }
    }
  }
  
  MPI_Finalize();
}
