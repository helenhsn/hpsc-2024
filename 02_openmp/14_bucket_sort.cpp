#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>


void bucket_sort_par(int n, int range, int THRESHOLD)
{
  double start;
  start = omp_get_wtime();

  omp_set_num_threads(std::min(range, THRESHOLD));
  printf("TASK 0 = %f\n", omp_get_wtime() - start);


  std::vector<int> key(n);
  
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    // printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 

  start = omp_get_wtime();
  #pragma omp parallel for shared(bucket) schedule(guided)
  for (int i=0; i<n; i++)
  {
    int number = key[i];
    #pragma omp atomic update
    bucket[number]++;
  }

  printf("TASK 1 = %f\n", omp_get_wtime() - start);


  start = omp_get_wtime();
  
  std::vector<int> temp(range,0);
  std::vector<int> offset(range,0);

  #pragma omp parallel for
  for (int i=1; i<range; i++)
  {
    offset[i] = bucket[i-1];
  }
  
  #pragma omp parallel
  for (int j=1; j<range; j<<=1) 
  {
    #pragma omp for
    for (int k=0; k<range; k++)
    {
      temp[k] = offset[k];
    }

    #pragma omp for
    for (int i=j; i<range; i++)
    {
      offset[i] += temp[i-j]; 
    }
  }

  printf("TASK 2= %f\n", omp_get_wtime() - start);

  start = omp_get_wtime();
  #pragma omp parallel for schedule(static)
  for (int i=0; i<range; i++) {
    int nb_occurences = bucket[i]; 
    int off = offset[i];
    for (int k=0; k<nb_occurences; k++) {
      key[k+off] = i;
    }
  }
  printf("TASK 3 = %f\n", omp_get_wtime() - start);


  // for (int i=0; i<n; i++) {
  //   printf("%d ",key[i]);
  // }
}

void bucket_sort(int n, int range)
{
  double start;
  std::vector<int> key(n);
  
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    // printf("%d ",key[i]);
  }
  printf("\n");
  
  start = omp_get_wtime();

  std::vector<int> bucket(range,0); 

  for (int i=0; i<n; i++)
    bucket[key[i]]++;

  printf("TASK 1 = %f\n", omp_get_wtime() - start);
  
  std::vector<int> temp(range,0);
  std::vector<int> offset(range, 0);


  start = omp_get_wtime();

  for (int i=1; i<range; i++) 
  {
    offset[i] = offset[i-1] + bucket[i-1];
  }

  printf("TASK 2 = %f\n", omp_get_wtime() - start);


  start = omp_get_wtime();

  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
  printf("TASK 3 = %f\n", omp_get_wtime() - start);

  // for (int i=0; i<n; i++) {
  //   printf("%d ",key[i]);
  // }
}

int main() {
  std::vector<int> n_values = {50, 500, 1000, 10000, 1000000, 100000000, 500000000};
  std::vector<int> range_values = {6, 50, 100, 1000, 50000, 100000, 500000};
  std::vector<int> thresholds = {32, 32, 32, 32, 32, 256, 1024};

  for (int i=0; i<n_values.size(); i++)
  {
    double start = omp_get_wtime();
    bucket_sort_par(n_values[i], range_values[i], thresholds[i]);
    double end_par = omp_get_wtime() - start;

    start = omp_get_wtime();
    bucket_sort(n_values[i], range_values[i]);
    double end_nonpar = omp_get_wtime() - start;
    printf("%i] --- For N=%i array length: \n OMP algo computed in %f seconds \n VS SEQUENTIAL algo computed in %f seconds\n",i, n_values[i], end_par, end_nonpar);
  }

}