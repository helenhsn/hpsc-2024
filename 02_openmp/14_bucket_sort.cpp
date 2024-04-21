#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

void bucket_sort_par(int n, int range)
{
  omp_set_num_threads(std::min(range, 3000));
  printf("n threads = %i", omp_get_max_threads());
  std::vector<int> key(n);
  
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    // printf("%d ",key[i]);
  }
  printf("\n");

  int bucket[range] = {0}; 

  #pragma omp parallel for shared(bucket)
  for (int i=0; i<n; i++)
  {
    #pragma omp atomic update
    bucket[key[i]]++;
  }

  std::vector<int> temp(range,0);
  std::vector<int> offset(range, 0);


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

  #pragma omp parallel for shared(offset)
  for (int i=0; i<range; i++) {
    for (int k=0; k<bucket[i]; k++) {
      key[offset[i]] = i;
      #pragma omp atomic update
      offset[i]++;
    }
  }

  // for (int i=0; i<n; i++) {
  //   printf("%d ",key[i]);
  // }
}

void bucket_sort(int n, int range)
{
  
  std::vector<int> key(n);
  
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    // printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 

  for (int i=0; i<n; i++)
    bucket[key[i]]++;

  std::vector<int> temp(range,0);
  std::vector<int> offset(range, 0);


  for (int i=1; i<range; i++) 
  {
    offset[i] = offset[i-1] + bucket[i-1];
  }


  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  // for (int i=0; i<n; i++) {
  //   printf("%d ",key[i]);
  // }
}

int main() {
  std::vector<int> n_values = {50, 500, 1000, 10000, 1000000, 100000000, 500000000};
  std::vector<int> range_values = {5, 50, 100, 1000, 5000, 10000, 50000};

  for (int i=0; i<n_values.size(); i++)
  {
    double start = omp_get_wtime();
    bucket_sort_par(n_values[i], range_values[i]);
    double end_par = omp_get_wtime() - start;

    start = omp_get_wtime();
    bucket_sort(n_values[i], range_values[i]);
    double end_nonpar = omp_get_wtime() - start;
    printf("%i] --- For N=%i, elapsed seconds PAR algo = %f // elapsed seconds NONPAR algo = %f\n",i, n_values[i], end_par, end_nonpar);
  }

}