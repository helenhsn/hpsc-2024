#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

#define THRESHOLD 512

void merge(std::vector<int>& vec, int begin, int mid, int end) {
  std::vector<int> tmp(end-begin+1);
  int left = begin;
  int right = mid+1;

  for (int i=0; i<tmp.size(); i++) { 
    if (left > mid)
    {
      tmp[i] = vec[right++];
    }
    else if (right > end)
    {
      tmp[i] = vec[left++];
    }
    else if (vec[left] <= vec[right])
    {
      tmp[i] = vec[left++];
    }
    else
    {
      tmp[i] = vec[right++];
    }
  }

  for (int i=0; i<tmp.size(); i++) 
    vec[i+begin] = tmp[i];
}

void  merge_sort_nonpar(std::vector<int>& vec, int begin, int end)
{
  if(begin < end) {
    int mid = (begin + end) / 2;
    merge_sort_nonpar(vec, begin, mid);
    merge_sort_nonpar(vec, mid+1, end);
    merge(vec, begin, mid, end);
  }
}

void merge_sort(std::vector<int>& vec, int begin, int end) {
  if(begin < end) {
    int mid = (begin + end) / 2;
    if (end+1-begin>THRESHOLD)
    {
      // printf("PB");
      #pragma omp task shared(vec) firstprivate(mid, begin) untied
      merge_sort(vec, begin, mid);

      #pragma omp task shared(vec) firstprivate(mid, end) untied
      merge_sort(vec, mid+1, end);
      
      #pragma omp taskwait
    }
    else
    {
      // printf("yo");
      merge_sort_nonpar(vec, begin, mid);
      merge_sort_nonpar(vec, mid+1, end);
    }
    merge(vec, begin, mid, end);
  }
}

int main() {

  std::vector<int> n_values = {50, 500, 100000, 1000000, 50000000};
  for (int i=0; i<n_values.size(); i++)
  {
    int n = n_values[i];
    std::vector<int> vec(n);
    for (int i=0; i<n; i++) {
      vec[i] = rand() % (10 * n);
      // printf("%d ",vec[i]);
    }

    double start = omp_get_wtime();
    omp_set_num_threads(std::min(int(n*0.5), THRESHOLD));
    #pragma omp parallel firstprivate(n)
    {
      #pragma omp single
      #pragma omp task untied
      {
        merge_sort(vec, 0, n-1);
      }
    }
    double elapsed_seconds_par = omp_get_wtime() - start;

    start = omp_get_wtime();
    merge_sort(vec, 0, n-1);
    double elapsed_seconds_nonpar = omp_get_wtime() - start;
    
    printf("For array of size N = %i : elapsed time (seconds) for OMP algo  = %f VS sequential algo = %f \n", n, elapsed_seconds_par, elapsed_seconds_nonpar);
  }
}