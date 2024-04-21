#include <cstdio>
#include <iostream>
#include <omp.h>
#include <vector>

#define THRESHOLD 256

double compute_pi_par(long n, long NUM_THREADS)
{
	
	double dx = 1. / n;
	double pi = 0.0;

	omp_set_num_threads(NUM_THREADS);
	double temp[NUM_THREADS];

	printf("(NTHREADS = %i)", omp_get_max_threads());

	#pragma omp parallel for schedule(guided)
	for (int tid =0; tid<NUM_THREADS; tid++)
	{
		temp[tid] = 0.0;
	}


	#pragma omp parallel for firstprivate(dx) reduction(+:temp[:NUM_THREADS])
	for (long i=0; i<n; i++) {
		double x = (i+0.5)*dx;
		temp[omp_get_thread_num()] += 4.0/(1.0+x*x)*dx;	
	}

	#pragma omp parallel for reduction(+:pi)
	for (int tid=0; tid<NUM_THREADS; ++tid)
	{
		pi += temp[tid];
	}

	return pi;
}

double compute_pi_nonpar(long n)
{
	double dx = 1. / n;
  	double pi = 0;
	for (long i=0; i<n; i++) {
    	double x = (i + 0.5) * dx;
    	pi += 4.0 / (1.0 + x * x) * dx;
  	}
	return pi;
}

int main() {

	std::vector<long> n_values = {50, 10000, 500000, 1000000000, 21474836470};
	for (long &n: n_values)
	{
		long n_threads = n/5 > THRESHOLD ? THRESHOLD : n/10;

		double start = omp_get_wtime();
		double pi_par = compute_pi_par(n, n_threads);
		double elapsed_seconds_par = omp_get_wtime() - start;

		start = omp_get_wtime();
		double pi_nonpar = compute_pi_nonpar(n);
		double elapsed_seconds_nonpar = omp_get_wtime() - start;

		printf("FOR N=%li elements: \n OMP algo : PI = %17.15f computed in %f seconds \n VS SEQUENTIAL algo : PI = %17.15f computed in %f seconds\n\n",n, pi_par, elapsed_seconds_par, pi_nonpar, elapsed_seconds_nonpar);
	}
}
