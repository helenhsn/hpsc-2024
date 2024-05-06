#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>
#include <iostream>

#define N 16

void print_tobits(unsigned short num)
{

  printf("\n");
  for(int i = N-1; i>-1; --i){

    printf("%u", (num & (1<<i)) ? 1:0);
  }
}

int main() {


  const int n = 73;

  printf("\n>>>>TESTING WITH ARRAY OF SIZE %i\n", n);
  
  int nb_slices = ceil((float)n/16);
  int ceil_n = nb_slices*16;
  int max_16index = (ceil_n==n) ? 16 : ceil_n-n;

  float x[ceil_n], y[ceil_n], m[ceil_n], fx[n], fy[n], fxSIMD[n], fySIMD[n];
  int range[ceil_n]; // array for the if condition in SIMD

  for(int i=0; i<ceil_n; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    range[i] = i;
    if (i < n)
    {
      fx[i] = fy[i] = fxSIMD[i] = fySIMD[i] = 0;
    }

  }

  // creating a mask for the last 16-float buffer who is not necessarily entirely used
  unsigned short mask_padding = 0;
  for (int i=0; i<max_16index; i++)
  {
      mask_padding |= (1<<(N-(i+1)));
  }

  for(int i=0; i<n; i++) {

    // running the basic algorithm to compare with SIMD version
    for(int j=0; j<n; j++) {
      if(i != j) {
        float rx = x[i] - x[j];
        float ry = y[i] - y[j];

        float r = std::sqrt(rx * rx + ry * ry);

        fx[i] -= rx * m[j]/(r *r * r);
        fy[i] -= ry * m[j]/(r *r * r);
      }
    }

    // j-loop written using SIMD instructions (HOMEWORK) 
    // it has been rewritten for so that it can work for any array size we want (odd, even, above 16 or below... whatever)

    for (int k=0;k<nb_slices;++k)
    {
      
      int offset = k*N;
    
      __m512 minusx_vec = _mm512_sub_ps(_mm512_set1_ps(x[i]), _mm512_loadu_ps(x + offset)); // minusx_vec = (x[i]-x[j]) for all j=0..N-1
      __m512 minusy_vec = _mm512_sub_ps(_mm512_set1_ps(y[i]), _mm512_loadu_ps(y + offset)); // minusy_vec = (y[i]-y[j]) for all j=0..N-1
    
      // if condition but in SIMD fashion (not very efficient -> should have used bit shifting to do that instead...)
      __m512 minusij_vec = _mm512_sub_ps(_mm512_set1_ps(i), _mm512_loadu_ps(range+offset)); // minusij_vec = (i - j) for all j=0..N-1
      __m512 zeros = _mm512_set1_ps(0);
      __mmask16 mask_if = _mm512_cmp_ps_mask(minusij_vec, zeros, _MM_CMPINT_NE);

      __m512 rxvec = _mm512_mask_blend_ps(mask_if, zeros, minusx_vec);
      __m512 rxsqvec = _mm512_mul_ps(rxvec, rxvec);

      __m512 ryvec =  _mm512_mask_blend_ps(mask_if, zeros, minusy_vec);
      __m512 rysqvec = _mm512_mul_ps(ryvec, ryvec);

      __m512 rvec = _mm512_add_ps(rxsqvec, rysqvec);

      __mmask16 mask_div = _mm512_cmp_ps_mask(rvec, zeros, _MM_CMPINT_EQ); // mask to filter the cases for which r is equal to 0

      // we cancel the values that we do not need (because of the padding) using the mask computed at the beginning
      if (k==(nb_slices-1)) {
        mask_div |=mask_padding;
      }


      rvec = _mm512_mask_blend_ps(mask_div, rvec, _mm512_set1_ps(1));
      rvec = _mm512_rsqrt14_ps(rvec); //rvec now contains 1/r or 1 (if r=0)
      rvec = _mm512_mul_ps(rvec, _mm512_mul_ps(rvec, rvec)); //rvec now contains 1/(r*r*r) or 1

      rvec = _mm512_mask_blend_ps(mask_div, rvec, zeros); // rvec contains either 1/(r*r*r) or 0.0 (if r=0)

      __m512 mvec = _mm512_loadu_ps(m + offset);
      __m512 fxvec_temp = _mm512_mask_sub_ps(zeros, mask_if, zeros, _mm512_mul_ps(_mm512_mul_ps(rxvec, rvec), mvec));
      __m512 fyvec_temp = _mm512_mask_sub_ps(zeros, mask_if, zeros, _mm512_mul_ps(_mm512_mul_ps(ryvec, rvec), mvec));
      
      fxSIMD[i] += _mm512_reduce_add_ps(fxvec_temp); // adding the result of each 16-float slice
      fySIMD[i] += _mm512_reduce_add_ps(fyvec_temp);
    }
    printf("\n%d ==> NON SIMD %g %g || SIMD = %g %g",i,fx[i],fy[i], fxSIMD[i],fySIMD[i]);
  }
}
