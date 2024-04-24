#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], fxSIMD[N], fySIMD[N];
  int range[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = fxSIMD[i] = fySIMD[i] = 0;
    range[i] = i;
  }

  printf("\n");
  float a[N];

  for(int i=0; i<N; i++) {

    for(int j=0; j<N; j++) {
      if(i != j) {
        float rx = x[i] - x[j];
        float ry = y[i] - y[j];

        float r = std::sqrt(rx * rx + ry * ry);
        fx[i] -= rx * m[j]/(r *r * r);
        fy[i] -= ry * m[j]/(r *r * r);
      }
    }

    // HOMEWORK: vectorization of the for loop above for parameter j.
    {
      __m512 minusx_vec = _mm512_sub_ps(_mm512_set1_ps(x[i]), _mm512_load_ps(x)); // minusx_vec = (x[i]-x[j]) for all j=0..N-1
      
      __m512 minusy_vec = _mm512_sub_ps(_mm512_set1_ps(y[i]), _mm512_load_ps(y)); // minusy_vec = (y[i]-y[j]) for all j=0..N-1
    
      // if condition
      __m512 minusij_vec = _mm512_sub_ps(_mm512_set1_ps(i), _mm512_load_ps(range)); // minusij_vec = (i - j) for all j=0..N-1
      __m512 zeros = _mm512_set1_ps(0);
      __mmask16 mask_if = _mm512_cmp_ps_mask(minusij_vec, zeros, _MM_CMPINT_NE);

      __m512 rxvec = _mm512_mask_blend_ps(mask_if, zeros, minusx_vec);
      __m512 rxsqvec = _mm512_mul_ps(rxvec, rxvec);

      __m512 ryvec =  _mm512_mask_blend_ps(mask_if, zeros, minusy_vec);
      __m512 rysqvec = _mm512_mul_ps(ryvec, ryvec);

      __m512 rvec = _mm512_add_ps(rxsqvec, rysqvec);


      __mmask16 mask_div = _mm512_cmp_ps_mask(rvec, zeros, _MM_CMPINT_EQ);
      rvec = _mm512_mask_blend_ps(mask_div, rvec, _mm512_set1_ps(1));
      rvec = _mm512_rsqrt14_ps(rvec); //rvec contains 1/r now
      rvec = _mm512_mul_ps(rvec, _mm512_mul_ps(rvec, rvec)); //rvec contains now 1/(r*r*r)

      rvec = _mm512_mask_blend_ps(mask_div, rvec, zeros); // rvec contains either 1/(r*r*r) or 0.0 (if r=0)

      __m512 mvec = _mm512_load_ps(m);
      __m512 fxvec_temp = _mm512_mask_sub_ps(zeros, mask_if, zeros, _mm512_mul_ps(_mm512_mul_ps(rxvec, rvec), mvec));
      __m512 fyvec_temp = _mm512_mask_sub_ps(zeros, mask_if, zeros, _mm512_mul_ps(_mm512_mul_ps(ryvec, rvec), mvec));


      fxSIMD[i] = _mm512_reduce_add_ps(fxvec_temp);
      fySIMD[i] = _mm512_reduce_add_ps(fyvec_temp);
    }

    printf("COMPARISON for value %d ==> NON SIMD %g %g || SIMD = %g %g \n\n",i,fx[i],fy[i], fxSIMD[i],fySIMD[i]);

}
}
