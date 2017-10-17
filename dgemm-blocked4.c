/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */
#include <malloc.h>
#include <stdio.h>
#include "emmintrin.h"
const char* dgemm_desc = "cc-mh blocked dgemm.";

// Declare size of L1 cache,3*L1^2 < Mfast
// Notice! Since we use unrolling here, (i.e 5 2x2 blocks), the L1 BLOCK size should be multiple of 10
#if !defined(BLOCK_SIZE_L1)
#define BLOCK_SIZE_L1 30
// #define BLOCK_SIZE 719
#endif

//Declare size of L2 cache, should not fit in L1 but fit in L2
#if !defined(BLOCK_SIZE_L2)
#define BLOCK_SIZE_L2 256
#endif

#define min(a,b) (((a)<(b))?(a):(b))
/* This is the entrance to do bloking called from square_dgemm.
 * As to do a 2-level cache blocking, this is the L2-blokcing,
 * L1 blocking process is called inside loop
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */

void do_block(int lda, int M, int N, int K, double* A, double* B, double* C){
  for (int i = 0; i < M; i += BLOCK_SIZE_L1)
  {
    int Ms = min(BLOCK_SIZE_L1, M-i);
    for (int j = 0; j < N; j += BLOCK_SIZE_L1){
      /* Accumulate block dgemms into block of C */
      int Ns = min (BLOCK_SIZE_L1, N-j);
      for (int k = 0; k < K; k += BLOCK_SIZE_L1*2)
      {
         int Ks = min (BLOCK_SIZE_L1*2, K-k);
        //  if(Ms%10 !=0 || Ns%10 != 0 || Ks%10 != 0)
        //     do_block_l1_naive(lda, Ms, Ns, Ks, A + i*lda + k, B + k*lda + j, C + i*lda + j);
        //  else
              do_block_l1_SIMD(lda, Ms, Ns, Ks, A + i*lda + k, B + k*lda + j, C + i*lda + j);
      }
    }
  }
}
void do_block_l1_naive (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */
    for (int j = 0; j < N; ++j)
    {
      /* Compute C(i,j) */
      double cij = C[i*lda+j];
      for (int k = 0; k < K; ++k)
#ifdef TRANSPOSE
      	cij += A[i*lda+k] * B[j*lda+k];
#else
      	cij += A[i*lda+k] * B[k*lda+j];
#endif
      C[i*lda+j] = cij;
    }
}
/* L1 blocking : Square blocked matrix multiply
 * Refer to CS267 Algorithm 3
 * Pseduo Code :
 * for i = 1:M
 *    for j= 1:N
   *    {Read Cij into fast memory}
   *    for k= 1: K
   *       {Read Aik into fast memory}
   *       {Read Bkj into fast memmory}
   *       Cij := Cij + Aik * Bkj
   *    end for
   *    {Write back Cij to slow memory}
 * where C is M-by-N, A is M-by-K, and B is K-by-N.

 * Other features:
 *- use register
 *- unrolling: since _m128d is a pair of 64 bit numbers, one clue is that one dimension of the tile should be "2" or a multiple of 2.
                The other is that there are only 16 registers.
                You have a few choices - make all 3 matrices fit into the registers(3 2X2 block),
                or some subset  (e.g. 2 matrices, C,A, then 5 2X2 block) and stream the other(B) matrix from the L1 cache.*/

void do_block_l1_SIMD(int lda, int M, int N, int K, double* A, double* B, double* C){
  int i, j;
  for(i=0; i<M; i+=2){
    for(j=0; j+9<N; j+=10){
      //Each time load 5 2x2 blocks from C into register
      //Load 128-bits (2 packed double-precision (64-bit) floating-point elements) from memory into dst. mem_addr must be aligned on a 16-byte boundary or a general-protection exception may be generated.
      register __m128d c1 = _mm_load_pd(C+i*lda + j);
      register __m128d c2 = _mm_load_pd(C+i*lda + j + lda);
      register __m128d c3 = _mm_load_pd(C+i*lda + j + 2);
      register __m128d c4 = _mm_load_pd(C+i*lda + j + 2 + lda);
      register __m128d c5 = _mm_load_pd(C+i*lda + j + 4);
      register __m128d c6 = _mm_load_pd(C+i*lda + j + 4 + lda);
      register __m128d c7 = _mm_load_pd(C+i*lda + j + 6);
      register __m128d c8 = _mm_load_pd(C+i*lda + j + 6 + lda);
      register __m128d c9 = _mm_load_pd(C+i*lda + j + 8);
      register __m128d c10 = _mm_load_pd(C+i*lda + j + 8 + lda);

      for(int k=0; k<K; k+=2){
        //keep a1 a2 a3 a4 inside registers until finish using them
        //Notice we fill 128bits register with 64 double, twice, for A block
        register __m128d a1 = _mm_load1_pd(A+i*lda + k);
        register __m128d a2 = _mm_load1_pd(A+i*lda + k + lda);
        register __m128d a3 = _mm_load1_pd(A+i*lda + k + 1);
        register __m128d a4 = _mm_load1_pd(A+i*lda + k + lda + 1);


        //load matrix B streamly 5 times for 5 block

        //s1: fill c1 c2
        register __m128d b1 = _mm_load_pd(B + k*lda+ j);
        register __m128d b2 = _mm_load_pd(B + k*lda + j + lda);

        //Vectorize the inner loop
        c1 = _mm_add_pd(c1, _mm_mul_pd(a1, b1));
        c1 = _mm_add_pd(c1, _mm_mul_pd(a3, b2));

        c2 = _mm_add_pd(c2, _mm_mul_pd(a2, b1));
  			c2 = _mm_add_pd(c2, _mm_mul_pd(a4, b2));

        //s2: fill c3 c4
        b1 = _mm_load_pd(B + k*lda + j + 2);
        b2 = _mm_load_pd(B + k*lda + j + lda + 2);

        //Vectorize the inner loop
        c3 = _mm_add_pd(c3, _mm_mul_pd(a1, b1));
        c3 = _mm_add_pd(c3, _mm_mul_pd(a3, b2));

        c4 = _mm_add_pd(c4, _mm_mul_pd(a2, b1));
  			c4 = _mm_add_pd(c4, _mm_mul_pd(a4, b2));

        //s1: fill c5 c6
        b1 = _mm_load_pd(B + k*lda + j + 4);
        b2 = _mm_load_pd(B + k*lda + j + lda + 4);

        //Vectorize the inner loop
        c5 = _mm_add_pd(c5, _mm_mul_pd(a1, b1));
        c5 = _mm_add_pd(c5, _mm_mul_pd(a3, b2));

        c6 = _mm_add_pd(c6, _mm_mul_pd(a2, b1));
        c6 = _mm_add_pd(c6, _mm_mul_pd(a4, b2));

        //s1: fill c1 c2
        b1 = _mm_load_pd(B + k*lda + j + 6);
        b2 = _mm_load_pd(B + k*lda + j + lda + 6);

        //Vectorize the inner loop
        c7 = _mm_add_pd(c7, _mm_mul_pd(a1, b1));
        c7 = _mm_add_pd(c7, _mm_mul_pd(a3, b2));

        c8 = _mm_add_pd(c8, _mm_mul_pd(a2, b1));
        c8 = _mm_add_pd(c8, _mm_mul_pd(a4, b2));

        //s1: fill c1 c2
        b1 = _mm_load_pd(B + k*lda + j + 8);
        b2 = _mm_load_pd(B + k*lda + j + lda + 8);

        //Vectorize the inner loop
        c9 = _mm_add_pd(c9, _mm_mul_pd(a1, b1));
        c9 = _mm_add_pd(c9, _mm_mul_pd(a3, b2));

        c10 = _mm_add_pd(c10, _mm_mul_pd(a2, b1));
        c10 = _mm_add_pd(c10, _mm_mul_pd(a4, b2));
      }
      //{Write back Cij to slow memory}
  		_mm_store_pd(C+i*lda + j, c1);
  		_mm_store_pd(C+i*lda + j + lda, c2);
  		_mm_store_pd(C+i*lda + j + 2, c3);
  		_mm_store_pd(C+i*lda + j + lda + 2, c4);
  		_mm_store_pd(C+i*lda + j + 4, c5);
  		_mm_store_pd(C+i*lda + j + lda + 4, c6);
  		_mm_store_pd(C+i*lda + j + 6, c7);
  		_mm_store_pd(C+i*lda + j + lda + 6, c8);
  		_mm_store_pd(C+i*lda + j + 8, c9);
  		_mm_store_pd(C+i*lda + j + lda + 8, c10);
    }
    // int lack = N-j;
    // printf("lack:%d\n", lack);
    // if(lack == 4 || lack == 8){
      // for(int ex = j; ex < N; ex+=4){
      //   register __m128d c1 = _mm_load_pd(C+i*lda + ex);
      //   register __m128d c2 = _mm_load_pd(C+i*lda + ex + lda);
      //   register __m128d c3 = _mm_load_pd(C+i*lda + j + 2);
      //   register __m128d c4 = _mm_load_pd(C+i*lda + j + 2 + lda);
      //   for(int k=0; k<K; k+=2){
      //     //keep a1 a2 a3 a4 inside registers until finish using them
      //     //Notice we fill 128bits register with 64 double, twice, for A block
      //     register __m128d a1 = _mm_load1_pd(A+i*lda + k);
      //     register __m128d a2 = _mm_load1_pd(A+i*lda + k + lda);
      //     register __m128d a3 = _mm_load1_pd(A+i*lda + k + 1);
      //     register __m128d a4 = _mm_load1_pd(A+i*lda + k + lda + 1);


          //load matrix B streamly 5 times for 5 block

          //s1: fill c1 c2
    //       register __m128d b1 = _mm_load_pd(B + k*lda+ ex);
    //       register __m128d b2 = _mm_load_pd(B + k*lda + ex + lda);
    //
    //       //Vectorize the inner loop
    //       c1 = _mm_add_pd(c1, _mm_mul_pd(a1, b1));
    //       c1 = _mm_add_pd(c1, _mm_mul_pd(a3, b2));
    //
    //       c2 = _mm_add_pd(c2, _mm_mul_pd(a2, b1));
    // 			c2 = _mm_add_pd(c2, _mm_mul_pd(a4, b2));
    //
    //       //s1: fill c1 c2
    //       b1 = _mm_load_pd(B + k*lda+ ex);
    //       b2 = _mm_load_pd(B + k*lda + ex + lda);
    //
    //       //Vectorize the inner loop
    //       c3 = _mm_add_pd(c3, _mm_mul_pd(a1, b1));
    //       c3 = _mm_add_pd(c3, _mm_mul_pd(a3, b2));
    //
    //       c4 = _mm_add_pd(c4, _mm_mul_pd(a2, b1));
    //       c4 = _mm_add_pd(c4, _mm_mul_pd(a4, b2));
    //     }
    //     _mm_store_pd(C+i*lda + ex, c1);
    //     _mm_store_pd(C+i*lda + ex + lda, c2);
    //     _mm_store_pd(C+i*lda + j + 2, c3);
    //     _mm_store_pd(C+i*lda + j + lda + 2, c4);
    //   }
    // }
    // if(lack == 6){
    //   register __m128d c1 = _mm_load_pd(C+i*lda + j);
    //   register __m128d c2 = _mm_load_pd(C+i*lda + j + lda);
    //   register __m128d c3 = _mm_load_pd(C+i*lda + j + 2);
    // }
      for(int ex = j; ex < N; ex+=2){
        register __m128d c1 = _mm_load_pd(C+i*lda + ex);
        register __m128d c2 = _mm_load_pd(C+i*lda + ex + lda);
        for(int k=0; k<K; k+=2){
          //keep a1 a2 a3 a4 inside registers until finish using them
          //Notice we fill 128bits register with 64 double, twice, for A block
          register __m128d a1 = _mm_load1_pd(A+i*lda + k);
          register __m128d a2 = _mm_load1_pd(A+i*lda + k + lda);
          register __m128d a3 = _mm_load1_pd(A+i*lda + k + 1);
          register __m128d a4 = _mm_load1_pd(A+i*lda + k + lda + 1);


          //load matrix B streamly 5 times for 5 block

          //s1: fill c1 c2
          register __m128d b1 = _mm_load_pd(B + k*lda+ ex);
          register __m128d b2 = _mm_load_pd(B + k*lda + ex + lda);

          //Vectorize the inner loop
          c1 = _mm_add_pd(c1, _mm_mul_pd(a1, b1));
          c1 = _mm_add_pd(c1, _mm_mul_pd(a3, b2));

          c2 = _mm_add_pd(c2, _mm_mul_pd(a2, b1));
          c2 = _mm_add_pd(c2, _mm_mul_pd(a4, b2));
        }
        _mm_store_pd(C+i*lda + ex, c1);
        _mm_store_pd(C+i*lda + ex + lda, c2);
      }
  }
}
double * Matrix_Alignment(double* M, int oldSize, int newSize){
  __m128d zeros = _mm_setzero_pd();
  //Matrix should be aligned to 16-byte for SSE
  double *pM;/// = (double*) _aligned_malloc(newSize*newSize * sizeof(double), 16); // align to 16-byte for SSE
  posix_memalign((void**)&pM, 16, newSize*newSize * sizeof(double));
  int bonder = oldSize - oldSize%2;
    for(int i=0; i<oldSize; i++){
      for(int j=0; j<bonder; j+=2){
        __m128d tmp = _mm_loadu_pd(M + i*oldSize + j); //use loadu instead of load because loadu does no require alignment
        _mm_store_pd(pM + i*newSize + j, tmp);
      }
      if(bonder!=oldSize){
        pM[i*newSize + oldSize-1] = M[i*oldSize + oldSize-1];
        pM[i*newSize + oldSize] = 0;
      }

      for(int j=bonder+2; j<newSize; j+=2) {
        // pM[i*newSize + j] = 0;
        _mm_store_pd(pM+i*newSize+j, zeros);
      }
  }
	for(int i=oldSize; i<newSize; i++) {
		double* addr = pM + i * newSize;
		for(int j=0; j<newSize; j+=10) {
			_mm_store_pd(addr+j, zeros);
			_mm_store_pd(addr+j+2, zeros);
			_mm_store_pd(addr+j+4, zeros);
			_mm_store_pd(addr+j+6, zeros);
			_mm_store_pd(addr+j+8, zeros);
		}
    return pM;
  }
}
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */
void square_dgemm (int lda, double* A, double* B, double* C)
{
  //TODO: Optimize a parameter, maybe tilling size here?
  /* Do matrix padding first. */
  int lda_unpadded = lda;
  lda = lda + 10 - lda % 10;
  double* C_unpadded = C;

  __m128d zeros = _mm_setzero_pd();
  posix_memalign((void**)&C, 16, sizeof(double)*lda*lda);
  for(int i=0; i<lda*lda; i+=10) {
    _mm_store_pd(C+i,   zeros);
    _mm_store_pd(C+i+2, zeros);
    _mm_store_pd(C+i+4, zeros);
    _mm_store_pd(C+i+6, zeros);
    _mm_store_pd(C+i+8, zeros);
  }

  A = Matrix_Alignment(A, lda_unpadded, lda);
  B = Matrix_Alignment(B, lda_unpadded, lda);
  #ifdef TRANSPOSE
    for (int i = 0; i < lda; ++i)
      for (int j = i+1; j < lda; ++j) {
          double t = B[i*lda+j];
          B[i*lda+j] = B[j*lda+i];
          B[j*lda+i] = t;
    }
  #endif
  /* For each block-row of A */
  for (int i = 0; i < lda; i += BLOCK_SIZE_L2){
    /* For each block-column of B */
    /* Correct block dimensions if block "goes off edge of" the matrix */
    int M = min (BLOCK_SIZE_L2, lda-i);
    for (int j = 0; j < lda; j += BLOCK_SIZE_L2){
      /* Accumulate block dgemms into block of C */
      int N = min (BLOCK_SIZE_L2, lda-j);
      for (int k = 0; k < lda; k += BLOCK_SIZE_L2)
      {
	       int K = min (BLOCK_SIZE_L2, lda-k);
	       /* Perform individual block dgemm */
         //Notice this fetch a large portion of data into L2 cache first, and then fetch data from the L2 cache into the L1 cache to perform multiplication
	       do_block(lda, M, N, K, A + i*lda + k, B + k*lda + j, C + i*lda + j);
      }
    }
  }

// Copy back Matrix C:C->C_unpadded
  if(lda_unpadded%2 == 1){
    for(int i=0; i<lda_unpadded; i++){
      double * addrbase_u = C_unpadded+i*lda_unpadded;
      double * addrbase_p = C+i*lda;
      for(int j=0; j<lda_unpadded-1; j+=2){
        register __m128d tmp = _mm_load_pd(addrbase_p + j);
        _mm_storeu_pd(addrbase_u + j, tmp);
      }
      C_unpadded[(i+1)*lda_unpadded-1] = C[i*lda+lda_unpadded-1];
    }

  }
  else{
    for(int i=0; i<lda_unpadded; i++){
      double * addrbase_u = C_unpadded+i*lda_unpadded;
      double * addrbase_p = C+i*lda;
      for(int j=0; j<lda_unpadded; j+=2){
        register __m128d tmp = _mm_load_pd(addrbase_p+j);
        _mm_storeu_pd(addrbase_u+j, tmp);
      }
    }
  }
  //
free(C);
C = C_unpadded;
free(A);
free(B);
}
