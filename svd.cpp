#include <stdio.h>
#include <cuda.h>
#include "magma_v2.h"
#include "magma_lapack.h"
#include <iostream>
#include <fstream>

#define min(a,b) (((a)<(b))?(a):(b))

using namespace std;

int main(int argc , char ** argv) {

  magma_init(); // initialize Magma
  real_Double_t gpu_time, cpu_time;

  // Matrix size
  magma_int_t m=4, n=2, n2=m*n, min_mn = min(m,n);
  float *a, *r; // a,r - mxn matrices
  float *u, *vt;// u - mxm matrix , vt - nxn matrix on the host
  float *s1 , *s2; // vectors of singular values
  magma_int_t info;
  magma_int_t ione = 1;
  float work[1], error = 1.; // used in difference computations
  float mone = -1.0, * h_work ; // h_work - workspace
  float dummy[1];
  magma_int_t lwork ; // workspace size
  magma_int_t ISEED[4] = {0 ,0 ,0 ,1}; // seed

  // Allocate host memory
  magma_smalloc_cpu(&a,n2 ); // host memory for a
  magma_smalloc_cpu(&vt, n*n); // host memory for vt
  magma_smalloc_cpu(&u, m*m); // host memory for u
  magma_smalloc_cpu(&s1 , min_mn ); // host memory for s1
  magma_smalloc_cpu(&s2 , min_mn ); // host memory for s2
  magma_smalloc_pinned(&r,n2 ); // host memory for r
  magma_int_t nb = magma_get_sgesvd_nb(m,n); // optim . block size

  magma_sgesvd(MagmaAllVec, MagmaAllVec, m, n,
                NULL, m, NULL, NULL, m, NULL, n, dummy, -1,
                &info );

  lwork = (magma_int_t) MAGMA_S_REAL( dummy[0] );
  magma_smalloc_pinned(& h_work , lwork ); // host mem . for h_work

  std::cout << "Loading matrix" << '\n';

  std::fstream firingMatrix("data/out/20170629_fish2_00/NNMF.nmf", std::ios_base::in);

  float nextFire;
  int indexOfA = 0;
  for(int i = 0; i < n2; ++i){
    a[i] = i;
    std::cout << i << '\n';
  }
  // while (firingMatrix >> nextFire) {
  //
  //     a[indexOfA] = nextFire;
  //     indexOfA++;
  //
  //     if (indexOfA % 1000000 == 0) {
  //
  //       std::cout << indexOfA << endl;
  //
  //     }
  //
  // }

  std::cout << "Done Loading" << '\n';

  // lapackf77_slarnv(&ione, ISEED, &n2, a);
  lapackf77_slacpy(MagmaFullStr, &m, &n, a, &m, r, &m);

  // MAGMA
  gpu_time = magma_wtime();

  // compute the singular value decomposition of r ( copy of a)
  // and optionally the left and right singular vectors :
  // r = u* sigma *vt; the diagonal elements of sigma (s1 array )
  // are the singular values of a in descending order
  // the first min (m,n) columns of u contain the left sing . vec .
  // the first min (m,n) columns of vt contain the right sing .vec .
  magma_sgesvd(MagmaAllVec,MagmaAllVec,m,n,r,m,s1,u,m,vt,n,h_work,
  lwork,&info );

  std::cout << info << std::endl;

  gpu_time = magma_wtime() - gpu_time ;
  printf(" sgesvd gpu time: %7.5f\n", gpu_time); // Magma time

  int numSigFig = 2;

  ofstream sMatrixFile("data/sMatrix.txt");
  if (sMatrixFile.is_open()) {
    for(long i = 0; i < 2; i++){
       sMatrixFile << s1[i] << "\n" ;
     }
   }
   sMatrixFile.close();
   cout<<"sMatrix dumped"<<endl;


  ofstream uMatrixFile("data/uMatrix.txt");
  if (uMatrixFile.is_open()) {
    for(long i = 0; i < m; i++){
      for(int j = 0; j < m; j++) {
          uMatrixFile << u[i*n + j] << "\n";
      }
     }
   }
   uMatrixFile.close();
   cout<<"uMatrix dumped"<<endl;

   ofstream vtMatrixFile("data/vtMatrix.txt");
   if (vtMatrixFile.is_open()) {
     for(long i = 0; i < n; i++){
       for(int j = 0; j < n; j++) {
           vtMatrixFile << vt[i*n + j] << "\n";
       }
      }
    }
    vtMatrixFile.close();
    cout<<"vtMatrix dumped"<<endl;


  // values
  // Free memory
  free(a); // free host memory
  free(vt); // free host memory
  free(s1); //free host memory
  free(s2); // free host memory
  free(u); // free host memory
  magma_free_pinned( h_work ); // free host memory
  magma_free_pinned(r); // free host memory
  magma_finalize( ); // finalize Magma
  return EXIT_SUCCESS ;

}
// sgesvd gpu time : 15.00651
// sgesvd cpu time : 115.81860
// difference : 5.943540e -07
