#include "magma_svd.h"

#define min(a,b) (((a)<(b))?(a):(b))

void performSVD(long mValue, long nValue, float* originalMatrix, float* sMatrix; float* uMatrix, float* vtMatrix) {

  magma_init(); // initialize Magma
  real_Double_t gpu_time, cpu_time;

  // Matrix size
  magma_int_t m=mValue, n=nValue, n2=m*n, min_mn = min(m,n);

  if(argc == 2){
    m = atoi(argv[1]);
  }

  printf("m=%d,n=%d\n",m,n);
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
  magma_smalloc_cpu(&u, m*n); // host memory for u
  magma_smalloc_cpu(&s1 , min_mn ); // host memory for s1
  magma_smalloc_cpu(&s2 , min_mn ); // host memory for s2
  magma_smalloc_pinned(&r,n2 ); // host memory for r
  magma_int_t nb = magma_get_sgesvd_nb(m,n); // optim . block size

  magma_sgesvd(MagmaSomeVec, MagmaSomeVec, m, n,
                NULL, m, NULL, NULL, m, NULL, n, dummy, -1,
                &info );

  lwork = (magma_int_t) MAGMA_S_REAL( dummy[0] );
  magma_smalloc_pinned(& h_work , lwork ); // host mem . for h_work

  std::cout << "Loading matrix" << '\n';

  for (long i = 0; i < m*n; i++) {

    a[i] = originalMatrix[i];

  }

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
  magma_sgesvd(MagmaSomeVec,MagmaSomeVec,m,n,r,m,s1,u,m,vt,n,h_work,
  lwork,&info );

  std::cout << info << std::endl;

  gpu_time = magma_wtime() - gpu_time ;
  printf(" sgesvd gpu time: %7.5f\n", gpu_time); // Magma time

  for (long i = 0; i < min_mn; i++) {

    sMatrix[i] = s1[i];

  }

  cout<<"sMatrix dumped"<<endl;

  for(long i = 0; i < m; i++){

    for(int j = 0; j < numSigFig; j++) {

        uMatrixFile[i*n + j] = u[i*n + j];

    }

  }

  cout<<"uMatrix dumped"<<endl;

  for(long i = 0; i < n; i++) {

    for(int j = 0; j < numSigFig; j++) {

        vtMatrix[i*n + j] = vt[i*n + j];

    }

  }

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

}
