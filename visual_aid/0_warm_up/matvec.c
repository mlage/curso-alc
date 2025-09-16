#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<omp.h>

#ifdef DTYPE_F64
typedef double dtype_t;
#else
typedef float dtype_t;
#endif

// gcc -o exec matvec.c -O3 -fopenmp

void Ax_by_rows( dtype_t *b, dtype_t **A, dtype_t *x, uint32_t m, uint32_t n ){
	if ( b==NULL || A==NULL || x==NULL || m<1 || n<1 ) return;
	dtype_t q;
	for (uint32_t i=0; i<m; i++){
		q = 0.;
		for (uint32_t j=0; j<n; j++) q += A[i][j]*x[j];
		b[i] = q;
	}
}

void Ax_by_cols( dtype_t *b, dtype_t **A, dtype_t *x, uint32_t m, uint32_t n ){
	if ( b==NULL || A==NULL || x==NULL || m<1 || n<1 ) return;
	for (uint32_t i=0; i<m; i++) b[i]=0.;
	dtype_t q;
	for (uint32_t j=0; j<n; j++){
		q = x[j];
		for (uint32_t i=0; i<m; i++) b[i] += A[i][j]*q;
	}
}

dtype_t ** random_matrix(uint32_t m, uint32_t n){
	if ( m<1 || n<1 ) return NULL;
	void *ptr = malloc(sizeof(dtype_t)*m*n + sizeof(dtype_t*)*m);
	dtype_t **A = (dtype_t **)ptr;
	A[0] = (dtype_t *)&A[m];
	for (uint32_t i=1; i<m; i++) A[i] = &(A[0][i*n]);
	for (uint32_t i=0; i<m; i++){
		for (uint32_t j=0; j<n; j++)
			A[i][j] = ((dtype_t) rand()) / RAND_MAX;
	}
  return A;
}

void free_matrix( dtype_t **A ){
	if (A); free(A);
}

int main(int argc, char *argv[]){
	uint32_t m=100, n=100;
	if (argc>1) m = atoi(argv[1]);
	if (argc>2) n = atoi(argv[2]);

	// set random seed with time
  srand(omp_get_wtime());

	dtype_t **A = random_matrix(m,n);
	if ( A == NULL ) { printf("Not enough memory.\n"); return 1; }

	dtype_t **X = random_matrix(n,1);
	if ( X == NULL ) { free_matrix(A); printf("Not enough memory.\n"); return 1; }

	dtype_t **B = random_matrix(m,1);
	if ( B == NULL ) { free_matrix(X); free_matrix(A); printf("Not enough memory.\n"); return 1; }

	double start, stop;

	start = omp_get_wtime();
	Ax_by_rows(B[0],A,X[0],m,n);
	stop = omp_get_wtime();
	printf("Ax_by_rows: %.4es\n",stop-start);
	
	start = omp_get_wtime();
	Ax_by_cols(B[0],A,X[0],m,n);
	stop = omp_get_wtime();
	printf("Ax_by_cols: %.4es\n",stop-start);

	free_matrix(A);
	free_matrix(X);
	free_matrix(B);
	return 0;
}
