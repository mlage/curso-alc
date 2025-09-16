/*
Para compilar biblioteca:

gcc -shared -fPIC -o img_compression.so img_compression.c -fopenmp -O3
*/


#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>

#ifdef DTYPE_F64
typedef double dtype_t;
#else
typedef float dtype_t;
#endif


//////////// FUNCTION SIGNATURES ////////////
void matrix_vector_product( dtype_t *q, dtype_t *A, dtype_t *v, uint32_t m, uint32_t n );
void compress( char *filename, dtype_t *img, uint32_t rows, uint32_t cols, dtype_t threshold );
void decompress( dtype_t *img, char *filename );
/////////////////////////////////////////////

//----------------------------------------------------------------------------------------------
// q = Av
void matrix_vector_product( dtype_t *q, dtype_t *A, dtype_t *v, uint32_t m, uint32_t n ){
  dtype_t aux;
  dtype_t *rowA;
  #pragma omp parallel for if (m>10000) private(aux,rowA)
  for (uint32_t i=0; i<m; i++){
    aux = 0.0;
    rowA = &A[i*n];
    for (uint32_t j=0; j<n; j++) aux += rowA[j]*v[j];
    q[i] = aux;
  }
}
//----------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------
void compress( char *filename, dtype_t *img, uint32_t rows, uint32_t cols, dtype_t threshold ){
  if (filename == NULL || img == NULL || rows<1 || cols < 1) return;
  
  char wavelet_inverse_file[] = "wavelet_basis_64_inv.bin";
  FILE *fptr = fopen(wavelet_inverse_file,"rb");
  if (!fptr){
    printf("ERROR: failed to open %s. Terminated.\n",wavelet_inverse_file);
    return;
  }
  dtype_t *wavelets = (dtype_t *)malloc(sizeof(dtype_t)*64*64);
  size_t out_fread = fread(wavelets,sizeof(dtype_t),64*64,fptr);
  if (out_fread != 64*64){
    printf("ERROR: failed to read %s. Terminated.\n",wavelet_inverse_file);
    fclose(fptr);
    free(wavelets);
    return;
  }
  fclose(fptr);
  
  uint32_t rows_padded = rows + (rows%8 ? (8-rows%8) : 0);
  uint32_t cols_padded = cols + (cols%8 ? (8-cols%8) : 0);
  
  uint32_t pixels = rows*cols;
  uint32_t pixels_padded = rows_padded*cols_padded;
  
  uint32_t blocks = rows_padded*cols_padded/64;
  
  uint64_t *basis_uses = (uint64_t *)malloc(blocks*sizeof(uint64_t));
  
  dtype_t *weights = (dtype_t *)malloc(sizeof(dtype_t)*blocks*64);
  
  dtype_t x[64] = {0};
  dtype_t q[64] = {0};
  #pragma omp parallel for private(x,q)
  for (uint32_t block=0; block<blocks; block++){
    uint32_t bi = block/(cols_padded/8);
    uint32_t bj = block%(cols_padded/8);
    
    for ( uint32_t k = 0; k<64; k++) x[k]=0;
    
    uint32_t li=0, lj=0;
    for ( uint32_t i = bi*8; i<(bi+1)*8 && i<rows; i++){
      for ( uint32_t j = bj*8; j<(bj+1)*8 && j<cols; j++) {
        x[li*8+lj] = img[i*cols+j];
        lj++;
      }
      li++;
      lj=0;
    }
    
    matrix_vector_product(q,wavelets,x,64,64);
    
    uint64_t use = 0;
    for ( uint32_t k=0; k<64; k++ ){
      if (q[k] >= threshold || q[k] <= -threshold){
      //if (fabs(q[k]) >= threshold){
        use += 1<<k;
        weights[block*64+k] = q[k];
      }
    }
    basis_uses[block] = use;
  }
  
  dtype_t *weights_copy = (dtype_t *)malloc(sizeof(dtype_t)*blocks*64);
  uint32_t weight_count=0;
  for (uint32_t i=0; i<blocks; i++){
    uint64_t use = basis_uses[i];
    for (uint8_t j=0; j<64; j++){
      if ( (use>>j)&1 ){
        weights_copy[weight_count++] = weights[i*64+j];
      }
    }
  }
  free(weights);
  weights = weights_copy;
  weights_copy = NULL;
  
  fptr = fopen(filename,"wb");
  if (!fptr){
    printf("ERROR: failed to open %s. Terminated.\n",filename);
    free(wavelets);
    free(basis_uses);
    free(weights);
    return;
  }
  fwrite(&rows,sizeof(uint32_t),1,fptr);
  fwrite(&cols,sizeof(uint32_t),1,fptr);
  fwrite(&weight_count,sizeof(uint32_t),1,fptr);
  fwrite(basis_uses,sizeof(uint64_t),blocks,fptr);
  fwrite(weights,sizeof(dtype_t),weight_count,fptr);
  fclose(fptr);
  
  free(wavelets);
  free(basis_uses);
  free(weights);
  return;
}
//----------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------
void decompress( dtype_t *img, char *filename ){
  if ( img == NULL || filename == NULL ) return;
  
  char wavelet_file[] = "wavelet_basis_64.bin";
  FILE *fptr = fopen(wavelet_file,"rb");
  if (!fptr){
    printf("ERROR: failed to open %s. Terminated.\n",wavelet_file);
    return;
  }
  dtype_t *wavelets = (dtype_t *)malloc(sizeof(dtype_t)*64*64);
  size_t out_fread = fread(wavelets,sizeof(dtype_t),64*64,fptr);
  if (out_fread != 64*64){
    printf("ERROR: failed to read %s. Terminated.\n",wavelet_file);
    fclose(fptr);
    free(wavelets);
    return;
  }
  fclose(fptr);
  
  fptr = fopen(filename,"rb");
  if (!fptr){
    printf("ERROR: failed to open %s. Terminated.\n",filename);
    free(wavelets);
    return;
  }
  uint32_t dims[3] = {0}; //rows, cols, weight_count;
  if ( fread(dims,sizeof(uint32_t),3,fptr) != 3 ){
    printf("ERROR: failed to read %s. Terminated.\n",filename);
    fclose(fptr);
    free(wavelets);
    return;
  }
  uint32_t rows=dims[0], cols=dims[1], weight_count=dims[2];
  
  uint32_t rows_padded = rows + (rows%8 ? (8-rows%8) : 0);
  uint32_t cols_padded = cols + (cols%8 ? (8-cols%8) : 0);
  
  uint32_t pixels = rows*cols;
  uint32_t pixels_padded = rows_padded*cols_padded;
  
  uint32_t blocks = rows_padded*cols_padded/64;
  
  uint64_t *basis_uses = (uint64_t *)malloc(blocks*sizeof(uint64_t));
  
  dtype_t *weights = (dtype_t *)malloc(sizeof(dtype_t)*weight_count);
  
  if ( fread(basis_uses,sizeof(uint64_t),blocks,fptr) != blocks ){
    printf("ERROR: failed to read %s. Terminated.\n",filename);
    fclose(fptr);
    free(basis_uses);
    free(weights);
    free(wavelets);
    return;
  }
  if ( fread(weights,sizeof(dtype_t),weight_count,fptr) != weight_count ){
    printf("ERROR: failed to read %s. Terminated.\n",filename);
    fclose(fptr);
    free(basis_uses);
    free(weights);
    free(wavelets);
    return;
  }
  fclose(fptr);
  
  dtype_t x[64] = {0};
  dtype_t q[64] = {0};
  uint32_t counter=0;
  for (uint32_t block=0; block<blocks; block++){
    uint64_t use = basis_uses[block];
    for ( uint32_t k=0; k<64; k++ ){
      if ( (use>>k)&1 ){
        q[k] = weights[counter++];
      } else q[k] =0.;
    }
    matrix_vector_product(x,wavelets,q,64,64);
    uint32_t bi = block/(cols_padded/8);
    uint32_t bj = block%(cols_padded/8);
    
    uint32_t li=0, lj=0;
    for ( uint32_t i = bi*8; i<(bi+1)*8 && i<rows; i++){
      for ( uint32_t j = bj*8; j<(bj+1)*8 && j<cols; j++) {
        img[i*cols+j] = x[li*8+lj];
        lj++;
      }
      li++;
      lj=0;
    }
  }
  
  free(basis_uses);
  free(weights);
  free(wavelets);
  return;
}
//----------------------------------------------------------------------------------------------
