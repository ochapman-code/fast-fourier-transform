#ifndef FFT_PARALLEL
#define FFT_PARALLEL

#include <time.h>
#include <stdio.h>
#include <math.h>       // log2
#include <complex.h>    // cexp
#include <stdlib.h>     // malloc
#include <omp.h>

#define ENABLE_PARALLEL 1

#define PI 3.14159265358979323846
typedef double complex cplx;


cplx* bit_reverse(cplx array[], int N){
    
    cplx temp;
    int n_bits = log2(N - 1) + 1;               // Number of bits

    #pragma omp parallel for if(ENABLE_PARALLEL)
    for (int i=0; i<N; i++){
        
        int r = 0;
        for (int j=0; j<n_bits; j++){
            int bit_j = (i >> j) & 1;           // Get the jth bit from the right
            r = r | (bit_j << (n_bits-j-1));    // Add the jth bit from the left
        }

        if (i < r) {
            temp = array[i];
            array[i] = array[r];
            array[r] = temp;
        }
    }

    return array;
}


cplx* W_fill(int N){

    cplx* W_array = (cplx*) malloc(N/2 * sizeof(cplx));
    #pragma omp parallel for if(ENABLE_PARALLEL)
    for (int i=0; i<N/2; i++)
        W_array[i] = cexp(- 2 * I * i * PI / N);

    return W_array;
}


void fft_iterate(cplx array[], cplx W_array[], int N){

    array = bit_reverse(array, N);

    for (int layer=1; layer<log2(N - 1) + 1; layer++) {
                    
        int group_size = 1 << layer;
        int group_num = N >> layer;
        int group_size_half = group_size >> 1;

        #pragma omp parallel for if(ENABLE_PARALLEL)
        for (int group=0; group<group_num; group++){

            int index_0_i = group * group_size;

            for (int index_2_i=0; index_2_i<group_size_half*group_num; index_2_i += group_num) {

                int index_1_i = index_0_i + group_size_half;

                cplx f = array[index_0_i];
                cplx g = array[index_1_i] * W_array[index_2_i];
                
                array[index_0_i] = f + g;
                array[index_1_i] = f - g;

                index_0_i ++;
            }
        }
    }
}


int main(){
    
    int N = pow(2,15);

    cplx* my_array = (cplx*) malloc(N * sizeof(cplx));      // Initialise array to be Fourier Transformed
    cplx* W = W_fill(N);                                    // Initialise W vector

    // Fill array
    #pragma omp parallel for if(ENABLE_PARALLEL)
    for (int i=0; i<N; i++){
        my_array[i] = i<N/2;
    }
            
    fft_iterate(my_array, W, N);
            
    // my_array now contains the fft of its original data        
    
    free(my_array);
    free(W);

    return 0;
}


#endif