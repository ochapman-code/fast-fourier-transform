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
    // Reverse a binary array to begin FFT
    // e.g. [0,1,2,3] -> [0,2,1,3] as 01 and 10 are swapped
    
    cplx temp;
    int n_bits = log2(N - 1) + 1;               // Number of bits to be reversed

    #pragma omp parallel for if(ENABLE_PARALLEL)
    for (int i=0; i<N; i++){
        
        int r = 0;
        for (int j=0; j<n_bits; j++){
            int bit_j = (i >> j) & 1;           // Get the jth bit from the right
            r = r | (bit_j << (n_bits-j-1));    // Add the jth bit from the left
        }

        if (i < r) {                            // Only swap the pair once, take only lower value
            temp = array[i];
            array[i] = array[r];
            array[r] = temp;
        }
    }

    return array;
}


cplx* W_fill(int N){
    // Create a lookup table of exp(- 2 I pi / N) ^ i for 0 to N/2
    // This speeds computation by reducing unnecessary calculations

    cplx* W_array = (cplx*) malloc(N/2 * sizeof(cplx));
    #pragma omp parallel for if(ENABLE_PARALLEL)
    for (int i=0; i<N/2; i++)
        W_array[i] = cexp(- 2 * I * i * PI / N);

    return W_array;
}


void fft_iterate(cplx array[], cplx W_array[], int N){
    // Perform the FFT computation
    // Requires the input array, lookup table and size
    // Note: N must be a power of 2

    array = bit_reverse(array, N);

    for (int layer=1; layer<log2(N - 1) + 1; layer++) {
        // Each layer performs FFTs on groups in powers of 2 from 0 to n until the final FFT is performed
        // Computations in these groups are independent so can be parallelised
                    
        int group_size = 1 << layer;
        int group_num = N >> layer;
        int group_size_half = group_size >> 1;

        #pragma omp parallel for if(ENABLE_PARALLEL)            // Parallelise at group level
        for (int group=0; group<group_num; group++){

            int index_0_i = group * group_size;

            for (int index_2_i=0; index_2_i<group_size_half*group_num; index_2_i += group_num) {

                int index_1_i = index_0_i + group_size_half;

                cplx f = array[index_0_i];
                cplx g = array[index_1_i] * W_array[index_2_i];     // Computation reduced with just one multiplication on W_array
                
                array[index_0_i] = f + g;
                array[index_1_i] = f - g;

                index_0_i ++;
            }
        }
    }
}


int main(){             // This is inluded for a demonstration
                        // main can be removed to use this as a library
    // Steps:
    // 1) Create input array my_array
    // 2) Create lookup table W
    // 3) Perform FFT
    
    int N = pow(2,15);      // Must be a power of 2
                            // Pad with zeroes to next power of 2 if needed - it's more efficient

    cplx* my_array = (cplx*) malloc(N * sizeof(cplx));      // Initialise array to be Fourier Transformed
    cplx* W = W_fill(N);                                    // Initialise W vector
    
    #pragma omp parallel for if(ENABLE_PARALLEL)
    for (int i=0; i<N; i++){                            // Fill array with data to be transformed
        my_array[i] = i<N/2;
    }
            
    fft_iterate(my_array, W, N);    // Perform FFT
            
    // my_array now contains the FFT of its original array        
    
    free(my_array);                 // Free arrays from memory after use
    free(W);

    return 0;
}


#endif
