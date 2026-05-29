#ifndef _MY_FFT_H
#define _MY_FFT_H

#include <stdio.h>
#include <stdlib.h>

#include <complex.h>
#include <math.h>

#include <fftw3.h>

#include "my_thr_lib.h"

#define _VERBOSE

#define _SUPER_VERBOSE

#ifdef VARIABLE_BLOCK_SIDE
	extern int BLOCK_SIDE_SIZE;
#else
#ifndef BLOCK_SIDE_SIZE
	#define BLOCK_SIDE_SIZE 8
#endif
#endif

#define LIBNAME(x) my_fft_ ## x

typedef struct my_plan {
	void *plans_X, *plans_Y;
	int plans_X_n, plans_Y_n;
	int NX_size, NY_size, number_of_threads;
	fftw_complex *input, *output, *scratch_array;
	void (*exec)(struct my_plan *);
} *LIBNAME(plan);

void LIBNAME(init) (unsigned long int number_of_threads);
void LIBNAME(clear) (void);
LIBNAME(plan) LIBNAME(plan_dft_2d) (double complex *input, double complex *output, double complex *scratch_array, unsigned long int NX_size, unsigned long int NY_size, int DIR, unsigned FLAGS, unsigned long int number_of_threads);
LIBNAME(plan) LIBNAME(plan_dft_r2c_2d) (double *input, double complex *output, double complex *scratch_array, unsigned long int NX_size, unsigned long int NY_size, int DIR, unsigned FLAGS, unsigned long int number_of_threads);
LIBNAME(plan) LIBNAME(plan_dft_c2r_2d) (double complex *input, double *output, double complex *scratch_array, unsigned long int NX_size, unsigned long int NY_size, int DIR, unsigned FLAGS, unsigned long int number_of_threads);
void LIBNAME(destroy_plan) (LIBNAME(plan) plan);
void LIBNAME(execute) (LIBNAME(plan) plan);

void arrays_transpose (double complex *input, double complex *output, int row_elements_N, int column_elements_M);
void arrays_transpose_with_threads_in (double complex *input, double complex *output, int column_elements_M, int row_elements_N, int number_of_threads);
void arrays_transpose_with_threads_out (double complex *input, double complex *output, int column_elements_M, int row_elements_N, int number_of_threads);
//void unit_transpose (double complex *input, double complex *output, unsigned long int current_block, unsigned long int row_elements_N, unsigned long int column_elements_M, unsigned long int row_blocks_number, unsigned long int column_blocks_number, unsigned long int row_block_skip, unsigned long int column_block_skip);

#endif
