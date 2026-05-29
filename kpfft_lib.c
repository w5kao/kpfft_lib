
#include "kpfft_lib.h"
#include <assert.h>
#include <string.h>
#include <unistd.h>

// Use inplace transpose where possible
#define INPLACE_TR

#ifdef VARIABLE_BLOCK_SIDE
int BLOCK_SIDE_SIZE = 8;
#endif

unsigned long int each_thread_elements, last_thread_elements;


struct DFT_thread_input {
	void *plan;
	unsigned long int elements_number;
	fftw_complex *input;
	fftw_complex *output;
	size_t sz;
};

struct arrays_transpose_thread_input {
	double complex *input, *output;
	int elements_number;
	int r_size, c_size; // Row size, column size
	int r_blocks, c_blocks;
	int r_start, c_start;

	unsigned long int index, row_elements_N, column_elements_M, row_blocks_number, column_blocks_number, row_block_skip, column_block_skip;
};

struct plans_init_thread_input {
	unsigned long int index, elements_number;
};

struct DFT_thread_input *thr_array_DFT;
struct arrays_transpose_thread_input *thr_arrays_transpose;

void LIBNAME(init) (unsigned long int number_of_threads) {

	if (number_of_threads > 0) {
		my_thr_pool_init (number_of_threads);
		thr_array_DFT = (struct DFT_thread_input *) malloc (number_of_threads*sizeof(struct DFT_thread_input));
		thr_arrays_transpose = (struct arrays_transpose_thread_input *) malloc (number_of_threads*sizeof(struct arrays_transpose_thread_input));
	}
}

void LIBNAME(clear) (void) {
	my_thr_pool_clear();
}

/* Functions to call on execute */
static void _exec_dft_0(LIBNAME(plan));
static void _exec_dft(LIBNAME(plan));
static void _exec_dft_tr_inplace(LIBNAME(plan));
static void _exec_dft_inplace(LIBNAME(plan));
static void _exec_c2r(LIBNAME(plan));
static void _exec_r2c(LIBNAME(plan));

/* Helpers to generate FFTW plans */
static void _create_plans_list(LIBNAME(plan) return_plan, int DIR, unsigned FLAGS, int n, fftw_plan *dst, int sz, double complex *from, double complex *to, int tnum);
static void _create_plans_list_r2c(LIBNAME(plan) return_plan, int DIR, unsigned FLAGS, int n, fftw_plan *dst, int sz, double *from, double complex *to, int tnum);
static void _create_plans_list_c2r(LIBNAME(plan) return_plan, int DIR, unsigned FLAGS, int n, fftw_plan *dst, int sz, double complex *from, double *to, int tnum);

static LIBNAME(plan) _base_plan(double complex *input, double complex *output, double complex *scratch_array, unsigned long int NX_size, unsigned long int NY_size, unsigned long int number_of_threads) {

	LIBNAME(plan) return_plan;

	assert(input != NULL);
	assert(output != NULL);

	int tnum = number_of_threads;
	if (tnum == 0) tnum = 1; // Generate single plan for all

	return_plan = (LIBNAME(plan)) malloc (sizeof(struct my_plan)); assert(return_plan != NULL);
	return_plan->NX_size = NX_size;
	return_plan->NY_size = NY_size;
	return_plan->input = input;
	return_plan->output = output;
	return_plan->scratch_array = scratch_array;
	return_plan->number_of_threads = tnum;

	return_plan->plans_Y = malloc(tnum*sizeof(fftw_plan)); assert(return_plan->plans_Y != NULL);
	return_plan->plans_X = malloc(tnum*sizeof(fftw_plan)); assert(return_plan->plans_X != NULL);

	return return_plan;
}

/* FLAGS = FFTW_EXHAUSTIVE | FFTW_DESTROY_INPUT, DIR = +1 OR -1 */
LIBNAME(plan) LIBNAME(plan_dft_2d) (double complex *input, double complex *output, double complex *scratch_array, unsigned long int NX_size, unsigned long int NY_size, int DIR, unsigned FLAGS, unsigned long int number_of_threads) {
	unsigned long int i;
	double complex *temp_u, *temp_u_omega;

	LIBNAME(plan) return_plan = _base_plan(input, output, scratch_array, NX_size, NY_size, number_of_threads);

	int tnum = return_plan->number_of_threads;

#ifdef INPLACE_TR
	if ((return_plan->input == return_plan->output) && (NX_size == NY_size) && number_of_threads > 0) {
		_create_plans_list(return_plan, DIR, FLAGS, NX_size, return_plan->plans_Y, NY_size, return_plan->input, return_plan->input, tnum);
		_create_plans_list(return_plan, DIR, FLAGS, NY_size, return_plan->plans_X, NX_size, return_plan->input, return_plan->input, tnum);
		return_plan->exec = _exec_dft_inplace;
	} else if (NX_size == NY_size  && number_of_threads > 0) {
		_create_plans_list(return_plan, DIR, FLAGS, NX_size, return_plan->plans_Y, NY_size, return_plan->input, return_plan->output, tnum);
		_create_plans_list(return_plan, DIR, FLAGS, NY_size, return_plan->plans_X, NX_size, return_plan->output, return_plan->output, tnum);
		return_plan->exec = _exec_dft_tr_inplace;
	} else 
#endif
	if (return_plan->input == return_plan->output) {
		fprintf(stderr, "Inplace DFT not supported.\n");
		exit(2);
	} else {
		_create_plans_list(return_plan, DIR, FLAGS, NX_size, return_plan->plans_Y, NY_size, return_plan->input, return_plan->scratch_array, tnum);
		_create_plans_list(return_plan, DIR, FLAGS, NY_size, return_plan->plans_X, NX_size, return_plan->output, return_plan->scratch_array, tnum);
		return_plan->exec = (number_of_threads == 0) ? _exec_dft_0 : _exec_dft;
	}

	return (LIBNAME(plan)) return_plan;
}

LIBNAME(plan) LIBNAME(plan_dft_r2c_2d) (double *input, double complex *output, double complex *scratch_array, unsigned long int NX_size, unsigned long int NY_size, int DIR, unsigned FLAGS, unsigned long int number_of_threads) {
	unsigned long int i;
	double complex *temp_u, *temp_u_omega;

	LIBNAME(plan) return_plan = _base_plan((double complex*)input, output, scratch_array, NX_size, NY_size, number_of_threads);

	int tnum = return_plan->number_of_threads;

	_create_plans_list_r2c(return_plan, DIR, FLAGS, NX_size, return_plan->plans_Y, NY_size, (double*)return_plan->input, return_plan->scratch_array, tnum);
	_create_plans_list(return_plan, DIR, FLAGS, NY_size/2+1, return_plan->plans_X, NX_size, return_plan->output, return_plan->scratch_array, tnum);
	return_plan->exec = _exec_r2c;

	return (LIBNAME(plan)) return_plan;
}

LIBNAME(plan) LIBNAME(plan_dft_c2r_2d) (double complex *input, double *output, double complex *scratch_array, unsigned long int NX_size, unsigned long int NY_size, int DIR, unsigned FLAGS, unsigned long int number_of_threads) {
	unsigned long int i;
	double complex *temp_u, *temp_u_omega;

	LIBNAME(plan) return_plan = _base_plan(input, (double complex*)output, scratch_array, NX_size, NY_size, number_of_threads);

	int tnum = return_plan->number_of_threads;

	_create_plans_list(return_plan, DIR, FLAGS, NY_size/2+1, return_plan->plans_X, NX_size, return_plan->scratch_array, return_plan->input, tnum);
	_create_plans_list_c2r(return_plan, DIR, FLAGS, NX_size, return_plan->plans_Y, NY_size, return_plan->scratch_array, (double*)return_plan->output, tnum);
	return_plan->exec = _exec_c2r;

	return (LIBNAME(plan)) return_plan;
}

void LIBNAME(destroy_plan) (LIBNAME(plan) plan) {
	int i;
	int tnum = plan->number_of_threads;
	fftw_plan *temp_plan;
	
	temp_plan = plan->plans_X;
	for (i=0; i < tnum; ++i) {
		fftw_destroy_plan((*temp_plan));
		++temp_plan;
	}

	temp_plan = plan->plans_Y;
	for (i=0; i < tnum; ++i) {
		fftw_destroy_plan((*temp_plan));
		++temp_plan;
	}

	free (plan);
}

void my_fft_execute (LIBNAME(plan) plan) {
	plan->exec(plan);
}


static void _create_plans_list(LIBNAME(plan) return_plan, int DIR, unsigned FLAGS, int n, fftw_plan *dst, int sz, double complex *from, double complex *to, int tnum) {

	(void)return_plan;

	int per_thread = (n+tnum-1)/tnum;
	const int shape = sz;

	if (per_thread < tnum) -- per_thread;

	for (int i = 0; i < tnum; ++i) {
	
		int work = (i == tnum-1) ? n-(tnum-1)*per_thread : per_thread;
		dst[i] = fftw_plan_many_dft (
			1, & shape, work,
			from, & shape, 1, sz,
			to, NULL, 1, sz,
			DIR, FLAGS
		); assert(dst[i] != NULL);

		from += work*sz;
		to += work*sz;
	}
}


static void _create_plans_list_r2c(LIBNAME(plan) return_plan, int DIR, unsigned FLAGS, int n, fftw_plan *dst, int sz, double *from, double complex *to, int tnum) {

	(void)return_plan;
	(void)DIR;

	size_t dst_stride = sz/2+1;
	int per_thread = (n+tnum-1)/tnum;
	const int shape = sz;

	if (per_thread < tnum) -- per_thread;

	for (int i = 0; i < tnum; ++i) {
		int work = (i == tnum-1) ? n-(tnum-1)*per_thread : per_thread;
		dst[i] = fftw_plan_many_dft_r2c (
			1, & shape, work,
			from, NULL, 1, sz,
			to, NULL, 1, dst_stride,
			FLAGS
		);
		from += work*sz;
		to += work*dst_stride;
	}
}

static void _create_plans_list_c2r(LIBNAME(plan) return_plan, int DIR, unsigned FLAGS, int n, fftw_plan *dst, int sz, double complex *from, double *to, int tnum) {

	(void)return_plan;
	(void)DIR;

	size_t src_stride = sz/2+1;
	int per_thread = (n+tnum-1)/tnum;
	const int shape =  sz;

	if (per_thread < tnum) -- per_thread;

	for (int i = 0; i < tnum; ++i) {
		int work = (i == tnum-1) ? n-(tnum-1)*per_thread : per_thread;
		dst[i] = fftw_plan_many_dft_c2r(
			1, & shape, work,
			from, NULL, 1, src_stride,
			to, NULL, 1, sz,
			FLAGS
		);
		from += work*src_stride;
		to += work*sz;
	}
}

static void _assign_plans(LIBNAME(plan) plan, void *plans, fftw_complex *input, fftw_complex *output, size_t sz) {

	(void)input;

	int tnum = plan->number_of_threads;

	fftw_plan *temp_plan = plans;
	for (int i = 0; i < tnum; ++i) {
		thr_array_DFT [i].plan = temp_plan;
		thr_array_DFT [i].elements_number = each_thread_elements;
		thr_array_DFT [i].output = output;
		thr_array_DFT [i].sz = sz;
		my_thr_data_assign (i,  (void *) &thr_array_DFT[i]);
		temp_plan ++; //= each_thread_elements;
	}
	thr_array_DFT [tnum-1].elements_number = last_thread_elements;
}

void DFT_for_arrays_thr (void * input_thr) {

	struct DFT_thread_input *input;
	fftw_plan *temp_plan;
	unsigned long int i;

	input = (struct DFT_thread_input *) input_thr;
	temp_plan = input->plan;
	fftw_execute(*temp_plan);
}

// "Plain" version for testing
static void _exec_dft_0(LIBNAME(plan) plan) {

	fftw_plan *temp_plan = plan->plans_Y;

	fftw_execute(*temp_plan);

	arrays_transpose (plan->scratch_array, plan->output, plan->NY_size, plan->NX_size);

	temp_plan = plan->plans_X;
	fftw_execute(*temp_plan);

	arrays_transpose (plan->scratch_array, plan->output, plan->NX_size, plan->NY_size);
}


void test_fft(LIBNAME(plan) plan) {

	int tnum = plan->number_of_threads;

	each_thread_elements = (unsigned long int) (0.5 + (1.0*(plan->NX_size))/tnum);
	last_thread_elements = plan->NX_size - each_thread_elements*(tnum-1);

	_assign_plans(plan, plan->plans_Y, plan->input, plan->scratch_array, plan->NY_size);
	my_thr_manager (DFT_for_arrays_thr);

}

static void _exec_dft(LIBNAME(plan) plan) {

	int tnum = plan->number_of_threads;

	each_thread_elements = (unsigned long int) (0.5 + (1.0*(plan->NX_size))/tnum);
	last_thread_elements = plan->NX_size - each_thread_elements*(tnum-1);

	_assign_plans(plan, plan->plans_Y, plan->input, plan->scratch_array, plan->NY_size);
	my_thr_manager (DFT_for_arrays_thr);

//		dump2("fft", plan->scratch_array, plan->NY_size, plan->NX_size);
//		exit(1);

	arrays_transpose_with_threads_out (plan->scratch_array, plan->output, plan->NY_size, plan->NX_size, tnum);

//	x_arrays_transpose_with_threads (plan->scratch_array, plan->output, plan->NX_size, plan->NY_size, tnum);

	each_thread_elements = (unsigned long int) (0.5 + (1.0*(plan->NY_size))/tnum);
	last_thread_elements = plan->NY_size - each_thread_elements*(tnum-1);

	_assign_plans(plan, plan->plans_X, plan->output, plan->scratch_array, plan->NX_size);
	my_thr_manager (DFT_for_arrays_thr);

	arrays_transpose_with_threads_out (plan->scratch_array, plan->output, plan->NX_size, plan->NY_size, tnum);

//	x_arrays_transpose_with_threads (plan->scratch_array, plan->output, plan->NY_size, plan->NX_size, tnum);
}

/**
 * Perform DFT with inplace transpose
 * First dimension FFTW DFT is out-of-place, second in-place
 * Scratch array not required
 */
static void _exec_dft_tr_inplace(LIBNAME(plan) plan) {

	int tnum = plan->number_of_threads;

	each_thread_elements = (unsigned long int) (0.5 + (1.0*(plan->NX_size))/(tnum));
	last_thread_elements = plan->NX_size - each_thread_elements*(tnum-1);

	_assign_plans(plan, plan->plans_Y, plan->input, plan->output, plan->NY_size);
	my_thr_manager (DFT_for_arrays_thr);

	arrays_transpose_with_threads_in (plan->output, plan->output, plan->NX_size, plan->NY_size, tnum);

	each_thread_elements = (unsigned long int) (0.5 + (1.0*(plan->NY_size))/tnum);
	last_thread_elements = plan->NY_size - each_thread_elements*(tnum-1);

	_assign_plans(plan, plan->plans_X, plan->output, plan->output, plan->NX_size);
	my_thr_manager (DFT_for_arrays_thr);

	arrays_transpose_with_threads_in (plan->output, plan->output, plan->NY_size, plan->NX_size, tnum);
}

/**
 * Inplace DFT, both 1d DFT and transpose
 * Scratch array not required
 */
static void _exec_dft_inplace(LIBNAME(plan) plan) {

	int tnum = plan->number_of_threads;

	each_thread_elements = (unsigned long int) (0.5 + (1.0*(plan->NX_size))/(tnum));
	last_thread_elements = plan->NX_size - each_thread_elements*(tnum-1);

	_assign_plans(plan, plan->plans_Y, plan->input, plan->input, plan->NY_size);
	my_thr_manager (DFT_for_arrays_thr);

	arrays_transpose_with_threads_in (plan->input, plan->input, plan->NX_size, plan->NY_size, tnum);

	each_thread_elements = (unsigned long int) (0.5 + (1.0*(plan->NY_size))/tnum);
	last_thread_elements = plan->NY_size - each_thread_elements*(tnum-1);

	_assign_plans(plan, plan->plans_X, plan->input, plan->input, plan->NX_size);
	my_thr_manager (DFT_for_arrays_thr);

	arrays_transpose_with_threads_in (plan->input, plan->input, plan->NY_size, plan->NX_size, tnum);
}


static void _exec_r2c(LIBNAME(plan) plan) {

	int tnum = plan->number_of_threads;
	int nx = plan->NX_size;
	int ny2 = plan->NY_size/2+1;


	each_thread_elements = (nx + tnum-1)/tnum;
	last_thread_elements = nx - each_thread_elements*(tnum-1);

	_assign_plans(plan, plan->plans_Y, plan->input, plan->scratch_array, plan->NY_size);
	my_thr_manager (DFT_for_arrays_thr);

	arrays_transpose_with_threads_out (plan->scratch_array, plan->output, ny2, plan->NX_size, tnum);

	each_thread_elements = (ny2 + tnum-1)/tnum;
	last_thread_elements = ny2 - each_thread_elements*(tnum-1);

	_assign_plans(plan, plan->plans_X, plan->output, plan->scratch_array, nx);
	my_thr_manager (DFT_for_arrays_thr);

	arrays_transpose_with_threads_out (plan->scratch_array, plan->output, nx, ny2, tnum);

//	x_arrays_transpose_with_threads (plan->scratch_array, plan->output, plan->NY_size, plan->NX_size, tnum);
}

static void _exec_c2r(LIBNAME(plan) plan) {

	int tnum = plan->number_of_threads;
	int nx = plan->NX_size;
	int ny2 = plan->NY_size/2+1;

	arrays_transpose_with_threads_out (plan->input, plan->scratch_array, ny2, plan->NX_size, tnum);

	each_thread_elements = (ny2 + tnum-1)/tnum;
	last_thread_elements = ny2 - each_thread_elements*(tnum-1);

	_assign_plans(plan, plan->plans_X, plan->scratch_array, plan->input, nx);
	my_thr_manager (DFT_for_arrays_thr);

	arrays_transpose_with_threads_out (plan->input, plan->scratch_array, nx, ny2, tnum);

	each_thread_elements = (nx + tnum-1)/tnum;
	last_thread_elements = nx - each_thread_elements*(tnum-1);

	_assign_plans(plan, plan->plans_Y, plan->input, plan->scratch_array, plan->NY_size);
	my_thr_manager (DFT_for_arrays_thr);

}



static inline void unit_transpose (double complex *in, double complex *out, int in_stride, int out_stride, int r_len) {

	int i, j;

	for (j = 0; j < BLOCK_SIDE_SIZE; ++j) {
		for (i = 0; i < r_len; ++i) {
			out[i*out_stride] = in[i];
		}
		in += in_stride;
		out ++;
	}
}

void arrays_transpose (double complex *input, double complex *output, int row_size, int col_size) {

	int row_blocks_number, column_blocks_number, total_number_of_blocks, i,j;
	double complex *block_ptr_in, *block_ptr_out;

	row_blocks_number = row_size/BLOCK_SIDE_SIZE;
	column_blocks_number = col_size/BLOCK_SIDE_SIZE;

	double complex *src = input;
	double complex *dst = output;

	for (int j = 0; j < column_blocks_number; ++ j) {

		src = input + j*row_size*BLOCK_SIDE_SIZE;
		dst = output + j*BLOCK_SIDE_SIZE;

		for (int i = 0; i < row_blocks_number; ++ i) {
			unit_transpose(src, dst, row_size, col_size, BLOCK_SIDE_SIZE);
			src += BLOCK_SIDE_SIZE;
			dst += BLOCK_SIDE_SIZE*col_size;
		}

		// Transpose rest of row less than block size
		if(row_size - row_blocks_number*BLOCK_SIDE_SIZE > 0)
			unit_transpose (src, dst, row_size, col_size, row_size - row_blocks_number*BLOCK_SIDE_SIZE);
	}

	// Transpose rest of rows less than block size
	if (col_size - column_blocks_number*BLOCK_SIDE_SIZE > 0) {

		src = input + column_blocks_number*BLOCK_SIDE_SIZE*row_size;
		dst = output + column_blocks_number*BLOCK_SIDE_SIZE;

		for (int j = 0; j < col_size - column_blocks_number*BLOCK_SIDE_SIZE; ++ j) {
			for (int i = 0; i < row_size; ++ i) {
				dst[i*col_size] = src[i];
			}
			src += row_size;
			dst ++;
		}
	}
}

static void arrays_transpose_thr (void * _data);

void arrays_transpose_with_threads_out (double complex *input, double complex *output, int row_size, int col_size, int number_of_threads) {

	unsigned long int row_blocks_number, column_blocks_number, total_number_of_blocks;
	int i,j;
	double complex *block_ptr_in, *block_ptr_out;

	row_blocks_number = row_size/(BLOCK_SIDE_SIZE);
	column_blocks_number = col_size/(BLOCK_SIDE_SIZE);

	total_number_of_blocks = row_blocks_number*column_blocks_number;

	each_thread_elements = (unsigned long int) (0.5 + (1.0*total_number_of_blocks)/(number_of_threads));
	last_thread_elements = total_number_of_blocks - each_thread_elements*(number_of_threads-1);

	for (i = 0; i < number_of_threads; ++i) {
		thr_arrays_transpose [i].input = input;
		thr_arrays_transpose [i].output = output;
		thr_arrays_transpose [i].elements_number = each_thread_elements;
		thr_arrays_transpose [i].r_size = row_size;
		thr_arrays_transpose [i].c_size = col_size;
		thr_arrays_transpose [i].r_blocks = row_blocks_number;
		thr_arrays_transpose [i].c_blocks = column_blocks_number;
		thr_arrays_transpose [i].r_start = (i*each_thread_elements) % row_blocks_number;
		thr_arrays_transpose [i].c_start = (i*each_thread_elements)/row_blocks_number;

		my_thr_data_assign (i,  (void *) &thr_arrays_transpose[i]);
	}

	thr_arrays_transpose [(number_of_threads-1)].elements_number = last_thread_elements;
	my_thr_manager (arrays_transpose_thr);
}

static void arrays_transpose_thr (void * _data) {

	struct arrays_transpose_thread_input *data = (struct arrays_transpose_thread_input *) _data;
	unsigned long int i, final_index;

	int r_start = data->r_start;
	int c_start = data->c_start;
	int row_size = data->r_size;
	int col_size = data->c_size;

	double complex *src = data->input  + c_start*row_size*BLOCK_SIDE_SIZE + r_start*BLOCK_SIDE_SIZE;
	double complex *dst = data->output + r_start*col_size*BLOCK_SIDE_SIZE + c_start*BLOCK_SIDE_SIZE;

	int num = data->elements_number;

	while (num > 0) {

		unit_transpose (src, dst, row_size, col_size, BLOCK_SIDE_SIZE);

		-- num;
		++ r_start;
		
		src += BLOCK_SIDE_SIZE;
		dst += BLOCK_SIDE_SIZE*col_size;

		if (r_start >= data->r_blocks) {
		
			// Transpose rest of row less than block size
			unit_transpose (src, dst, row_size, col_size, row_size - data->r_blocks*BLOCK_SIDE_SIZE);

			++ c_start;
			r_start = 0;

			// Recalculate for next blocks row
			src = data->input  + c_start*row_size*BLOCK_SIDE_SIZE;
			dst = data->output + c_start*BLOCK_SIDE_SIZE;
		}
	}

	// Last thread, should transpose rest of rows less than BLOCK
	// Not parallel for now

	if (
		(c_start >= data->c_blocks) &&
		(col_size - data->c_blocks*BLOCK_SIDE_SIZE > 0)
	) {

		src = data->input + data->c_blocks*BLOCK_SIDE_SIZE*row_size;
		dst = data->output + data->c_blocks*BLOCK_SIDE_SIZE;

		for (int j = 0; j < col_size - data->c_blocks*BLOCK_SIDE_SIZE; ++ j) {
			for (int i = 0; i < row_size; ++ i) {
				dst[i*col_size] = src[i];
			}
			src += row_size;
			dst ++;
		}
	}

}

void arrays_transpose_thr_in (void * _data) {

	struct arrays_transpose_thread_input *data = (struct arrays_transpose_thread_input *) _data;
	unsigned long int i, final_index;

	int num = data->elements_number;
	int row_size = data->r_size;
	int blocks_num = data->r_blocks;
	
	int r_start = data->r_start;
	int c_start = data->c_start;
	int r_rest = row_size - blocks_num*BLOCK_SIDE_SIZE;
	
	double complex *in = data->input;
	double complex *src = in + c_start*BLOCK_SIDE_SIZE*row_size + r_start*BLOCK_SIDE_SIZE;
	double complex *dst = in + r_start*BLOCK_SIDE_SIZE*row_size + c_start*BLOCK_SIDE_SIZE;

	while (num > 0) {

		double complex *src2 = src;
		double complex *dst2 = dst;

		if (r_start == c_start) {

			// Diagonal element - process only top triangle
			for (int j = 0; j < BLOCK_SIDE_SIZE; ++j) {
				for (int i = j+1; i < BLOCK_SIDE_SIZE; ++i) {
					double complex tmp = src2[i];
					src2[i] = dst2[i*row_size];
					dst2[i*row_size] = tmp;
				}
				dst2 += 1;
				src2 += row_size;
			}

		} else {
			// Full block transpose
			for (int j = 0; j < BLOCK_SIDE_SIZE; ++j) {
				for (int i = 0; i < BLOCK_SIDE_SIZE; ++i) {
					double complex tmp = src2[i];
					src2[i] = dst2[i*row_size];
					dst2[i*row_size] = tmp;
				}
				dst2 += 1;
				src2 += row_size;
			}
		}

		src += BLOCK_SIDE_SIZE;
		dst += BLOCK_SIDE_SIZE*row_size;

		-- num;
		++ r_start;

		if (r_start >= blocks_num) {

			// Transpose rest of row less than block size
			for (int j = 0; j < BLOCK_SIDE_SIZE; ++j) {
				for (int i = 0; i < r_rest; ++i) {
					double complex tmp = src[i];
					src[i] = dst[i*row_size];
					dst[i*row_size] = tmp;
				}
				src += row_size;
				dst += 1;
			}

			++ c_start;
			r_start = c_start;

			src = in + c_start*BLOCK_SIDE_SIZE*row_size + r_start*BLOCK_SIDE_SIZE;
			dst = in + r_start*BLOCK_SIDE_SIZE*row_size + c_start*BLOCK_SIDE_SIZE;
		}
	}

	// Last thread, should transpose last right-bottom square less than BLOCK
	if (
		(c_start >= blocks_num) &&
		(row_size - blocks_num*BLOCK_SIDE_SIZE > 1)
	) {

//		fprintf(stderr, "-- extra rows\n");

		src = in + blocks_num*BLOCK_SIDE_SIZE*row_size + blocks_num*BLOCK_SIDE_SIZE;
		dst = src;

		for (int j = 0; j < r_rest; ++ j) {
			for (int i = j+1; i < r_rest; ++ i) {
				double complex tmp = src[i];
				src[i] = dst[i*row_size];
				dst[i*row_size] = tmp;
			}
			src += row_size;
			dst += 1;
		}
	}
}

void arrays_transpose_with_threads_in (double complex *input, double complex *output, int row_size, int col_size, int number_of_threads) {

	(void)output;

	assert(row_size == col_size);

	int blocks_num = row_size/BLOCK_SIDE_SIZE;
	int total_blocks = blocks_num*(blocks_num+1)/2;

	each_thread_elements = (total_blocks+number_of_threads-1)/number_of_threads;
	last_thread_elements = total_blocks - each_thread_elements*(number_of_threads-1);

	int start_n = 0;
	int start_m = 0;

//	fprintf(stderr, "-- blocks/line = %i, total=%i, per_thread = %i (%i)\n", blocks_num, total_blocks, each_thread_elements, last_thread_elements);

	for (int i = 0; i < number_of_threads; ++i) {
	
		thr_arrays_transpose [i].input = input;
		thr_arrays_transpose [i].elements_number = each_thread_elements;
		thr_arrays_transpose [i].r_start = start_m;
		thr_arrays_transpose [i].c_start = start_n;
		thr_arrays_transpose [i].r_blocks = blocks_num;
		thr_arrays_transpose [i].r_size = row_size;

//		fprintf(stderr, "-- thr %i: start=%i/%i, p=%p\n", i, start_m, start_n, &thr_arrays_transpose[i]);

		int num = each_thread_elements;

		while (start_m + num >= blocks_num) {
			num -= (blocks_num - start_m);
			start_n ++;
			start_m = start_n;
		}

		start_m += num;
		my_thr_data_assign (i, (void *) &thr_arrays_transpose[i]);
	}

	thr_arrays_transpose [(number_of_threads-1)].elements_number = last_thread_elements;
	my_thr_manager (arrays_transpose_thr_in);
}
