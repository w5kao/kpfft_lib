/* Program for benchmarking of my_fft library. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <complex.h>
#include <math.h>

#include <fftw3.h>

#include <time.h>
#include <sys/time.h>
#include <linux/limits.h>

#include "kpfft_lib.h"

#include "aff.h"

#ifdef USE_OMP
#include <omp.h>
#elif defined(_OPENMP)
#error OpenMP enabled but not used
#endif


#define VERBOSE
#define _SUPER_VERBOSE

#define ACCURACY_CONTROL

/* FFTW method uses best value from 10 runs */
#define RUNS_NUMBER 10
#define _FORGET_WISDOM_BEFORE_FFTW
#define _READ_THR_WISDOM_FOR_FFTW
#define _SAVE_THR_WISDOM_FOR_FFTW

#define PLAN_FLAGS FFTW_MEASURE
//#define PLAN_FLAGS FFTW_PATIENT
//#define PLAN_FLAGS FFTW_EXAUSTIVE

#define MODE_C 1
#define MODE_C2R 2
#define MODE_R2C 3

fftw_plan *plans_fwd_X, *plans_fwd_Y;
double complex *in, *out, *out2;

unsigned long int NX=0, NY=0, num_reps=0, threads_number=0;

void complex_array_print (double complex *input, unsigned long int NX_size, unsigned long int NY_size);

#ifdef __linux
long int get_stat(int n) {

	FILE *f;
	pid_t pid = getpid();
	char path[32];
	long int ret = -1;

	snprintf(path, sizeof(path), "/proc/%i/stat", pid);
	f = fopen(path, "r");
	if (f == NULL) return ret;

	char *line = NULL, *tok;
	size_t len = 0;
	int rc = getline(& line, & len, f);
	if (rc == 0) return -1;
	if (len > 0) {
		tok = strtok(line, " ");
		while (tok != NULL) {
			if (n == 0) {
				ret = atoll(tok);
				break;
			}
			tok = strtok(NULL, " ");
			-- n;
		}
	}
	if (line != NULL) free(line);
	fclose(f);
	return ret;
}

long int get_cpu() {
	return get_stat(13);
}

#else

long int get_cpu() {
	return -1;
}

#endif





double complex_array_control_sum (double complex *input, unsigned long int num_els) {
	unsigned long int i;
	double sum = 0.0;

	for (i = 0; i < num_els; ++i) {
		sum += cabs(input[i]);
	}

	return sum;
}

double real_array_control_sum (double *input, unsigned long int num_els) {
	unsigned long int i;
	double sum = 0.0;

	for (i = 0; i < num_els; ++i) {
		sum += fabs(input[i]);
	}

	return sum;
}

void complex_array_print (double complex *input, unsigned long int rows_N, unsigned long int cols_M) {
	unsigned long int i, j;
	double complex *temp_ptr;

	printf ("\n");
	temp_ptr = input;
	for (i = 0; i < rows_N; ++i) {
		for (j = 0; j < cols_M; ++j) {
			printf ("%.2e+i*%.2e ", creal(*temp_ptr), cimag(*temp_ptr));
			++temp_ptr;
		}
		printf ("\n");
	}
}

void complex_array_indexes (double complex *input, unsigned long int rows_N, unsigned long int cols_M) {
	unsigned long int i, j;
	double complex *temp_ptr;

	temp_ptr = input;
	for (i = 0; i < rows_N; ++i) {
		for (j = 0; j < cols_M; ++j) {
			(*temp_ptr) = (i+1) + I*(j+1);
			++temp_ptr;
		}
		printf ("\n");
	}
}

static void complex_array_harmonics (double complex *input, unsigned long int NX_size, unsigned long int NY_size, double k_x, double k_y) {
	unsigned long int i, j;
	double complex *temp_ptr;
	double x, y, delta_x, delta_y;

	delta_x = 2.0*(M_PI)/NX_size;
	delta_y = 2.0*(M_PI)/NY_size;
	temp_ptr = input;
	for (i = 0; i < NX_size; ++i) {
		for (j = 0; j < NY_size; ++j) {
			x = delta_x*i;
			y = delta_y*j;
			(*temp_ptr) = cexp(I*(k_x*x + k_y*y));
			++temp_ptr;
		}
	}
}

static void real_array_harmonics (double *input, unsigned long int NX_size, unsigned long int NY_size, double k_x, double k_y) {
	unsigned long int i, j;
	double *temp_ptr;
	double x, y, delta_x, delta_y;

	delta_x = 2.0*(M_PI)/NX_size;
	delta_y = 2.0*(M_PI)/NY_size;
	temp_ptr = input;
	for (i = 0; i < NX_size; ++i) {
		for (j = 0; j < NY_size; ++j) {
			x = delta_x*i;
			y = delta_y*j;
			(*temp_ptr) = sin(k_x*x + k_y*y);
			++temp_ptr;
		}
	}
}

static void complex_array_element_print (double complex *input, unsigned long int rows_N, unsigned long int cols_M, unsigned long int row, unsigned long int col) {
	double complex element = input[row*cols_M + col];
	printf ("array[%lu,%lu] = %.15e + i*%.15e\n", row, col, creal(element), cimag(element));

}

static void load_wisdom(const char *fname) {

	FILE *f_p;

	if ((f_p = fopen(fname, "rb")) == NULL) {
		printf ("Can't read wisdom file!\n");
		printf ("Trying to calculate plans without wisdom...\n");
	} else {
		if (fftw_import_wisdom_from_file (f_p)) {
			printf ("Wisdom successfully imported.\n");
		} else {
			printf ("File with wisdom is corrupted.\n");
			printf ("Trying to calculate plans without wisdom...\n");
		}
		fclose (f_p);
	}
}

static void save_wisdom(const char *fname) {

	FILE *f_p;

	if ((f_p = fopen(fname, "w+")) == NULL) {
		printf ("Can't open wisdom file for writing!\n");
		printf ("Wisdom has been lost!\n");
	} else {
		printf ("Exporting wisdom to file...\n");
		fftw_export_wisdom_to_file (f_p);
		fclose (f_p);
	}
}

static struct timespec start, finish;

static void measure_start() {
	clock_gettime(CLOCK_MONOTONIC, &start);
}

static double measure_end() {

	double seconds;

	clock_gettime(CLOCK_MONOTONIC, &finish);
	seconds = (finish.tv_sec - start.tv_sec);
	seconds += (finish.tv_nsec - start.tv_nsec) / (1.0e+9);
	return seconds;
}


const char *wisdom_base = "wisdom";

int main (int argc, char** argv) {
	double mismatch=0.0, seconds, my_seconds_min, FFTW_seconds_min, ops, MFLOPS, my_MFLOPS_max, FFTW_MFLOPS_max;
	unsigned long int i, runs;
	FILE *f_p;
	fftw_plan normal_plan;
	my_fft_plan plan_fwd;
	char wisdom[256];
	char *output = NULL;
	long int cpu0, cpu1;
	int mode;

	/*Parsing command line.*/
	if (argc < 6)
	{
		printf ("Usage:\n");
		printf("%s (c|r2c|c2r) NX NY num_of_repeats number_of_threads [output_file]\n", argv[0]);
		exit (1);
	}

	if (strcmp(argv[1], "c") == 0)
		mode = MODE_C;
	else if (strcmp(argv[1], "c2r") == 0)
		mode = MODE_C2R;
	else if (strcmp(argv[1], "r2c") == 0)
		mode = MODE_R2C;
	else {
		printf("Unknown mode\n");
		exit(1);
	}

	sscanf(argv[2], "%lu", &NX);
	sscanf(argv[3], "%lu", &NY);
	sscanf(argv[4], "%lu", &num_reps);
	sscanf(argv[5], "%lu", &threads_number);
	
	if (argc > 6) {
		output = argv[6];
	}
	
#ifdef VARIABLE_BLOCK_SIDE
	const char *env = getenv("BLOCK_SIDE");
	if (env) {
		BLOCK_SIDE_SIZE = atoi(env);
	}
#endif
	
#ifdef VERBOSE
	printf ("NX = %lu, NY = %lu, num_reps = %lu, threads_number=%lu block_side=%i\n", NX, NY, num_reps, threads_number, BLOCK_SIDE_SIZE);
#endif /* VERBOSE */
/*** initialization and calculation of arrays ***/

#ifdef USE_OMP
	omp_set_dynamic(0);     // Explicitly disable dynamic teams
	omp_set_num_threads(1); 
#endif

	ops = 5*NX*NY*log2(NX*NY);

	my_fft_init (threads_number);

	size_t sz = (NX)*(NY)*sizeof(double complex);

	in = (double complex*) fftw_malloc (sz);
	out = (double complex*) fftw_malloc (sz);
	out2 = (double complex*) fftw_malloc (sz);

	if (getenv("CPU_AFF") != NULL && threads_number > 0) {
	printf("Setting CPU affinity ...\n");
	set_aff(threads_number);
	set_mem_owner(in, sz, threads_number);
	set_mem_owner(out, sz, threads_number);
	set_mem_owner(out2, sz, threads_number);

	}

//	fftw_init_threads();
//	fftw_plan_with_nthreads(1);

	snprintf(wisdom, sizeof(wisdom), "%s_1d.%s.t%lu.f%i", wisdom_base, output, threads_number, PLAN_FLAGS);
	load_wisdom(wisdom);

	clock_gettime(CLOCK_MONOTONIC, &start);

	measure_start();
	switch(mode) {
		case MODE_C:
			plan_fwd = my_fft_plan_dft_2d (in, out, out2, NX, NY, +1, PLAN_FLAGS, threads_number);
			break;
		case MODE_C2R:
			plan_fwd = my_fft_plan_dft_c2r_2d (in, (double*)out, out2, NX, NY, +1, PLAN_FLAGS, threads_number);
			break;
		case MODE_R2C:
			plan_fwd = my_fft_plan_dft_r2c_2d ((double*)in, out, out2, NX, NY, -1, PLAN_FLAGS, threads_number);
			break;
	}
	seconds = measure_end();
	printf ("MY plan created, elapsed:  %e sec\n", seconds);

	save_wisdom(wisdom);

	/*
	for (i = 0; i < NX*NY; ++i) {
		in[i] = sin(i) + I*cos(i);
	}
	*/
	my_MFLOPS_max = 0.0;
	switch(mode) {
		case MODE_C:
		case MODE_C2R:
			complex_array_harmonics (in, NX, NY, -1.0, -2.0);
			break;
		case MODE_R2C:
			real_array_harmonics ((double*)in, NX, NY, -1.0, -2.0);
			break;
	}
	for (runs=0; runs < (RUNS_NUMBER); ++runs) {

		clock_gettime(CLOCK_MONOTONIC, &start);
		measure_start();
		cpu0 = get_cpu();
		for (i = 0; i < num_reps; ++i) {
			my_fft_execute (plan_fwd);
		}
		seconds = measure_end();
		cpu1 = get_cpu();

		printf ("Time for %lu Fourier transform using my_fft is %.15e seconds (CPU - %f sec).\n", num_reps, seconds, 1.*(cpu1-cpu0)/100);
		MFLOPS = (1.0e-6)*num_reps*ops/seconds;
		if (MFLOPS > my_MFLOPS_max) {
			my_MFLOPS_max = MFLOPS;
			my_seconds_min = seconds;
		}
	}
	printf ("MFLOPs = %lf\n", my_MFLOPS_max);
//	printf ("Eff. CPUs = %f\n", 1.*(cpu1-cpu0)/100/seconds);

#ifdef VERBOSE
	switch(mode) {
		case MODE_C:
		case MODE_R2C:
			printf ("Control sum of cabs(out-array) is: %.15e\n", complex_array_control_sum(out, NX*NY));
			complex_array_element_print (out, NX, NY, 1, 2);
			complex_array_element_print (out, NX, NY, 2, 1);
			break;
		case MODE_C2R: {
			double* out_r = (double*)out;
			printf ("Control sum of cabs(out-array) is: %.15e\n", real_array_control_sum(out_r, NX*NY));
			printf("[1,2] = %e\n", out_r[NY*1+2]);
			printf("[2,1] = %e\n", out_r[NY*2+1]);
			break;
			}
	}
#endif /* VERBOSE */

	my_fft_destroy_plan (plan_fwd);

	my_fft_clear();

#ifdef FORGET_WISDOM_BEFORE_FFTW
	fftw_forget_wisdom();
#endif /* FORGET_WISDOM_BEFORE_FFTW */

	fflush(stdout);

#ifdef ACCURACY_CONTROL

#ifdef USE_OMP
	omp_set_num_threads(threads_number); 
#endif

	if (threads_number > 0) {
		fftw_init_threads();
		fftw_plan_with_nthreads(threads_number);
	}

	snprintf(wisdom, sizeof(wisdom), "%s_2d.%s.t%lu.f%i", wisdom_base, output, threads_number, PLAN_FLAGS);

#ifdef READ_THR_WISDOM_FOR_FFTW
	load_wisdom(wisdom);
#endif

	measure_start();
	switch(mode) {
		case MODE_C:
			normal_plan = fftw_plan_dft_2d(NX, NY, in, out2, +1, PLAN_FLAGS);
			break;
		case MODE_C2R:
			normal_plan = fftw_plan_dft_c2r_2d(NX, NY, in, (double*)out2, PLAN_FLAGS);
			break;
		case MODE_R2C:
			normal_plan = fftw_plan_dft_r2c_2d(NX, NY, (double*)in, out2, PLAN_FLAGS);
			break;
	}
	seconds = measure_end();
	printf ("FFTW plan created, elapsed:  %e sec\n", seconds);

#ifdef SAVE_THR_WISDOM_FOR_FFTW
	save_wisdom(wisdom);
#endif

	FFTW_MFLOPS_max = 0.0;
	switch(mode) {
		case MODE_C:
		case MODE_C2R:
			complex_array_harmonics (in, NX, NY, -1.0, -2.0);
			break;
		case MODE_R2C:
			real_array_harmonics ((double*)in, NX, NY, -1.0, -2.0);
			break;
	}

	for (runs=0; runs < (RUNS_NUMBER); ++runs) {

		measure_start();
		cpu0 = get_cpu();
		for (i = 0; i < num_reps; ++i) {
			fftw_execute (normal_plan);
		}
		seconds = measure_end();
		cpu1 = get_cpu();

		printf ("Time for %lu Fourier transform using FFTW is %.15e seconds (CPU - %f sec).\n", num_reps, seconds, 1.*(cpu1-cpu0)/100);
		MFLOPS = (1.0e-6)*num_reps*ops/seconds;
		if (MFLOPS > FFTW_MFLOPS_max) {
			FFTW_MFLOPS_max = MFLOPS;
			FFTW_seconds_min = seconds;
		}
	}
	printf ("MFLOPs = %lf\n", FFTW_MFLOPS_max);
	printf ("Eff. CPUs = %f\n", 1.*(cpu1-cpu0)/100/seconds);
#ifdef VERBOSE
	switch(mode) {
		case MODE_C:
		case MODE_R2C:
			printf ("Control sum of cabs(out2-array) is: %.15e\n", complex_array_control_sum(out2, NX*NY));
			complex_array_element_print (out2, NX, NY, 1, 2);
			complex_array_element_print (out2, NX, NY, 2, 1);
			break;
		case MODE_C2R: {
			double* out_r = (double*)out2;
			printf ("Control sum of abs(out2-array) is: %.15e\n", real_array_control_sum(out_r, NX*NY));
			printf("[1,2] = %e\n", out_r[NY*1+2]);
			printf("[2,1] = %e\n", out_r[NY*2+1]);
			break;
			}
	}
#endif /* VERBOSE */

	fftw_destroy_plan (normal_plan);
	fflush(stdout);

	switch(mode) {
		case MODE_C:
		case MODE_R2C:
			for (i = 0; i < NX*NY; ++i) {
				mismatch += cabs(out[i] - out2[i]);
			}
			break;
		case MODE_C2R: {
			double* out_r1 = (double*)out;
			double* out_r2 = (double*)out2;
			for (i = 0; i < NX*NY; ++i) {
				mismatch += fabs(out_r1[i] - out_r2[i]);
			}
			break;
			}
	}
	printf ("Mismatch between my_fft and fftw = %.15e\n", mismatch);

#endif /* ACCURACY_CONTROL */

	if (output) {
		if ((f_p = fopen(output, "a+")) == NULL) {
			printf ("Can't open output file %s for writing!\n", output);
		} else {
			fprintf (f_p, "%lu\t%lu\t%lu\t%lf\t%lf\t%lf\t%lf\t%i\n", NX, NY, threads_number, my_MFLOPS_max, FFTW_MFLOPS_max, my_seconds_min, FFTW_seconds_min, BLOCK_SIDE_SIZE);
			fclose (f_p);
		}
	}

	return 0;
}
