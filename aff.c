
#ifdef __linux

#define _GNU_SOURCE
#include <sched.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>

#include "aff.h"
#include "my_thr_lib.h"

typedef struct {
	int cpu;
} _aff_data_t;

static void _thr_set_aff(void *_data) {
	_aff_data_t *data = (_aff_data_t*) _data;
	cpu_set_t set;
	
	CPU_ZERO(& set);
	CPU_SET(data->cpu, & set);
	if (sched_setaffinity(gettid(), sizeof(set), & set) != 0) {
		fprintf(stderr, "*** setaffinity failed for thread %i\n", gettid());
	}
}

void set_aff(int tnum) {

	cpu_set_t set;
	_aff_data_t *data = (_aff_data_t*)malloc(tnum*sizeof(_aff_data_t)); assert(data != NULL);
	
	CPU_ZERO(& set);
	if (sched_getaffinity(getpid(), sizeof(set), & set) != 0) {
		fprintf(stderr, "*** getaffinity failed, ignoring\n");
		goto _ret;
	}

	int idx = 0;
	
//	printf("--- d: cpu_count=%i\n", CPU_COUNT(& set));
	
	for (int i = 0; i < CPU_SETSIZE; ++ i) {
		if (CPU_ISSET(i, & set)) {
			data[idx].cpu = i;
			my_thr_data_assign (idx, (void *) & data[idx]);

//			printf("--- Bind thread %i to CPU %i\n", idx, i);

			++ idx;
			if (idx >= tnum) break;
		}
	}

	if (idx < tnum) {
		fprintf(stderr, "*** initial affinity contains less CPUs than requested threads number (only %i)\n", idx);
		goto _ret;
	}

	my_thr_manager (_thr_set_aff);

_ret:
	free(data);
}

typedef struct {
	void *p;
	size_t sz;
} _mem_data_t;

static void _thr_set_mem(void *_data) {
	_mem_data_t *data = (_mem_data_t*) _data;
	memset(data->p, 0, data->sz);
}

void set_mem_owner(void *in, size_t sz, int tnum) {
	size_t per_t = (sz+tnum-1)/tnum;
	_mem_data_t *data = (_mem_data_t*)malloc(tnum*sizeof(_mem_data_t)); assert(data != NULL);

	for (int i = 0; i < tnum; ++ i) {
		data[i].p = in;
		data[i].sz = (i == tnum-1) ? sz - (tnum-1)*per_t : per_t;
		my_thr_data_assign (i, (void *) & data[i]);
	}
	my_thr_manager (_thr_set_mem);

	free(data);
}

#else // ! __linux

void set_aff(int tnum) {}
void set_mem_owner(void *in, size_t sz, int tnum) { }

#endif

