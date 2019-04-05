#include <vector>
#include <cstdio>
#include <queue>
#include <iostream>
#include <assert.h>
#include <omp.h>
#include <cmath>
#include <unordered_set>


#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)
#else
#include <x86intrin.h>
#endif

#define USE_AVX
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif


typedef unsigned idx_t;

float fvec_L2sqr(const float *x, const float *y, size_t d);

void find_nearest(int nq, int num_results, int *results,       // matrix [n_queries, num_results]
                  int nq, int d, float *queries,               // matrix [n_queries, vec_dimension]
                  int nb, int d, float *vertices,              // matrix [n_vertices, vec_dimension]
                  int nb, int max_degree, int *edges,          // matrix [n_queries, max_degree]
                  int nq, int max_path, int *trajectories,     // matrix [n_queries, max_path]
                  int *initial_vertex_id,                      // number
                  int *ef,                                     // number
                  int *nt)                                     // number