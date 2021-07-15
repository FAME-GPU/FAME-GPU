#ifndef _FAME_INTERNAL_COMMON_H_
#define _FAME_INTERNAL_COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <iostream>
#include <complex.h>
#include <FAME_Use_Single.h>
using namespace std;

#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);
#define pi 3.141592653589793
#define idouble_pi I*2*pi
#define BILLION 1E9


#if defined(USE_SINGLE)

typedef float realCPU;
typedef float _Complex cmpxCPU;
#define lapack_complex_float cmpxCPU
#define FAME_lapacke_pteqr LAPACKE_cpteqr
#define FAME_lapacke_stegr LAPACKE_cstegr

#else

typedef double realCPU;
typedef double _Complex cmpxCPU;
#define lapack_complex_double cmpxCPU
#define FAME_lapacke_pteqr LAPACKE_zpteqr
#define FAME_lapacke_stegr LAPACKE_zstegr
#endif // end of USE_SINGLE

typedef struct{
	realCPU A[3];
	realCPU B[3];
	realCPU C[3];
	realCPU D[3];
	realCPU E[3];
	realCPU F[3];
	realCPU G[3];
	realCPU H[3];
	realCPU Ii[3];
	realCPU J[3];
	realCPU K[3];
	realCPU L[3];
	realCPU M[3];
	realCPU N[3];
	realCPU O[3];
	realCPU P[3];
	realCPU Q[3];
	realCPU R[3];
	realCPU S[3];
	realCPU T[3];
	realCPU U[3];
	realCPU V[3];
	realCPU W[3];
	realCPU X[3];
	realCPU Y[3];
	realCPU Z[3];
	realCPU s[3];
	realCPU t[3];
	realCPU k[3];
} VERTEX;

typedef struct{
	int    grid_nums[3];
	realCPU edge_len[3];
	realCPU mesh_len[3];
} MESH;

typedef struct{
	realCPU  a;
	realCPU  b;
	realCPU  c;
	realCPU  alpha;
	realCPU  beta;
	realCPU  gamma;
	int     Permutation[3];
	realCPU  theta_1;
	realCPU  theta_2;
	realCPU  theta_3;
	int     m1;
	int     m2;
	int     m3;
	int     m4;
	realCPU* t1;
	realCPU* t2;
	realCPU* t3;
	realCPU* t4;
	int     rho_1;
	int     rho_2;
	int     rho_3;
	int     rho_4;
	int     rho_5;
	char    flag[6];
	realCPU  length_a1;
	realCPU  length_a2;
	realCPU  length_a3;
} LATTICE_CONSTANT;

typedef struct{
    char    data_name[50];
	char    material_type[50];
	int*    Binout;
	int*    BInOut;
	int*    BInOut_index;
	int*    BInOut_index_length;
    int     material_num;
    int*    sphere_num;
    realCPU* sphere_centers;
	realCPU* sphere_radius;
	int*    cylinder_num;
	realCPU* cylinder_top_centers;
	realCPU* cylinder_bot_centers;
	realCPU* cylinder_radius;
	realCPU* ele_permitt_in;
	realCPU  ele_permitt_out;
	realCPU* mag_permeab_in;
	realCPU  mag_permeab_out;
	realCPU* reciprocity_in;
	realCPU  reciprocity_out;
	realCPU* chirality_in;
	realCPU  chirality_out;
	int     flag_radius_adjustment;
	realCPU  flag_radius;
} MATERIAL;

typedef struct{
	int     part_num;
	int     Wave_vec_num;
	realCPU* WaveVector;
	char    path_string[50];
	realCPU  reciprocal_lattice_vector_b[9];
	VERTEX  vertex;
} RECIP_LATTICE;

typedef struct{
	char   lattice_type[50];
	int    Permutation[3];
	realCPU lattice_vec_a[9];
	realCPU lattice_vec_a_orig[9];
	realCPU Omega[9];
	LATTICE_CONSTANT lattice_constant;
} LATTICE;

typedef struct{
	realCPU* Lambda_q_sqrt;
	cmpxCPU* Lambda_x;
	cmpxCPU* Lambda_y;
	cmpxCPU* Lambda_z;
	cmpxCPU* D_kx;
	cmpxCPU* D_ky;
	cmpxCPU* D_kz;
	cmpxCPU* D_k;
	cmpxCPU* D_ks;
	cmpxCPU* Pi_Qr;
	cmpxCPU* Pi_Pr;
	cmpxCPU* Pi_Qrs;
	cmpxCPU* Pi_Prs;
	cmpxCPU* Pi_Qr_110;
	cmpxCPU* Pi_Pr_110;
	cmpxCPU* Pi_Qrs_110;
	cmpxCPU* Pi_Prs_110;
	cmpxCPU* Pi_Qr_101;
	cmpxCPU* Pi_Pr_101;
	cmpxCPU* Pi_Qrs_101;
	cmpxCPU* Pi_Prs_101;
	cmpxCPU* Pi_Qr_011;
	cmpxCPU* Pi_Pr_011;
	cmpxCPU* Pi_Qrs_011;
	cmpxCPU* Pi_Prs_011;
	
} LAMBDAS;

typedef struct{
	int*  C1_r;
	int*  C1_c;
	cmpxCPU* C1_v;
	int*  C2_r;
	int*  C2_c;
	cmpxCPU* C2_v;
	int*  C3_r;
	int*  C3_c;
	cmpxCPU* C3_v;
	int*   C_r;
	int*   C_c;
	cmpxCPU*  C_v;
	int*   C_110_r;
	int*   C_110_c;
	cmpxCPU*  C_110_v;
	int*   C_101_r;
	int*   C_101_c;
	cmpxCPU*  C_101_v;
	int*   C_011_r;
	int*   C_011_c;
	cmpxCPU*  C_011_v;
} MTX_C;

typedef struct{
	int    maxit;
	realCPU tol;
} LS;

typedef struct{
	int    nwant;
	int    nstep;
	int    maxit;
	realCPU tol;
} ES;

typedef struct{
	int device;
	int printf_user_option;
	int printf_parameter;
	int create_parameter;
	int create_B_inout;
	int create_wave_vector;
	int save_eigen_vector;
	int grid_nums_max;
	int mem_size;
	int radius_adjustment;
	realCPU radius;
} FLAG;

typedef struct{
	FLAG          flag;
	MESH          mesh;
	MATERIAL      material;
	RECIP_LATTICE recip_lattice;
	LS ls;
	ES es;
} POPT;

typedef struct{
	FLAG          flag;
	MESH          mesh;
	LATTICE       lattice;
	MATERIAL      material;
	RECIP_LATTICE recip_lattice;
	realCPU ce_error;
	LS ls;
	ES es;
} PAR;

typedef struct{
	int     idx;
	int*    ls_iter;
	int*    es_iter;
	realCPU* ls_time;
	realCPU* es_time;
  //realCPU* qr_time;
  //realCPU* qrs_time;
 // realCPU* fft_time;
 // realCPU* ifft_time;
 // realCPU* eigen_time;
} PROFILE;
#endif
