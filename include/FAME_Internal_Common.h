#ifndef _FAME_INTERNAL_COMMON_H_
#define _FAME_INTERNAL_COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>

#include "vec_plus.h"
#include "vec_norm.h"
#include "vec_inner_prod.h"
#include "mtx_print.h"
#include "mtx_prod.h"
#include "mtx_trans.h"
#include "mtx_trans_conj.h"
#include "mtx_cat.h"
#include "mtx_dot_prod.h"
#include "kron_vec.h"
#include "inv3.h"

#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);
#define pi 3.141592653589793
#define idouble_pi I*2*pi
#define BILLION 1E9

typedef double _Complex cmpx;

typedef struct{
	double A[3];
	double B[3];
	double C[3];
	double D[3];
	double E[3];
	double F[3];
	double G[3];
	double H[3];
	double I[3];
	double J[3];
	double K[3];
	double L[3];
	double M[3];
	double N[3];
	double O[3];
	double P[3];
	double Q[3];
	double R[3];
	double S[3];
	double T[3];
	double U[3];
	double V[3];
	double W[3];
	double X[3];
	double Y[3];
	double Z[3];
	double s[3];
	double t[3];
	double k[3];
} VERTEX;

typedef struct{
	int    grid_nums[3];
	double edge_len[3];
	double mesh_len[3];
} MESH;

typedef struct{
	double  a;
	double  b;
	double  c;
	double  alpha;
	double  beta;
	double  gamma;
	int     Permutation[3];
	double  theta_1;
	double  theta_2;
	double  theta_3;
	int     m1;
	int     m2;
	int     m3;
	int     m4;
	double* t1;
	double* t2;
	double* t3;
	double* t4;
	int     rho_1;
	int     rho_2;
	int     rho_3;
	int     rho_4;
	int     rho_5;
	char    flag[6];
	double  length_a1;
	double  length_a2;
	double  length_a3;
} LATTICE_CONSTANT;

typedef struct{
    char    data_name[50];
	char    material_type[50];
	int*    Binout;
	int*	GInOut;
	int*    GInOut_index;
	int*    GInOut_index_length;
    int     material_num;
    int*    sphere_num;
    double* sphere_centers;
	double* sphere_radius;
	int*    cylinder_num;
	double* cylinder_top_centers;
	double* cylinder_bot_centers;
	double* cylinder_radius;
	int     num_ele_permitt_in;
	int     num_reciprocity_in;
	int		num_mag_permeab_in;
	int		num_chirality_in;
	double* ele_permitt_in;
	double  ele_permitt_out;
	double* mag_permeab_in;
	double  mag_permeab_out;
	double* reciprocity_in;
	double  reciprocity_out;
	double* chirality_in;
	double  chirality_out;
	int     flag_radius_adjustment;
	double  flag_radius;
} MATERIAL;

typedef struct{
	int     part_num;
	int     Wave_vec_num;
	double* WaveVector;
	char    path_string[50];
	double  reciprocal_lattice_vector_b[9];
	VERTEX  vertex;
} RECIP_LATTICE;

typedef struct{
	char   lattice_type[50];
	int    Permutation[3];
	double lattice_vec_a[9];
	double lattice_vec_a_orig[9];
	double Omega[9];
	LATTICE_CONSTANT lattice_constant;
} LATTICE;

typedef struct{
	double* Lambda_q_sqrt;
	cmpx* Lambda_x;
	cmpx* Lambda_y;
	cmpx* Lambda_z;
	cmpx* D_kx;
	cmpx* D_ky;
	cmpx* D_kz;
	cmpx* D_k;
	cmpx* D_ks;
	cmpx* Pi_Qr;
	cmpx* Pi_Pr;
	cmpx* Pi_Qrs;
	cmpx* Pi_Prs;
	
} LAMBDAS;

typedef struct{
	int*  C1_r;
	int*  C1_c;
	cmpx* C1_v;
	int*  C2_r;
	int*  C2_c;
	cmpx* C2_v;
	int*  C3_r;
	int*  C3_c;
	cmpx* C3_v;
	int*   C_r;
	int*   C_c;
	cmpx*  C_v;
} MTX_C;

typedef struct{
	int    maxit;
	double tol;
} LS;

typedef struct{
	int    nwant;
	int    nstep;
	int    maxit;
	double tol;
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
	double radius;
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
	LS ls;
	ES es;
} PAR;

typedef struct{
	int     idx;
	int*    ls_iter;
	int*    es_iter;
	double* ls_time;
	double* es_time;
  //double* qr_time;
  //double* qrs_time;
 // double* fft_time;
 // double* ifft_time;
 // double* eigen_time;
} PROFILE;
#endif
