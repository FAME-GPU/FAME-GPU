#include "FAME_Internal_Common.h"
#include "FAME_Create_C_txt.h"
#include <complex.h>
//#include <iostream>
//using namespace std;

void FAME_Matrix_Curl_Simple( double* wave_vec,int* grid_num, double* edge_len, double* mesh_len, LATTICE lattice,
						      int* C1_r, int* C1_c, cmpx* C1_v,
						      int* C2_r, int* C2_c, cmpx* C2_v,
						      int* C3_r, int* C3_c, cmpx* C3_v);
void FAME_Matrix_Curl_General( double* wave_vec,int* grid_num, double* edge_len, double* mesh_len, LATTICE lattice,
						       int* C1_r, int* C1_c, cmpx* C1_v,
						       int* C2_r, int* C2_c, cmpx* C2_v,
						       int* C3_r, int* C3_c, cmpx* C3_v);
void spMTX_FDM_1d_quasiperiodic(int dim , double mesh_len, cmpx theta, int* K_row, int* K_col, cmpx* K_val);
void spMTX_FDM_2d_quasiperiodic(int dim2, int dim1, double mesh_len, cmpx theta, int* K_row, int* K_col, cmpx* K_val);
void spMTX_FDM_3d_quasiperiodic(int dim3, int dim2, int dim1, double mesh_len, cmpx theta, int* K_row, int* K_col, cmpx* K_val);

void speye(int dim, int* row, int* col, int* val)
{
	for(int i = 0; i < dim; i++)
	{
	    row[i] = i;
	    col[i] = i;
	    val[i] = 1;
	}
}

double dot(double* array1, double* array2, int len)
{
    double sum = 0.0;
    for( int i = 0; i < len; i++ )
    	sum += array1[i]*array2[i];
    return sum;
}

int spkron_mm(int*  row1, int*  col1, int*  val1, int size1, int len1,
              int*  row2, int*  col2, int*  val2, int size2, int len2,
              int* out_r, int* out_c, int* out_v)
{
    for( int i = 0; i < len1; i++ )
    	for( int j = 0; j < len2; j++ )
		{
	    	out_r[i*len2 + j] = row1[i]*size2 + row2[j];
	    	out_c[i*len2 + j] = col1[i]*size2 + col2[j];
	    	out_v[i*len2 + j] = val1[i]*val2[j];
		}
    return 0;
}

int spkron_mm( int*  row1, int*  col1,  int*  val1, int size1, int len1,
	         int*  row2, int*  col2, cmpx*  val2, int size2, int len2,
	         int* out_r, int* out_c, cmpx* out_v)
{
    for( int i = 0; i < len1; i++ )
        for( int j = 0; j < len2; j++ )
        {
            out_r[i*len2 + j] = row1[i]*size2 + row2[j];
            out_c[i*len2 + j] = col1[i]*size2 + col2[j];
            out_v[i*len2 + j] = val1[i]*val2[j];
        }
    return 0;
}

int spkron_mm( int*  row1, int*  col1, cmpx*  val1, int size1, int len1,
	         int*  row2, int*  col2,  int*  val2, int size2, int len2,
	         int* out_r, int* out_c, cmpx* out_v)
{
    for( int i = 0; i < len1; i++ )
        for( int j = 0; j < len2; j++ )
        {
            out_r[i*len2 + j] = row1[i]*size2 + row2[j];
	    	out_c[i*len2 + j] = col1[i]*size2 + col2[j];
            out_v[i*len2 + j] = val1[i]*val2[j];
        }
    return 0;
}

int spkron_mm( int*  row1, int*  col1, cmpx*  val1, int size1, int len1,
	         int*  row2, int*  col2, cmpx*  val2, int size2, int len2,
	         int* out_r, int* out_c, cmpx* out_v)
{
    for( int i = 0; i < len1; i++ )
        for( int j = 0; j < len2; j++ )
        {
            out_r[i*len2 + j] = row1[i]*size2 + row2[j];
            out_c[i*len2 + j] = col1[i]*size2 + col2[j];
            out_v[i*len2 + j] = val1[i]*val2[j];
        }
    return 0;
}

int FAME_Matrix_Curl(MTX_C* mtx_C, double* wave_vec,int* grid_num, double* edge_len, double* mesh_len, LATTICE lattice)
{
	if( (strcmp(lattice.lattice_type, "simple_cubic"          ) == 0) || \
		(strcmp(lattice.lattice_type, "primitive_orthorhombic") == 0) || \
		(strcmp(lattice.lattice_type, "primitive_tetragonal"  ) == 0) )
		FAME_Matrix_Curl_Simple( wave_vec, grid_num, edge_len, mesh_len, lattice,
						 		 mtx_C->C1_r, mtx_C->C1_c, mtx_C->C1_v,
						 		 mtx_C->C2_r, mtx_C->C2_c, mtx_C->C2_v,
						 		 mtx_C->C3_r, mtx_C->C3_c, mtx_C->C3_v );
	else
		FAME_Matrix_Curl_General( wave_vec, grid_num, edge_len, mesh_len, lattice,
						 		  mtx_C->C1_r, mtx_C->C1_c, mtx_C->C1_v,
						 		  mtx_C->C2_r, mtx_C->C2_c, mtx_C->C2_v,
						 		  mtx_C->C3_r, mtx_C->C3_c, mtx_C->C3_v );

	int N = grid_num[0] * grid_num[1] * grid_num[2];

	for(int i = 0; i < 2*N; i++)
	{
		mtx_C->C_r[i]     = mtx_C->C1_r[i] +   N;
		mtx_C->C_c[i]     = mtx_C->C1_c[i] + 2*N;
		mtx_C->C_v[i]     = -1.0*mtx_C->C1_v[i];
		mtx_C->C_r[i+2*N] = mtx_C->C1_r[i] + 2*N;
		mtx_C->C_c[i+2*N] = mtx_C->C1_c[i] +   N;
		mtx_C->C_v[i+2*N] =  1.0*mtx_C->C1_v[i];

		mtx_C->C_r[i+4*N] = mtx_C->C2_r[i];
		mtx_C->C_c[i+4*N] = mtx_C->C2_c[i] + 2*N;
		mtx_C->C_v[i+4*N] =  1.0*mtx_C->C2_v[i];
		mtx_C->C_r[i+6*N] = mtx_C->C2_r[i] + 2*N;
		mtx_C->C_c[i+6*N] = mtx_C->C2_c[i];
		mtx_C->C_v[i+6*N] = -1.0*mtx_C->C2_v[i];

		mtx_C->C_r[i+8*N]  = mtx_C->C3_r[i];
		mtx_C->C_c[i+8*N]  = mtx_C->C3_c[i] + N;
		mtx_C->C_v[i+8*N]  = -1.0*mtx_C->C3_v[i];
		mtx_C->C_r[i+10*N] = mtx_C->C3_r[i] + N;
		mtx_C->C_c[i+10*N] = mtx_C->C3_c[i];
		mtx_C->C_v[i+10*N] =  1.0*mtx_C->C3_v[i];
	}
	//FAME_Create_C_txt(mtx_C->C1_r, mtx_C->C1_c, mtx_C->C1_v, mtx_C->C2_r,  mtx_C->C2_c, mtx_C->C2_v, mtx_C->C3_r,  mtx_C->C3_c,  mtx_C->C3_v, grid_num);
	return 0;
}

void FAME_Matrix_Curl_Simple( double* wave_vec,int* grid_num, double* edge_len, double* mesh_len, LATTICE lattice,
						    int* C1_r, int* C1_c, cmpx* C1_v,
						    int* C2_r, int* C2_c, cmpx* C2_v,
						    int* C3_r, int* C3_c, cmpx* C3_v )
{
	int n1 = grid_num[0], n2 = grid_num[1], n3 = grid_num[2];

    cmpx i2pika1 = idouble_pi*dot(&lattice.lattice_vec_a[0], wave_vec, 3);
    cmpx i2pika2 = idouble_pi*dot(&lattice.lattice_vec_a[3], wave_vec, 3);
    cmpx i2pika3 = idouble_pi*dot(&lattice.lattice_vec_a[6], wave_vec, 3);

	/////////////// Create C1 with quasi_periodic  boundary condtion ///////////////
	int*  K_1_r = (int*) calloc( 2*n1, sizeof(int) );
	int*  K_1_c = (int*) calloc( 2*n1, sizeof(int) );
	cmpx* K_1_v = (cmpx*)calloc( 2*n1, sizeof(cmpx));

	spMTX_FDM_1d_quasiperiodic( n1, mesh_len[0], i2pika1, K_1_r, K_1_c, K_1_v );

	int* I_zy_r = (int*)malloc( n3*n2*sizeof(int));
    int* I_zy_c = (int*)malloc( n3*n2*sizeof(int));
    int* I_zy_v = (int*)malloc( n3*n2*sizeof(int));

	speye( n3*n2, I_zy_r, I_zy_c, I_zy_v );
	// C_1  =  kron( I_z , kron( I_y , K_x ) );
	spkron_mm( I_zy_r,  I_zy_c,  I_zy_v, n3*n2,   n3*n2,
		       K_1_r ,  K_1_c ,  K_1_v ,    n1,    2*n1,
		       C1_r  ,  C1_c  ,  C1_v  );

	free(K_1_r);  free(K_1_c);  free(K_1_v);
	free(I_zy_r); free(I_zy_c); free(I_zy_v);

	/////////////// Create C2 with quasi_periodic  boundary condtion ///////////////
	int*  K_2_r = (int*) calloc( 2*n2, sizeof(int) );
	int*  K_2_c = (int*) calloc( 2*n2, sizeof(int) );
	cmpx* K_2_v = (cmpx*)calloc( 2*n2, sizeof(cmpx));

	spMTX_FDM_1d_quasiperiodic( n2, mesh_len[1], i2pika2, K_2_r, K_2_c, K_2_v );

	int*  temp_r = (int*) calloc( 2*n1*n2, sizeof(int) );
	int*  temp_c = (int*) calloc( 2*n1*n2, sizeof(int) );
	cmpx* temp_v = (cmpx*)calloc( 2*n1*n2, sizeof(cmpx));

	int* I_x_r = (int*)malloc( n1*sizeof(int));
    int* I_x_c = (int*)malloc( n1*sizeof(int));
    int* I_x_v = (int*)malloc( n1*sizeof(int));

    speye( n1, I_x_r, I_x_c, I_x_v );
    // C_2  =  kron( I_z , kron( K_y , I_x ) );
	spkron_mm(   K_2_r,   K_2_c,   K_2_v, n2, 2*n2,
		       I_x_r,   I_x_c,   I_x_v, n1,   n1,
		      temp_r,  temp_c,  temp_v );

	free(K_2_r);  free(K_2_c);  free(K_2_v);
	free(I_x_r);  free(I_x_c);  free(I_x_v);

	int* I_z_r = (int*)malloc( n3*sizeof(int));
    int* I_z_c = (int*)malloc( n3*sizeof(int));
    int* I_z_v = (int*)malloc( n3*sizeof(int));

    speye( n3, I_z_r, I_z_c, I_z_v );

	spkron_mm( I_z_r ,  I_z_c ,  I_z_v ,    n3,      n3,
		       temp_r,  temp_c,  temp_v, n1*n2, 2*n1*n2,
		       C2_r  ,  C2_c  ,  C2_v  );

	free(temp_r);  free(temp_c);  free(temp_v);
	free(I_z_r);   free(I_z_c);   free(I_z_v);

	/////////////// Create C3 with quasi_periodic  boundary condtion ///////////////
	int*  K_3_r = (int*) calloc( 2*n3, sizeof(int) );
	int*  K_3_c = (int*) calloc( 2*n3, sizeof(int) );
	cmpx* K_3_v = (cmpx*)calloc( 2*n3, sizeof(cmpx));

	spMTX_FDM_1d_quasiperiodic( n3, mesh_len[2], i2pika3, K_3_r, K_3_c, K_3_v );

	int* I_yx_r = (int*)malloc( n1*n2*sizeof(int));
    int* I_yx_c = (int*)malloc( n1*n2*sizeof(int));
    int* I_yx_v = (int*)malloc( n1*n2*sizeof(int));
    // C_3  =  kron( K_z , kron( I_y , I_x ) );
	speye( n1*n2, I_yx_r, I_yx_c, I_yx_v );

	spkron_mm(   K_3_r , K_3_c , K_3_v ,    n3,  2*n3,
		         I_yx_r, I_yx_c, I_yx_v, n1*n2, n1*n2,
		         C3_r  , C3_c  , C3_v  );

	free(K_3_r);   free(K_3_c);   free(K_3_v);
	free(I_yx_r);  free(I_yx_c);  free(I_yx_v);
}
void FAME_Matrix_Curl_General( double* wave_vec,int* grid_num, double* edge_len, double* mesh_len, LATTICE lattice,
						       int* C1_r, int* C1_c, cmpx* C1_v,
						       int* C2_r, int* C2_c, cmpx* C2_v,
						       int* C3_r, int* C3_c, cmpx* C3_v )
{
	int n1 = grid_num[0], n2 = grid_num[1], n3 = grid_num[2];
	int i;

    cmpx i2pika1 = idouble_pi*dot(&lattice.lattice_vec_a[0], wave_vec, 3);
    cmpx i2pika2 = idouble_pi*dot(&lattice.lattice_vec_a[3], wave_vec, 3);
    cmpx i2pika3 = idouble_pi*dot(&lattice.lattice_vec_a[6], wave_vec, 3);

    cmpx i2pikt1 = idouble_pi*dot(lattice.lattice_constant.t1, wave_vec, 3);
    cmpx i2pikt2 = idouble_pi*dot(lattice.lattice_constant.t2, wave_vec, 3);
    cmpx i2pikt3 = idouble_pi*dot(lattice.lattice_constant.t3, wave_vec, 3);
    cmpx i2pikt4 = idouble_pi*dot(lattice.lattice_constant.t4, wave_vec, 3);

	/////////////// Create C1 with quasi_periodic  boundary condtion ///////////////
	int*  K_1_r = (int*) calloc( 2*n1, sizeof(int) );
	int*  K_1_c = (int*) calloc( 2*n1, sizeof(int) );
	cmpx* K_1_v = (cmpx*)calloc( 2*n1, sizeof(cmpx));

	spMTX_FDM_1d_quasiperiodic( n1, mesh_len[0], i2pika1, K_1_r, K_1_c, K_1_v );

	int* I_zy_r = (int*)malloc( n3*n2*sizeof(int));
    int* I_zy_c = (int*)malloc( n3*n2*sizeof(int));
    int* I_zy_v = (int*)malloc( n3*n2*sizeof(int));

	speye( n3*n2, I_zy_r, I_zy_c, I_zy_v );
	// C_1  =  kron( I_z , kron( I_y , K_1 ) );
	spkron_mm(  I_zy_r,  I_zy_c,  I_zy_v, n3*n2,   n3*n2,
		        K_1_r ,  K_1_c ,  K_1_v ,    n1,    2*n1,
		        C1_r  ,  C1_c  ,  C1_v   );

	free(K_1_r);  free(K_1_c);  free(K_1_v);
	free(I_zy_r); free(I_zy_c); free(I_zy_v);

	/////////////// Create C2 with quasi_periodic  boundary condtion ///////////////
	int*  K_2_r = (int*) calloc( 2*n2*n1, sizeof(int) );
	int*  K_2_c = (int*) calloc( 2*n2*n1, sizeof(int) );
	cmpx* K_2_v = (cmpx*)calloc( 2*n2*n1, sizeof(cmpx));

	spMTX_FDM_2d_quasiperiodic( n2, n1, mesh_len[1], 0.0, K_2_r, K_2_c, K_2_v );


	int rho_1 = 1;
	if(lattice.lattice_constant.theta_3 <= .5*pi)
		rho_1 = 0;
	// Assemble J2 matrix into K_2
	for( i = (2*n2*n1-n1); i < (2*n2*n1-n1) + lattice.lattice_constant.m1; i++ )
	{
		K_2_r[i] = i - (2*n2*n1-n1) + (n2*n1-n1);
    	K_2_c[i] = i - (2*n2*n1-n1) + (n1-lattice.lattice_constant.m1);
		K_2_v[i] = cexp( i2pika2 + i2pika1*(rho_1-1) )/mesh_len[1];
	}
	for( i = (2*n2*n1-n1) + lattice.lattice_constant.m1; i < 2*n2*n1; i++ )
	{
		K_2_r[i] = i - ((2*n2*n1-n1) + lattice.lattice_constant.m1) + (n2*n1-n1) + lattice.lattice_constant.m1;
    	K_2_c[i] = i - ((2*n2*n1-n1) + lattice.lattice_constant.m1);
		K_2_v[i] = cexp( i2pika2 + i2pika1*rho_1 )/mesh_len[1];
	}

    // C_2  =  kron( I_z , K_2 );
	int* I_z_r = (int*)malloc( n3*sizeof(int));
    int* I_z_c = (int*)malloc( n3*sizeof(int));
    int* I_z_v = (int*)malloc( n3*sizeof(int));


    speye( n3, I_z_r, I_z_c, I_z_v );

	spkron_mm(   I_z_r,   I_z_c,   I_z_v,    n3,      n3,
	  	         K_2_r,   K_2_c,   K_2_v, n1*n2, 2*n1*n2,
		         C2_r,    C2_c,    C2_v );

	free(K_2_r);   free(K_2_c);   free(K_2_v);
	free(I_z_r);   free(I_z_c);   free(I_z_v);

	/////////////// Create C3 with quasi_periodic  boundary condtion ///////////////
	spMTX_FDM_3d_quasiperiodic( n3, n2, n1, mesh_len[2], 0.0, C3_r, C3_c, C3_v );

	// Construct J3 matrix as [O , J31; J32, O]
	// Construct J31
	int*  temp_J31_r = (int*) malloc( n1*sizeof(int) );
    int*  temp_J31_c = (int*) malloc( n1*sizeof(int) );
    cmpx* temp_J31_v = (cmpx*)malloc( n1*sizeof(cmpx));

	//cout<<"lattice.lattice_constant.m3: "<<lattice.lattice_constant.m3<<endl;//M2=2,M3=1,m4=0
    for( i = 0; i < lattice.lattice_constant.m4; i++ )
    {
    	temp_J31_r[i] = i;
    	temp_J31_c[i] = i + n1 - lattice.lattice_constant.m4;
    	temp_J31_v[i] = cexp(i2pika3 + i2pikt3)/mesh_len[2];
    }
    for( i = lattice.lattice_constant.m4; i < n1; i++ )
    {
    	temp_J31_r[i] = i;
    	temp_J31_c[i] = i - lattice.lattice_constant.m4;
    	temp_J31_v[i] = cexp(i2pika3 + i2pikt4)/mesh_len[2];
    }
    int*  I_m3_r = (int*)malloc( lattice.lattice_constant.m3*sizeof(int));
    int*  I_m3_c = (int*)malloc( lattice.lattice_constant.m3*sizeof(int));
    int*  I_m3_v = (int*)malloc( lattice.lattice_constant.m3*sizeof(int));
    int*  J31_r = (int*) malloc( lattice.lattice_constant.m3*n1*sizeof(int) );
    int*  J31_c = (int*) malloc( lattice.lattice_constant.m3*n1*sizeof(int) );
    cmpx* J31_v = (cmpx*)malloc( lattice.lattice_constant.m3*n1*sizeof(cmpx));

    speye( lattice.lattice_constant.m3, I_m3_r, I_m3_c, I_m3_v );

    spkron_mm(  I_m3_r    ,  I_m3_c    ,  I_m3_v    , lattice.lattice_constant.m3,   lattice.lattice_constant.m3,
		        temp_J31_r,  temp_J31_c,  temp_J31_v,                          n1,                            n1,
		        J31_r     ,  J31_c     ,  J31_v     );

    free(I_m3_r); free(I_m3_c); free(I_m3_v); free(temp_J31_r); free(temp_J31_c); free(temp_J31_v);
    // Construct J32
    int*  temp_J32_r = (int*) malloc( n1*sizeof(int) );
    int*  temp_J32_c = (int*) malloc( n1*sizeof(int) );
    cmpx* temp_J32_v = (cmpx*)malloc( n1*sizeof(cmpx));
    for( i = 0; i < lattice.lattice_constant.m2; i++ )
    {
    	temp_J32_r[i] = i;
    	temp_J32_c[i] = i + n1 - lattice.lattice_constant.m2;
    	temp_J32_v[i] = cexp(i2pika3 + i2pikt2)/mesh_len[2];
    }
    for( i = lattice.lattice_constant.m2; i < n1; i++ )
    {
    	temp_J32_r[i] = i;
    	temp_J32_c[i] = i - lattice.lattice_constant.m2;
    	temp_J32_v[i] = cexp(i2pika3 + i2pikt1)/mesh_len[2];
    }

    int*  I_n2m3_r = (int*)malloc( (n2-lattice.lattice_constant.m3)*sizeof(int));
    int*  I_n2m3_c = (int*)malloc( (n2-lattice.lattice_constant.m3)*sizeof(int));
    int*  I_n2m3_v = (int*)malloc( (n2-lattice.lattice_constant.m3)*sizeof(int));
    int*  J32_r = (int*) malloc( (n2-lattice.lattice_constant.m3)*n1*sizeof(int) );
    int*  J32_c = (int*) malloc( (n2-lattice.lattice_constant.m3)*n1*sizeof(int) );
    cmpx* J32_v = (cmpx*)malloc( (n2-lattice.lattice_constant.m3)*n1*sizeof(cmpx));

    speye( n2-lattice.lattice_constant.m3, I_n2m3_r, I_n2m3_c, I_n2m3_v );

    spkron_mm(  I_n2m3_r  ,  I_n2m3_c  ,  I_n2m3_v  , n2-lattice.lattice_constant.m3, n2-lattice.lattice_constant.m3,
		        temp_J32_r,  temp_J32_c,  temp_J32_v, n1                            , n1                            ,
		        J32_r     ,  J32_c     ,  J32_v      );
    free(I_n2m3_r); free(I_n2m3_c); free(I_n2m3_v); free(temp_J32_r); free(temp_J32_c); free(temp_J32_v);
	// Assemblev J31 and J32 into C3
	for( i = 2*n3*n2*n1 - n2*n1; i < 2*n3*n2*n1 - n2*n1 + lattice.lattice_constant.m3*n1; i++)
	{
		C3_r[i] = J31_r[i-(2*n3*n2*n1 - n2*n1)] + n3*n2*n1 - n2*n1;
		C3_c[i] = J31_c[i-(2*n3*n2*n1 - n2*n1)] + (n2-lattice.lattice_constant.m3)*n1;
		C3_v[i] = J31_v[i-(2*n3*n2*n1 - n2*n1)];
	}
	for( i = 2*n3*n2*n1 - n2*n1 + lattice.lattice_constant.m3*n1; i < 2*n3*n2*n1; i++)
	{
		C3_r[i] = J32_r[i-(2*n3*n2*n1 - n2*n1 + lattice.lattice_constant.m3*n1)] + n3*n2*n1 - n2*n1 + lattice.lattice_constant.m3*n1;
		C3_c[i] = J32_c[i-(2*n3*n2*n1 - n2*n1 + lattice.lattice_constant.m3*n1)];
		C3_v[i] = J32_v[i-(2*n3*n2*n1 - n2*n1 + lattice.lattice_constant.m3*n1)];
	}
	free(J31_r);    free(J31_c);    free(J31_v);    free(J32_r);      free(J32_c);      free(J32_v);
}


void spMTX_FDM_1d_quasiperiodic(int dim, double mesh_len, cmpx theta, int* K_row, int* K_col, cmpx* K_val )
{
	double temp = 1.0/mesh_len;

	for( int i = 0; i < dim; i++ )
	{
	    K_row[i] =  i;
	    K_col[i] =  i;
	    K_val[i] = -temp;
	}
	for( int i = dim; i < (2*dim-1); i++ )
	{
	    K_row[i] = i - dim;
        K_col[i] = i - dim + 1;
        K_val[i] = temp;
	}
	K_row[2*dim - 1] = dim - 1;
    K_col[2*dim - 1] = 0;
	K_val[2*dim - 1] = temp*cexp(theta);
}
void spMTX_FDM_2d_quasiperiodic(int dim2, int dim1, double mesh_len, cmpx theta, int* K_row, int* K_col, cmpx* K_val )
{
	double temp = 1.0/mesh_len;

	for( int i = 0; i < dim2*dim1; i++ )
	{
	    K_row[i] =  i;
	    K_col[i] =  i;
	    K_val[i] = -temp;
	}
	for( int i = dim2*dim1; i < (2*dim2*dim1-dim1); i++ )
	{
	    K_row[i] = i - dim2*dim1;
        K_col[i] = i - dim2*dim1 + dim1;
        K_val[i] = temp;
	}
	for( int i = (2*dim2*dim1-dim1); i < 2*dim2*dim1; i++ )
	{
		K_row[i] = i - (2*dim2*dim1-dim1) + (dim2*dim1-dim1);
    	K_col[i] = i - (2*dim2*dim1-dim1);
		K_val[i] = temp*cexp(theta);
	}
}
void spMTX_FDM_3d_quasiperiodic(int dim3, int dim2, int dim1, double mesh_len, cmpx theta, int* K_row, int* K_col, cmpx* K_val )
{
	double temp = 1.0/mesh_len;

	for( int i = 0; i < dim3*dim2*dim1; i++ )
	{
	    K_row[i] =  i;
	    K_col[i] =  i;
	    K_val[i] = -temp;
	}
	for( int i = dim3*dim2*dim1; i < (2*dim3*dim2*dim1-dim2*dim1); i++ )
	{
	    K_row[i] = i - dim3*dim2*dim1;
        K_col[i] = i - dim3*dim2*dim1 + dim2*dim1;
        K_val[i] = temp;
	}
	for( int i = (2*dim3*dim2*dim1-dim2*dim1); i < 2*dim3*dim2*dim1; i++ )
	{
		K_row[i] = i - (2*dim3*dim2*dim1-dim2*dim1) + (dim3*dim2*dim1-dim2*dim1);
    	K_col[i] = i - (2*dim3*dim2*dim1-dim2*dim1);
		K_val[i] = temp*cexp(theta);
	}
}
