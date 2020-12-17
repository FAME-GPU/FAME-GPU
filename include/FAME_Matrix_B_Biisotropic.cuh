#ifndef _FAME_MATRIX_B_BIISOTROPIC_H_
#define _FAME_MATRIX_B_BIISOTROPIC_H_
int FAME_Matrix_B_Biisotropic(int n, MATERIAL material, double* B_eps, double* B_mu, cuDoubleComplex* B_xi, cuDoubleComplex* B_zeta, cuDoubleComplex* B_zeta_s, double* inv_Phi);
#endif
