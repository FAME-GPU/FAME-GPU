#ifndef _FAME_CREATE_BUFFER_ANISOTROPIC_H_
#define	_FAME_CREATE_BUFFER_ANISOTROPIC_H_

int FAME_Create_Buffer_Anisotropic(
CULIB_HANDLES*  cuHandles, 
FFT_BUFFER*     fft_buffer, 
LANCZOS_BUFFER* lBuffer, 
int N, int Nstep);

#endif