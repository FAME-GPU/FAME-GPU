#ifndef _FAME_CREATE_BUFFER_H_
#define	_FAME_CREATE_BUFFER_H_

int FAME_Create_Buffer(
CULIB_HANDLES*  cuHandles, 
FFT_BUFFER*     fft_buffer, 
LANCZOS_BUFFER* lBuffer, 
int N, int Nstep);

#endif