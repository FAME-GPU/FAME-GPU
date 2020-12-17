#ifndef _LANCZOS_LOCKPURGE_H_
#define _LANCZOS_LOCKPURGE_H_

int Lanczos_LockPurge(CULIB_HANDLES cuHandles,  LANCZOS_BUFFER *lBuffer, cuDoubleComplex* ev, 
    int mNwant, int Nstep, int size);


#endif