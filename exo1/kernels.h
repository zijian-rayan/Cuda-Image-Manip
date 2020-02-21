#ifndef _KERNELS_H_
#define _KERNELS_H_
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h> // Cuda C++ API

typedef unsigned char uchar;
bool LaunchVectAdd(const float* d_A, const float* d_B, float* d_C, size_t vsize);// d_A : device_A
bool LaunchProcImg_Inv(float* d_il,size_t dimx,size_t dimy, size_t pitch_in_byte); 
bool LaunchProcImg_Gray2Red(uchar* d_i1,size_t pitch1,uchar4*d_i2,size_t pitch2,size_t dimx,size_t dimy); // kernel launch (pitch in bytes)
bool LaunchProcImg_Gray2Pal(uchar* d_i1,size_t pitch1,uchar4*d_i2,size_t pitch2,size_t dimx,size_t dimy); // kernel launch (pitch in bytes)
void InitPal();
bool BindSurf1(const cudaArray* d_arr);
bool LaunchProcImg_Surf2Pal(uchar4* d_i2, size_t pitch2, size_t dimx, size_t dimy);

inline bool TestLastError(const char* msg="???")
{
  cudaError_t err=cudaGetLastError();
  if(err==cudaSuccess)return true;
   printf("Error \"%s\" in %s\n",cudaGetErrorString(err),msg);
  return false;
}
#endif