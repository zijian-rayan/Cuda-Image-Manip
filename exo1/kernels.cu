#include "kernels.h"
#define PAL_SIZE 256

/************************ GPU ***********************/
__constant__ uchar4 pal_1[PAL_SIZE]={{128,128,128,128}};
surface<void,cudaSurfaceType2D> surf_1;

/************************ test1 ***********************/
__global__ void KVectAddFast(const float* A, const float* B, float* C)
{
  size_t idx=threadIdx.x+blockDim.x*blockIdx.x;
  C[idx]=A[idx]+B[idx];
}

/************************ test2 ***********************/
__global__ void KVectAdd(const float* A, const float* B, float* C, size_t vsize)
{
  size_t idx=threadIdx.x+blockDim.x*blockIdx.x;
  if(idx<vsize)C[idx]=A[idx]+B[idx];
}
__global__ void KProcImg_Inv(float* I,size_t dimx,size_t dimy, size_t pitch)
{
  size_t x=threadIdx.x+blockDim.x*blockIdx.x;
  size_t y=threadIdx.y+blockDim.y*blockIdx.y;
  if(x<dimx&&y<dimy)
    I[x+y*pitch]=1-I[x+y*pitch];
}

/************************ test3 ***********************/
template <const uchar CH>__global__ void KProcImg_Gray2Red(uchar*d_i1,size_t pitch1,uchar4*d_i2,size_t pitch2,size_t dimx,size_t dimy)
{
  size_t x=threadIdx.x+blockDim.x*blockIdx.x;
  size_t y=threadIdx.y+blockDim.y*blockIdx.y;
  if(x<dimx&&y<dimy)
    ((unsigned int*)d_i2)[x+y*pitch2]=((unsigned int)d_i1[x+y*pitch1])<<(CH*8);
}
/************************ test3.beta ***********************/
__global__ void KProcImg_Gray2Pal(uchar*d_i1,size_t pitch1,uchar4*d_i2,size_t pitch2,size_t dimx,size_t dimy)
{
  size_t x=threadIdx.x+blockDim.x*blockIdx.x;
  size_t y=threadIdx.y+blockDim.y*blockIdx.y;
  if(x<dimx&&y<dimy)
    d_i2[x+y*pitch2]=pal_1[d_i1[x+y*pitch1]];
}

/************************ test4 ***********************/

__global__ void KProcImg_Surf2Pal(uchar4* i2, size_t pitch2, size_t dimx, size_t dimy)
{
	size_t x=threadIdx.x+blockIdx.x*blockDim.x;
	size_t y=threadIdx.y+blockIdx.y*blockDim.y;
	if(x<dimx && y<dimy)
	{
		uchar idx; 
		surf2Dread(&idx,surf_1,x*sizeof(uchar),y,cudaBoundaryModeClamp);
		i2[x+y*pitch2]=pal_1[idx];
	}
}
/************************ CPU ***********************/


/************************ test1 ***********************/
bool LaunchVectAdd(const float* d_A, const float* d_B, float* d_C, size_t nsize)
{
  const int BLOCK_SIZE =256;// x32:entre 32 - 1024
  
  if(nsize%BLOCK_SIZE==0)
  {
    dim3 Db(BLOCK_SIZE,1)/* 1 dimention */,Dg(nsize/BLOCK_SIZE,1,1);//Db(BLOCK_SZ), Dg(nsize/BLOCK_SZ)
    KVectAddFast<<<Dg,Db>>>(d_A,d_B,d_C);
  }
  else
  {
    dim3 Db(BLOCK_SIZE,1)/* 1 dimention */,Dg((nsize-1)/BLOCK_SIZE+1,1,1);//Db(BLOCK_SZ), Dg((nsize-1)/BLOCK_SIZE+1
    KVectAdd<<<Dg,Db>>>(d_A,d_B,d_C,nsize);
  }
  return TestLastError("Launch kernel KVectAdd");
}

/************************ test2 ***********************/
bool LaunchProcImg_Inv(float* d_il,size_t dimx,size_t dimy, size_t pitch_in_byte)
{
  const int BSZ_X =16,BSZ_Y=16;
  dim3 Db(BSZ_X,BSZ_Y,1),Dg((dimx+BSZ_X-1)/BSZ_X,(dimy+BSZ_Y-1)/BSZ_Y,1);
  KProcImg_Inv<<<Dg,Db>>>(d_il,dimx,dimy,pitch_in_byte/sizeof(float));
  return TestLastError("Launch kernel KProcImg_Inv");
}

/************************ test3 ***********************/

bool LaunchProcImg_Gray2Red(uchar* d_i1,size_t pitch1,uchar4*d_i2,size_t pitch2,size_t dimx,size_t dimy)
{
  //const int BSZ_X =256,BSZ_Y=1;//6.5  us
  //const int BSZ_X =1,BSZ_Y=256;//40   us
  //const int BSZ_X =16,BSZ_Y=16;//5.6  us
  const int BSZ_X =8,BSZ_Y=8;//5.6  us
  dim3 Db(BSZ_X,BSZ_Y,1),Dg((dimx+BSZ_X-1)/BSZ_X,(dimy+BSZ_Y-1)/BSZ_Y,1);
  KProcImg_Gray2Red<0><<<Dg,Db>>>(d_i1,pitch1/sizeof(uchar),d_i2,pitch2/sizeof(uchar4),dimx,dimy);
  return TestLastError("Launch kernel KProcImg_Gray2Red");
}
/************************ test3.beta ***********************/
void InitPal()
{
  uchar4 pal[PAL_SIZE], *p=pal;
  for(int i=0; i<32; i++, p++) {p->x = 0; p->y = (i<<3); p->z = 255; p->w=0;}
  for(int i=0; i<32; i++, p++) {p->x = 0; p->y = 255; p->z = 255-(i<<3); p->w=0;}
  for(int i=0; i<64; i++, p++) {p->x = (i<<2); p->y = 255; p->z = 0; p->w=0;}
  for(int i=0; i<128; i++, p++) {p->x = 255; p->y = 255-(i<<1); p->z = 0; p->w=0;}
  cudaMemcpyToSymbol(pal_1,pal,sizeof(uchar4)*PAL_SIZE,0,cudaMemcpyHostToDevice); // CUDA > 5.0
}
bool LaunchProcImg_Gray2Pal(uchar* d_i1,size_t pitch1,uchar4*d_i2,size_t pitch2,size_t dimx,size_t dimy)
{
  const int BSZ_X =16,BSZ_Y=16;
  dim3 Db(BSZ_X,BSZ_Y,1),Dg((dimx+BSZ_X-1)/BSZ_X,(dimy+BSZ_Y-1)/BSZ_Y,1);
  KProcImg_Gray2Pal<<<Dg,Db>>>(d_i1,pitch1/sizeof(uchar),d_i2,pitch2/sizeof(uchar4),dimx,dimy);
  return TestLastError("Launch kernel KProcImg_Gray2Red");
}
/************************ test4 ***********************/
bool BindSurf1(const cudaArray* d_arr)
{
	cudaBindSurfaceToArray(surf_1,d_arr);
	return TestLastError("Launch kernel BindSurf1");
}

bool LaunchProcImg_Surf2Pal(uchar4* d_i2, size_t pitch2, size_t dimx, size_t dimy)
{
	const int BSZ_X=16, BSZ_Y=16; 
	dim3 Db(BSZ_X,BSZ_Y,1),Dg((dimx+BSZ_X-1)/BSZ_X,(dimy+BSZ_Y-1)/BSZ_Y,1);
	KProcImg_Surf2Pal<<<Dg,Db>>>(d_i2,pitch2/sizeof(uchar4),dimx,dimy);
    return TestLastError("Launch kernel KProcImg_Surf2Pal");

}
	
