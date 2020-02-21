#include "kernels.h"
#include <cutil.h>
#define MEDIA_DIR "R:\\Polytech-Echange\\ees20\\DxGPU\\cuda.Addon\\"

/************************ test1 ***********************/
void test1()
{
  const size_t VECT_SZ=1024*1024+7,VECT_SZ_BYTE=VECT_SZ*sizeof(float);
  //alloc Host(CPU) data
  float *h_A=0,*h_B=0,*h_C=0;
  //h_A=new float[VECT_SZ];
  //h_C=new float[VECT_SZ];
  //h_B=new float[VECT_SZ];
  cudaMallocHost(&h_A,VECT_SZ_BYTE); 
  cudaMallocHost(&h_B,VECT_SZ_BYTE); 
  cudaMallocHost(&h_C,VECT_SZ_BYTE); 
  for(size_t i=0; i<VECT_SZ; i++) 
  { 
    h_A[i]=float(i)*float(i); 
    h_B[i]=sqrt(float(i)); 
    h_C[i]=0; 
  }
  // alloc Device(GPU) data :
  float *d_A=0, *d_B=0, *d_C=0;
  cudaMalloc ((void**)&d_A,VECT_SZ_BYTE);
  cudaMalloc ((void**)&d_B,VECT_SZ_BYTE);
  cudaMalloc ((void**)&d_C,VECT_SZ_BYTE);
  //show addresses
  printf("CPU : %p %p %p \nGPU : %p %p %p \n",h_A,h_B,h_C,d_A,d_B,d_C);
  //CPU->GPU data transfer
  cudaMemcpy (d_A,h_A,VECT_SZ_BYTE, cudaMemcpyHostToDevice);
  cudaMemcpy (d_B,h_B,VECT_SZ_BYTE, cudaMemcpyHostToDevice);
  //kernel launch
  LaunchVectAdd(d_A,d_B,d_C,VECT_SZ);
  cudaMemcpy (h_C,d_C,VECT_SZ_BYTE, cudaMemcpyDeviceToHost);
  // visual test:
  for(size_t i=0;i<10;i++)
    printf("[%9u] %18.3f+%18.3f=%18.3f(%18.3f)\n",i,h_A[i],h_B[i],h_C[i],h_A[i]+h_B[i]);
  for(size_t i=VECT_SZ-10; i<VECT_SZ ; i++)
    printf("[%9u] %18.3f +%18.3f = %18.3f (%18.3f)\n",i,h_A[i],h_B[i],h_C[i],h_A[i]+h_B[i]);
 //free all CPU/GPU
  cudaFree(d_A);cudaFree(d_B);cudaFree(d_C);
  cudaFreeHost(h_A);cudaFreeHost(h_B);cudaFreeHost(h_C);
  //delete[] h_A;delete[] h_B;delete[] h_C;

}
void test2()
{
  float *h_i1=0, *d_i1=0;   // test image
  unsigned int dimx=0, dimy=0;
  if(!cutil::LoadPGMf("lena.pgm",&h_i1,&dimx,&dimy))
  {
    printf("erreur de chargement d'image");
      return;
  }
  unsigned int sz_img=dimx*dimy*sizeof(float);//size en BYTE
  cudaMalloc((void**)&d_i1,sz_img);
  printf("Image %u x %u pixels, @CPU: %p GPU: %p\n",dimx,dimy,h_i1,d_i1);
  cudaMemcpy(d_i1,h_i1,sz_img,cudaMemcpyHostToDevice); // CPU->GPU
  LaunchProcImg_Inv(d_i1,dimx,dimy,dimx*sizeof(float)); // kernel launch (pitch in bytes)
  cudaMemcpy(h_i1,d_i1,sz_img,cudaMemcpyDeviceToHost); // GPU->CPU
  cutil::SavePGMf("lena_inv.pgm",h_i1,dimx,dimy);
  cudaFree(d_i1);//free gpu
  cutil::Free(h_i1);//free cpu

    return;

}

/************************ test2b ***********************/
void test2b()
{
  float *h_i1=0, *d_i1=0;   // test image
  unsigned int dimx=0, dimy=0;
  if(!cutil::LoadPGMf("lena.pgm",&h_i1,&dimx,&dimy))
  {
    printf("erreur de chargement d'image");
      return;
  }
  int pitch = dimx*sizeof(float);
  unsigned int sz_img=dimx*dimy*sizeof(float);//size en BYTE
  cudaMalloc((void**)&d_i1,sz_img);
  printf("Image %u x %u pixels, @CPU: %p GPU: %p\n",dimx,dimy,h_i1,d_i1);
  //			***			inf-droit			***
  cudaMemcpy(d_i1,h_i1,sz_img,cudaMemcpyHostToDevice); // CPU->GPU
  LaunchProcImg_Inv(d_i1+(dimy/2)*pitch/sizeof(float)+dimx/2,dimx/2,dimy/2,pitch); // kernel launch (pitch in bytes) quart inf-droit
  cudaMemcpy(h_i1,d_i1,sz_img,cudaMemcpyDeviceToHost); // GPU->CPU
  cutil::SavePGMf("lena_inv_inf-droit.pgm",h_i1,dimx,dimy);
  //			***			sup-gauche			***
  cutil::LoadPGMf("lena.pgm",&h_i1,&dimx,&dimy);
  cudaMemcpy(d_i1,h_i1,sz_img,cudaMemcpyHostToDevice); // CPU->GPU
  LaunchProcImg_Inv(d_i1,dimx/2,dimy/2,dimx*sizeof(float)); // kernel launch (pitch in bytes) quart sup-gauche
  cudaMemcpy(h_i1,d_i1,sz_img,cudaMemcpyDeviceToHost); // GPU->CPU
  cutil::SavePGMf("lena_inv_sup-gauche.pgm",h_i1,dimx,dimy);
  cudaFree(d_i1);//free gpu
  cutil::Free(h_i1);//free cpu
}
template <const uchar CH>__global__ void KProcImg_Gray2Red(uchar*d_i1,size_t pitch1,uchar4*d_i2,size_t pitch2,size_t dimx,size_t dimy)
{
  size_t x=threadIdx.x+blockDim.x*blockIdx.x;
  size_t y=threadIdx.y+blockDim.y*blockIdx.y;
  if(x<dimx&&y<dimy)
    ((unsigned int*)d_i2)[x+y*pitch2]=((unsigned int)d_i1[x+y*pitch1])<<(CH*8);
}

void test2d()
{
  float *h_i1=0, *d_i1=0;   // test image
  unsigned int dimx=0, dimy=0;
  if(!cutil::LoadPGMf("baboon_odd.pgm",&h_i1,&dimx,&dimy))
  {
    printf("erreur de chargement d'image");
      return;
  }
  size_t dpitch = dimx*sizeof(float),gpitch;
  unsigned int sz_img=dimx*dimy*sizeof(float);//size en BYTE
  cudaMallocPitch((void**)&d_i1,&gpitch,dpitch,dimy);
  printf("Image %u x %u pixels, @CPU: %p,dpitch: %d;  GPU: %p,gpitch: %d; \n",dimx,dimy,h_i1,dpitch,d_i1,gpitch);
  //			***			inf-droit			***
  cudaMemcpy2D(d_i1,gpitch,h_i1,dpitch,dimx*sizeof(float),dimy,cudaMemcpyHostToDevice);
  LaunchProcImg_Inv(d_i1+(dimy/2)*gpitch/sizeof(float)+dimx/2,dimx/2,dimy/2,gpitch); // kernel launch (pitch in bytes) quart inf-droit
  //LaunchProcImg_Inv(d_i1,dimx/2,dimy/2,gpitch); //kernel launch (pitch in bytes) quart sup-gauche
  cudaMemcpy2D(h_i1,dpitch,d_i1,gpitch,dimx*sizeof(float),dimy,cudaMemcpyDeviceToHost);
  cutil::SavePGMf("lena_inv_inf-droit2D.pgm",h_i1,dimx,dimy);
  //			***			sup-gauche			***
  cutil::LoadPGMf("baboon_odd.pgm",&h_i1,&dimx,&dimy);
  cudaMemcpy2D(d_i1,gpitch,h_i1,dpitch,dimx*sizeof(float),dimy,cudaMemcpyHostToDevice);
  LaunchProcImg_Inv(d_i1,dimx/2,dimy/2,gpitch); //kernel launch (pitch in bytes) quart sup-gauche
  cudaMemcpy2D(h_i1,dpitch,d_i1,gpitch,dimx*sizeof(float),dimy,cudaMemcpyDeviceToHost);
  cutil::SavePGMf("lena_inv_sup-gauche2D.pgm",h_i1,dimx,dimy);
  cudaFree(d_i1);//free gpu
  cutil::Free(h_i1);//free cpu

}
/************************ test3 ***********************/

void test3()
{
  uchar *h_i1=0,*d_i1=0;
  uchar4 *h_i2=0,*d_i2=0;
  unsigned int dimx=0,dimy=0;
  if(!cutil::LoadPGMub("baboon_odd.pgm",&h_i1,&dimx,&dimy))
  {
      printf("erreur de chargement d'image");
      return;
  }
  size_t pitch1=0,pitch2=0,sz_img2=dimx*dimy*sizeof(uchar4);
  cudaMallocHost(&h_i2,sz_img2);
  cudaMallocPitch((void**)&d_i1,&pitch1,dimx*sizeof(uchar),dimy);
  cudaMallocPitch((void**)&d_i2,&pitch2,dimx*sizeof(uchar4),dimy);
  printf("Image %u x %u pixels, @CPU: %p, %p GPU: %p, %p, Pitch gpu=%u,%u\n",dimx,dimy,h_i1,h_i2,d_i1,d_i2,pitch1,pitch2);
  cudaMemcpy2D(d_i1,pitch1,h_i1,dimx*sizeof(uchar),dimx*sizeof(uchar),dimy,cudaMemcpyHostToDevice); // CPU->GPU data transfer
  LaunchProcImg_Gray2Red(d_i1,pitch1,d_i2,pitch2,dimx,dimy); // kernel launch (pitch in bytes)
  cudaMemcpy2D(h_i2,dimx*sizeof(uchar4),d_i2,pitch2,dimx*sizeof(uchar4),dimy,cudaMemcpyDeviceToHost); // CPU->GPU data transfer
  cutil::SavePPM4ub("img_red.ppm",(uchar*)h_i2,dimx,dimy);
  cudaFree(d_i1);
  cutil::Free(h_i1);
  cudaFree(d_i2);
  cudaFreeHost(h_i2);
}

/************************ test3.beta ***********************/
void test3b()
{
  uchar *h_i1=0,*d_i1=0;
  uchar4 *h_i2=0,*d_i2=0;
  unsigned int dimx=0,dimy=0;
  if(!cutil::LoadPGMub("baboon_odd.pgm",&h_i1,&dimx,&dimy))
  {
      printf("erreur de chargement d'image");
      return;
  }
  size_t pitch1=0,pitch2=0,sz_img2=dimx*dimy*sizeof(uchar4);
  cudaMallocHost(&h_i2,sz_img2);
  cudaMallocPitch((void**)&d_i1,&pitch1,dimx*sizeof(uchar),dimy);
  cudaMallocPitch((void**)&d_i2,&pitch2,dimx*sizeof(uchar4),dimy);
  printf("Image %u x %u pixels, @CPU: %p, %p GPU: %p, %p, Pitch gpu=%u,%u\n",dimx,dimy,h_i1,h_i2,d_i1,d_i2,pitch1,pitch2);
  InitPal();
  cudaMemcpy2D(d_i1,pitch1,h_i1,dimx*sizeof(uchar),dimx*sizeof(uchar),dimy,cudaMemcpyHostToDevice); // CPU->GPU data transfer
  LaunchProcImg_Gray2Pal(d_i1,pitch1,d_i2,pitch2,dimx,dimy); // kernel launch (pitch in bytes)
  cudaMemcpy2D(h_i2,dimx*sizeof(uchar4),d_i2,pitch2,dimx*sizeof(uchar4),dimy,cudaMemcpyDeviceToHost); // CPU->GPU data transfer
  cutil::SavePPM4ub("img_pal_3b.ppm",(uchar*)h_i2,dimx,dimy);
  cudaFree(d_i1);
  cutil::Free(h_i1);
  cudaFree(d_i2);
  cudaFreeHost(h_i2);
}
/************************ test4 ***********************/
void test4()
{
  uchar *h_i1=0;
  cudaArray* d_arr=0;
  uchar4 *h_i2=0,*d_i2=0;
  unsigned int dimx=0,dimy=0;
  if(!cutil::LoadPGMub("baboon_odd.pgm",&h_i1,&dimx,&dimy))
  {
      printf("erreur de chargement d'image");
      return;
  }
  cudaChannelFormatDesc pixdesc =  {8,0,0,0,cudaChannelFormatKindUnsigned};
  cudaMallocArray(&d_arr,&pixdesc,dimx,dimy,cudaArraySurfaceLoadStore);
  BindSurf1(d_arr);
  size_t pitch2=0,sz_img2=dimx*dimy*sizeof(uchar4);
  cudaMallocHost(&h_i2,sz_img2);
  cudaMallocPitch((void**)&d_i2,&pitch2,dimx*sizeof(uchar4),dimy);
  printf("Image %u x %u pixels, @CPU: %p, %p  GPU: %p,  Pitch gpu=%u\n",dimx,dimy,h_i1,h_i2,d_i2,pitch2);
  InitPal();
  cudaMemcpy2DToArray(d_arr,0,0,h_i1,dimx*sizeof(uchar),dimx*sizeof(uchar),dimy,cudaMemcpyHostToDevice); // CPU->GPU data transfer

  LaunchProcImg_Surf2Pal(d_i2,pitch2,dimx,dimy); // kernel launch (pitch in bytes)
  cudaMemcpy2D(h_i2,dimx*sizeof(uchar4),d_i2,pitch2,dimx*sizeof(uchar4),dimy,cudaMemcpyDeviceToHost); // CPU->GPU data transfer
  cutil::SavePPM4ub("img_palArr.ppm",(uchar*)h_i2,dimx,dimy);
  cudaFreeArray(d_arr);
  cutil::Free(h_i1);
  cudaFree(d_i2);
  cudaFreeHost(h_i2);
}
void main()
{
    cudaSetDevice(0);
    //test1();
    //test2();
    //test2b();
	//test2d();
    //test3();
    //test3b();
    test4();
    cudaDeviceReset();
}