#ifndef _DX9CUDARENDER_H
#define _DX9CUDARENDER_H
#include <cuda_runtime.h>
#include <cuda_d3d9_interop.h>
#include <tchar.h>
#include <stdio.h>

#pragma comment(lib,"d3d9.lib")

inline bool TestError(LONG err, const char* msg)
{
	if(err==0) return true;
	printf("Error %lu (0x08lX) : %s\n",err & 0x0000FFFF, err, msg);
	return false;
}

class DX9_ShowCuda
{
  static const DWORD MAX_SURF=10;
public:
	cudaDeviceProp GPUprops;
  D3DDISPLAYMODE dmode;
  IDirect3D9* pD3D;
  IDirect3DDevice9*  pd3dDevice; // Rendering device
  IDirect3DSurface9 *pBB, *pPS[MAX_SURF];  // BackBuffer & PlainSurface
  DWORD dimx[MAX_SURF], dimy[MAX_SURF];

	bool IsValid() {return pBB!=0;}
	UINT GetDimX() {return dmode.Width;}
	UINT GetDimY() {return dmode.Height;}
  DX9_ShowCuda(UINT adapter=0): pD3D(0), pd3dDevice(0), pBB(0)
  {
		//printf("Debut Ctor DX9_ShowCuda\n");
    memset(pPS,0,sizeof(pPS[0])*MAX_SURF);
	  TestError(cudaSetDevice(0),"cudaSetDevice");
	  TestError(cudaSetDeviceFlags(cudaDeviceBlockingSync),"cudaSetDeviceFlags");
	  TestError(cudaGetDeviceProperties(&GPUprops,0),"cudaGetDeviceProperties");
    pD3D = Direct3DCreate9(D3D_SDK_VERSION);
    if(!pD3D) return;
		//printf("OK creation DX9\n");
    pD3D->GetAdapterDisplayMode(adapter,&dmode);
    D3DPRESENT_PARAMETERS d3dpp; // Set up the structure used to create the D3DDevice
    ZeroMemory( &d3dpp, sizeof(d3dpp) );
    d3dpp.Windowed = FALSE;
    d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
    d3dpp.BackBufferFormat = dmode.Format; //D3DFMT_X8R8G8B8;
    d3dpp.BackBufferCount = 1;
    d3dpp.BackBufferWidth=dmode.Width;
    d3dpp.BackBufferHeight=dmode.Height;
    //d3dpp.Flags=0;
    d3dpp.Flags=D3DPRESENTFLAG_LOCKABLE_BACKBUFFER; // if we want DC on backbuff

    WNDCLASS wcls={CS_NOCLOSE,::DefWindowProc,0,0,0,0,0,0,0,_T("WMain")};
    HWND hWnd=::CreateWindow((LPCTSTR)RegisterClass(&wcls),0,WS_POPUP | WS_MAXIMIZE | WS_VISIBLE,0,0,dmode.Width,dmode.Height,0,0,0,0);
    if(pD3D->CreateDevice(adapter, D3DDEVTYPE_HAL, hWnd, D3DCREATE_HARDWARE_VERTEXPROCESSING /*|D3DCREATE_NOWINDOWCHANGES*/,
			&d3dpp, &pd3dDevice)!=D3D_OK) { printf("Can't create device !\n"); return; }
		//printf("OK device\n");
    TestError(cudaD3D9SetDirect3DDevice(pd3dDevice),"cudaD3D9SetDirect3DDevice");
    TestError(pd3dDevice->TestCooperativeLevel(),"TestCooperativeLevel");
    if(!TestError(pd3dDevice->GetBackBuffer(0,0,D3DBACKBUFFER_TYPE_MONO,&pBB),"GetBackBuffer")) return;
    //printf("OK3 Ctor Dx9CudaRender\n");
  }
  DWORD CreateSurf(DWORD _dimx, DWORD _dimy)
  {
    DWORD idx=0;
    while(pPS[idx] && idx<MAX_SURF) idx++;
    if(idx>=MAX_SURF) return 0xFFFFFFFF;
    dimx[idx]=_dimx; dimy[idx]=_dimy;
    if(pd3dDevice->CreateOffscreenPlainSurface(dimx[idx],dimy[idx],D3DFMT_X8R8G8B8/*D3DFMT_UYVY*//*D3DFMT_YUY2*//*D3DFMT_L8*/,
      D3DPOOL_DEFAULT,pPS+idx,0)!=D3D_OK) { printf("Error: can't create offscreen surface\n"); return 0xFFFFFFFF; }
    TestError(cudaD3D9RegisterResource(pPS[idx], cudaD3D9RegisterFlagsNone),"RegisterResource");
    TestError(cudaD3D9ResourceSetMapFlags(pPS[idx],cudaD3D9MapFlagsNone),"SetMapFlags");
    return idx;
  }
  void* MapSurf(DWORD idx, size_t* psz, size_t* ppitch=0)
  {
    if(!TestError(cudaD3D9MapResources(1,(IDirect3DResource9**)(pPS+idx)),"MapResources")) return 0;
    TestError(cudaD3D9ResourceGetMappedSize(psz,pPS[idx],0,0),"GetMappedSize");
    void* ptr=0;
    TestError(cudaD3D9ResourceGetMappedPointer(&ptr,pPS[idx],0,0),"D3D9_GetMappedPointer");
    TestError(cudaD3D9ResourceGetMappedPitch(ppitch,0,pPS[idx],0,0),"D3D9_GetMappedPitch");
    return ptr;
  }
  void UnmapSurf(DWORD idx)
  {
    TestError(cudaD3D9UnmapResources(1,(IDirect3DResource9**)(pPS+idx)),"UnmapResources");
  }
  void Copy(DWORD idx, const RECT* pdst=0)
  {
		//RECT r={0,0,1920,1080};
    pd3dDevice->StretchRect(pPS[idx],0,pBB,pdst,D3DTEXF_LINEAR);
  }
  ~DX9_ShowCuda()
  {
    for(DWORD i=0; i<MAX_SURF; i++)
    {
      if(pPS[i])
      {
        cudaD3D9UnregisterResource(pPS[i]);
        pPS[i]->Release();
      }
    }
    if(pBB) pBB->Release();
    if(pd3dDevice) pd3dDevice->Release();
    if(pD3D) pD3D->Release();
  }
  HDC GetDC() { HDC dc=0; pBB->GetDC(&dc); return dc; }
  void ReleaseDC(HDC dc) {pBB->ReleaseDC(dc);}
  void Present() {pd3dDevice->Present(0,0,0,0);}
  void Clear(DWORD clr=0x000000) {pd3dDevice->Clear(0,0,D3DCLEAR_TARGET,clr,0,0);}
};

#endif // _DX9CUDARENDER_H
