#include <Windows.h>
#include "../../shared_folder/cuda_dll_module.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "cuda_calculate_class.h"

#include <time.h>

cuda_calculate_class *cdl=0;

//MATHFUNCSDLL_API double ShortBeep(double x,double y)
//{
//	char s[10] = "qwerty";
//    clock_t t1 = clock(); // время до
//	 Init_CUDA(2048,2048);
//	
//	 clock_t t2 = clock(); // время после
//	//printf("%lf", (double) (t2-t1) / (double)CLOCKS_PER_SEC ); // время в секундах
//	 return cdl->test(x,y); 
//}
MATHFUNCSDLL_API void Init_CUDA(int N_1out,int N_2out)
{
	if(cdl != 0) cdl=new cuda_calculate_class();
	cdl->init_plan(N_1out,N_2out);
}
MATHFUNCSDLL_API void Free_CUDA()
{
	cdl->cuda_free();
}
//MATHFUNCSDLL_API void Transform_CUDA(int nCols,int nRows,double *zArray)
//{
//	int a=cdl->Transform(nCols,nRows,(Complex*)zArray);
//}
MATHFUNCSDLL_API void Cuda_ConvertZ2Z(int nCols,int nRows,double FI0,double *zArray)
{
	cdl->Cuda_ConvertZ2Z(nCols,nRows,FI0,(Complex*)zArray);
}
//
//MATHFUNCSDLL_API double get_kaiser_CUDA(double alpha, int n, int N)
//{
//	return cdl->get_kaiser(alpha,n,N);
//}
//MATHFUNCSDLL_API double get_h_CUDA(double alpha,int n, int N_pf, double L)
//{
//	return cdl->get_h(alpha,n,N_pf,L);
//}
//MATHFUNCSDLL_API double test_iterpol1()
//{
//	return cdl->test_iterpol1();
//}
//MATHFUNCSDLL_API double test_iterpol2()
//{
//	return cdl->test_iterpol2();
//}
//MATHFUNCSDLL_API double test_get_ICFFT()
//{
//	return cdl->test_get_ICFFT();
//}
//MATHFUNCSDLL_API double test_get_IFFT2D_V2C()
//{
//	return cdl->test_get_IFFT2D_V2C();
//}
//MATHFUNCSDLL_API double test_get_CFFT2D_V2C()
//{
//	return cdl->test_get_CFFT2D_V2C();
//}
//MATHFUNCSDLL_API void how_to_opt()
//{
//	cdl->how_to_opt();
//}
//MATHFUNCSDLL_API double cuda_test_get_IFFT2D_V2C()
//{
//	return cdl->cuda_test_get_IFFT2D_V2C();
//}
//MATHFUNCSDLL_API double cuda_test_get_IFFT2D_V2C_using_CUDAIFFT()
//{
//	return cdl->cuda_test_get_IFFT2D_V2C_using_CUDAIFFT();
//}
//MATHFUNCSDLL_API double test_init_mas_UUSIG()
//{
//	return cdl->test_init_mas_UUSIG();
//}
//MATHFUNCSDLL_API double test_get_u_inter_for_vct()
//{
//	return cdl->test_get_u_inter_for_vct();
//}
//MATHFUNCSDLL_API double test_get_u_inter_for_vct2()
//{
//	return cdl->test_get_u_inter_for_vct2();
//}
MATHFUNCSDLL_API double test_Cuda_ConvertZ2Z()
{
	return cdl->test_Cuda_ConvertZ2Z();
}