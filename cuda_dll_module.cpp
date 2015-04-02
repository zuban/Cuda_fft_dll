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

MATHFUNCSDLL_API const char* cuda_get_error()
{
	return cdl->cuda_get_error();
}
MATHFUNCSDLL_API const char* cuda_get_error_str()
{
	return cdl->cuda_get_error_str();
}
//MATHFUNCSDLL_API bool Cuda_ConvertZ2Z(int nCols,int nRows,int N_1out, int N_2out,double dFStart, double dFStop, double dAzStart, double dAzStop,double *zArrayin,double *zArrayout)
//{
//	if(cdl != 0) cdl=new cuda_calculate_class();
//	return cdl->Cuda_ConvertZ2Z(nCols,nRows,N_1out, N_2out, dFStart,  dFStop,  dAzStart,  dAzStop,(doubleComplex*)zArrayin,(doubleComplex*)zArrayout);
//}
MATHFUNCSDLL_API bool SetArrayZ2Z(int nCols,int nRows,double dFStart, double dFStop, double dAzStart, double dAzStop,double *zArrayin)
{
	if(cdl != 0) cdl=new cuda_calculate_class();
	return cdl->SetArrayZ2Z(nCols,nRows,dFStart,dFStop,dAzStart,dAzStop,(doubleComplex*)zArrayin);
}
MATHFUNCSDLL_API bool CalcZ2Z(int N_1out, int N_2out,double dFStart, double dFStop, double dAzStart, double dAzStop,double *zArrayout)
{
	return cdl->CalcZ2Z(N_1out,N_2out,dFStart,dFStop,dAzStart,dAzStop,(doubleComplex*)zArrayout);
}

//MATHFUNCSDLL_API bool Cuda_ConvertC2C(int nCols,int nRows,int N_1out, int N_2out,float dFStart, float dFStop, float dAzStart, float dAzStop,float *zArrayin,float *zArrayout)
//{
//	if(cdl != 0) cdl=new cuda_calculate_class();
//	return cdl->Cuda_ConvertC2C(nCols,nRows,N_1out, N_2out, dFStart,  dFStop,  dAzStart,  dAzStop,(floatComplex*)zArrayin,(floatComplex*)zArrayout);
//}
MATHFUNCSDLL_API double get_xstart()
{
	return cdl->get_xstart();
}

MATHFUNCSDLL_API double get_xstop()
{
	return cdl->get_xstop();
}

MATHFUNCSDLL_API double get_zstart()
{
	return cdl->get_zstart();
}

MATHFUNCSDLL_API double get_zstop()
{
	return cdl->get_zstop();
}
