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
MATHFUNCSDLL_API bool Cuda_ConvertZ2Z(int nCols,int nRows,int N_1out, int N_2out,double dFStart, double dFStop, double dAzStart, double dAzStop,double *zArray)
{
	if(cdl != 0) cdl=new cuda_calculate_class();
	return cdl->Cuda_ConvertZ2Z(nCols,nRows,N_1out, N_2out, dFStart,  dFStop,  dAzStart,  dAzStop,(Complex*)zArray);
}