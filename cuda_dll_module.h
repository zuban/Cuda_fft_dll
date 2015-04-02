#ifdef CUDA_DLL_EXPORT
#define MATHFUNCSDLL_API extern "C" __declspec(dllexport) 
#else
#define MATHFUNCSDLL_API extern "C" __declspec(dllimport) 
#endif

MATHFUNCSDLL_API bool Cuda_ConvertZ2Z(int nCols,int nRows,int N_1out, int N_2out,double dFStart, double dFStop, double dAzStart, double dAzStop,double *zArrayin,double *zArrayout);
MATHFUNCSDLL_API bool Cuda_ConvertC2C(int nCols,int nRows,int N_1out, int N_2out,float dFStart, float dFStop, float dAzStart, float dAzStop,float *zArrayin,float *zArrayout);

MATHFUNCSDLL_API const char* cuda_get_error();
MATHFUNCSDLL_API const char* cuda_get_error_str();

MATHFUNCSDLL_API double get_xstart();
MATHFUNCSDLL_API double get_xstop();
MATHFUNCSDLL_API double get_zstart();
MATHFUNCSDLL_API double get_zstop();
