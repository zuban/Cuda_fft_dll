#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "helper_functions.h"
#include <helper_cuda.h>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>

typedef double2 Complex;

static  __device__ double device_get_kaiser(double alpha, int n, int N);
static __global__ void MakeComplex_ucc1_nn(Complex* mas,double dd1,int N_1out);
static __global__ void MakeComplex_ucc2_nn(Complex* mas,Complex cc,double dd2,int N_1in,int N_2in,int N_1out,int N_2out,double dd);
static __global__ void Make_Plan(Complex* out,Complex* mas,int N_1out,int N_2out,int N_1in,int N_2in);
static __global__ void Mul_Last(Complex* unif,Complex* ucc2_nn,Complex* ucc1_nn,int N_1out,int N_2out);
static __device__ double device_I0(double x);
static __device__ double device_round(double number);
static __global__ void cuda_parallel_first_mas(Complex *out,Complex *in,double k_start,double k_step,double k_stop,double fi_span,double fi_start,double fi_step,double *dev_h,int N_tapsd2,int N_k,int N_fi);
static __device__ Complex device_get_u_inter_for_first(double x,double U_start,double U_step,int N_u,Complex u_gr[],double h[],int N_tapsd2,int iphi,int N_fi);
static __device__ Complex device_get_u_inter_for_second(double x,double U_start,double U_step,int N_u,Complex u_gr[],double h[],int N_tapsd2,int ikdr,int N_k);
static __global__ void cuda_parallel_second_mas(Complex *out,Complex *in,double k_start,double fi_span,double fi_start,double fi_step,int N_tapsd2,double temp_alpha,int N_k,int N_fi,double *h,double k_stop);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __device__ __host__ inline Complex MakeComplex(double, double);

class cuda_calculate_class
{
public:

	const char* cuda_get_error();
	const char* cuda_get_error_file();
	bool init_plan(int N_1out,int N_2out);
	bool cuda_free();
	bool Cuda_ConvertZ2Z(int nCols,int nRows,double FI0,Complex *zArray);
	cuda_calculate_class();
	double test_Cuda_ConvertZ2Z();
private:
	double I0(double x);
	double get_kaiser(double alpha, int n, int N);
	double get_h(double alpha,int n, int N_pf, double L);
	void cuda_get_IFFT2D_V2C_using_CUDAIFFT(Complex u[],int  N_1in,int  N_2in, int N_1out, int N_2out, double k_1start, double k_1step, double k_2start, double k_2step,Complex* uout);
	void linear_init_mas_UUSIG(double FI0,double F_start, double F_stop, int N_k, int N_fi, double fi_degspan,Complex *uusig);
	
};