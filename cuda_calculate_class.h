#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "helper_functions.h"
#include <helper_cuda.h>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>

typedef double2 doubleComplex;
typedef float2 floatComplex;

static  __device__ double double_device_get_kaiser(double alpha, int n, int N);
static __global__ void double_MakeComplex_ucc1_nn(doubleComplex* mas,double dd1,int N_1out);
static __global__ void double_MakeComplex_ucc2_nn(doubleComplex* mas,doubleComplex cc,double dd2,int N_1in,int N_2in,int N_1out,int N_2out,double dd);
static __global__ void double_Make_Plan(doubleComplex* out,doubleComplex* mas,int N_1out,int N_2out,int N_1in,int N_2in);
static __global__ void double_Mul_Last(doubleComplex* unif,doubleComplex* ucc2_nn,doubleComplex* ucc1_nn,int N_1out,int N_2out);
static __device__ double double_device_I0(double x);
static __device__ double double_device_round(double number);
static __global__ void double_cuda_parallel_first_mas(doubleComplex *out,doubleComplex *in,double k_start,double k_step,double k_stop,double fi_span,double fi_start,double fi_step,double *dev_h,int N_tapsd2,int N_k,int N_fi);
static __device__ doubleComplex double_device_get_u_inter_for_first(double x,double U_start,double U_step,int N_u,doubleComplex u_gr[],double h[],int N_tapsd2,int iphi,int N_fi);
static __device__ doubleComplex double_device_get_u_inter_for_second(double x,double U_start,double U_step,int N_u,doubleComplex u_gr[],double h[],int N_tapsd2,int ikdr,int N_k);
static __global__ void double_cuda_parallel_second_mas(doubleComplex *out,doubleComplex *in,double k_start,double fi_span,double fi_start,double fi_step,int N_tapsd2,double temp_alpha,int N_k,int N_fi,double *h,double k_stop);
static __device__ __host__ inline doubleComplex double_ComplexMul(doubleComplex, doubleComplex);
static __device__ __host__ inline doubleComplex double_MakeComplex(double, double);


static  __device__ float float_device_get_kaiser(float alpha, int n, int N);
static __global__ void float_MakeComplex_ucc1_nn(floatComplex* mas,float dd1,int N_1out);
static __global__ void float_MakeComplex_ucc2_nn(floatComplex* mas,floatComplex cc,float dd2,int N_1in,int N_2in,int N_1out,int N_2out,float dd);
static __global__ void float_Make_Plan(floatComplex* out,floatComplex* mas,int N_1out,int N_2out,int N_1in,int N_2in);
static __global__ void float_Mul_Last(floatComplex* unif,floatComplex* ucc2_nn,floatComplex* ucc1_nn,int N_1out,int N_2out);
static __device__ float float_device_I0(float x);
static __device__ float float_device_round(float number);
static __global__ void float_cuda_parallel_first_mas(floatComplex *out,floatComplex *in,float k_start,float k_step,float k_stop,float fi_span,float fi_start,float fi_step,float *dev_h,int N_tapsd2,int N_k,int N_fi);
static __device__ floatComplex float_device_get_u_inter_for_first(float x,float U_start,float U_step,int N_u,floatComplex u_gr[],float h[],int N_tapsd2,int iphi,int N_fi);
static __device__ floatComplex float_device_get_u_inter_for_second(float x,float U_start,float U_step,int N_u,floatComplex u_gr[],float h[],int N_tapsd2,int ikdr,int N_k);
static __global__ void float_cuda_parallel_second_mas(floatComplex *out,floatComplex *in,float k_start,float fi_span,float fi_start,float fi_step,int N_tapsd2,float temp_alpha,int N_k,int N_fi,float *h,float k_stop);
static __device__ __host__ inline floatComplex float_ComplexMul(floatComplex, floatComplex);
static __device__ __host__ inline floatComplex float_MakeComplex(float, float);

class cuda_calculate_class
{
public:
	const char* cuda_get_error();
	const char* cuda_get_error_str();
	bool SetArrayZ2Z(int nCols,int nRows,double dFStart, double dFStop, double dAzStart, double dAzStop,doubleComplex *zArrayin,bool Device);//true GeForce
	bool CalcZ2Z(int N_1out, int N_2out,double dFStart, double dFStop, double dAzStart, double dAzStop,doubleComplex *zArrayout);
	cuda_calculate_class();
    double get_xstart();
    double get_xstop();
    double get_zstart();
    double get_zstop();
	float f_get_xstart();
    float f_get_xstop();
    float f_get_zstart();
    float f_get_zstop();
	bool SetArrayC2C(int nCols,int nRows,float dFStart, float dFStop, float dAzStart, float dAzStop,floatComplex *zArrayin,bool Device);
	bool CalcC2C(int N_1out, int N_2out,float dFStart, float dFStop, float dAzStart, float dAzStop,floatComplex *zArrayout);
	
private:
	bool Cuda_ConvertZ2Z(int nCols,int nRows,int N_1out, int N_2out,double dFStart, double dFStop, double dAzStart, double dAzStop,doubleComplex *zArrayin,doubleComplex *zArrayout);
    bool Cuda_ConvertC2C(int nCols,int nRows,int N_1out, int N_2out,float dFStart, float dFStop, float dAzStart, float dAzStop,floatComplex *zArrayin,floatComplex *zArrayout);
	double round(double number);
	float float_round(float number);


	bool double_init_plan(int N_1out,int N_2out);
	bool double_cuda_free();
	double double_I0(double x);
	double double_get_kaiser(double alpha, int n, int N);
	double double_get_h(double alpha,int n, int N_pf, double L);
	void double_cuda_get_IFFT2D_V2C_using_CUDAIFFT(doubleComplex u[],int  N_1in,int  N_2in, int N_1out, int N_2out, double k_1start, double k_1step, double k_2start, double k_2step,doubleComplex* uout);
	void double_linear_init_mas_UUSIG(double FI0,double F_start, double F_stop, int N_k, int N_fi, double fi_degspan,doubleComplex *uusig);


	bool float_init_plan(int N_1out,int N_2out);
	bool float_cuda_free();
	float float_I0(float x);
	float float_get_kaiser(float alpha, int n, int N);
	float float_get_h(float alpha,int n, int N_pf, float L);
	void float_cuda_get_IFFT2D_V2C_using_CUDAIFFT(floatComplex u[],int  N_1in,int  N_2in, int N_1out, int N_2out, float k_1start, float k_1step, float k_2start, float k_2step,floatComplex* uout);
	void float_linear_init_mas_UUSIG(float FI0,float F_start, float F_stop, int N_k, int N_fi, float fi_degspan,floatComplex *uusig);

};