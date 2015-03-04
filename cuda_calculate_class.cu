#include <cuda_runtime.h>

#include <cufft.h>
#include "cuda_calculate_class.h"
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>
double pi = 3.1415926;
const char *cuda_error;
const char *cuda_error_file;
Complex* ucc1_nn_dev;
Complex* ucc2_nn_dev;
Complex *dev_uni_n2_linear;

cufftHandle plan2D;
int OLDROW = 740;
int OLDCOL = 820;
int NEWROW;// = 2048;//N_1out
int NEWCOL;// = 1024;//N_2out

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      cuda_error = cudaGetErrorString(code);
	  cuda_error_file = file;
      if (abort) exit(code);
   }
}

const char* cuda_calculate_class::cuda_get_error()
{
	return cuda_error;
}
const char* cuda_calculate_class::cuda_get_error_str()
{
	return cuda_error_file;
}
bool cuda_calculate_class::init_plan(int N_1out,int N_2out)
{		
	NEWROW = N_1out;
	NEWCOL = N_2out;
	checkCudaErrors(cufftPlan2d(&plan2D,N_2out, N_1out, CUFFT_Z2Z));
	gpuErrchk(cudaMalloc((void **) &ucc1_nn_dev, sizeof(Complex) * N_1out));
	gpuErrchk(cudaMalloc((void **) &ucc2_nn_dev, sizeof(Complex) * N_2out));
	gpuErrchk(cudaMalloc((void **) &dev_uni_n2_linear, sizeof(Complex) * N_2out* N_1out));
	return true;
}

bool cuda_calculate_class::cuda_free()
{
	gpuErrchk(cudaFree(ucc1_nn_dev));
	gpuErrchk(cudaFree(ucc2_nn_dev));
	gpuErrchk(cudaFree(dev_uni_n2_linear));
	checkCudaErrors(cufftDestroy(plan2D));
	return true;
}

double cuda_calculate_class::get_kaiser(double alpha, int n, int N)
{
	double w = I0(pi * alpha * sqrt(1.0-(2.0 *double(n)/double(N-1) -1.0)*(2.0 * double(n)/double(N-1) -1.0)))/(I0(pi * alpha)); 
	return w;
}

double cuda_calculate_class::get_h(double alpha,int n, int N_pf, double L)
{
	double h = sin( pi* ( n - N_pf/2.0 + 0.5 ) / L + 1.0e-20) * get_kaiser(alpha,n, N_pf) / (( pi * (n - N_pf/2.0 +0.5) / L ) + 1.0e-20 ); 
	return h;
}

static __device__ Complex device_get_u_inter_for_first(double x,double U_start,double U_step,int N_u,Complex u_gr[],double h[],int N_tapsd2,int iphi,int N_fi)
{
	double ax = (x - U_start) / U_step;
	if (ax<0)
		return MakeComplex(0.0,0.0);
	if (ax>(N_u-1))
		return MakeComplex(0.0,0.0);
	int a = int(ax);
	double aa = ax -a;
	int L = 2000;
	int phase = device_round( aa * (double)L );
	int s_min=1;
	int s_max= 2* N_tapsd2;
	if ((a-N_tapsd2+s_min)<0)
		s_min=N_tapsd2-a;
	if ((a-N_tapsd2+s_max)>=(N_u-1))
		s_max=N_u - 1- a + N_tapsd2;
	Complex A= MakeComplex(0.0,0.0);
	for (unsigned int s = s_min ; s<=s_max ; s++)
	{
		A.x = A.x +h[ L * s - phase ]* u_gr[a - N_tapsd2 + s + iphi*N_u ].x;
		A.y = A.y +h[ L * s - phase ]* u_gr[a - N_tapsd2 + s + iphi*N_u ].y;
	}
	return A;
}

static __device__ Complex device_get_u_inter_for_second(double x,double U_start,double U_step,int N_u,Complex u_gr[],double h[],int N_tapsd2,int ikdr,int N_k)
{
double ax = (x - U_start) / U_step;
	if (ax<0)
		return MakeComplex(0.0,0.0);
	if (ax>(N_u-1))
		return MakeComplex(0.0,0.0);
	int a = int(ax);
	double aa = ax -a;
	int L = 2000;
	int phase = device_round( aa * (double)L );
	int s_min=1;
	int s_max= 2* N_tapsd2;
	if ((a-N_tapsd2+s_min)<0)
		s_min=N_tapsd2-a;
	if ((a-N_tapsd2+s_max)>=(N_u-1))
		s_max=N_u - 1- a + N_tapsd2;
	Complex A= MakeComplex(0.0,0.0);
	for (unsigned int s = s_min ; s<=s_max ; s++)
	{
		A.x = A.x +h[ L * s - phase ]* u_gr[ (a - N_tapsd2 + s) *N_k + ikdr ].x;
		A.y = A.y +h[ L * s - phase ]* u_gr[ (a - N_tapsd2 + s) *N_k + ikdr ].y;
	}
	return A;
}

double cuda_calculate_class::I0(double x)
{
 double a[19] = { 0.,4.2e-19,3.132e-17,2.06305e-15,
            1.1989083e-13,6.0968928e-12,2.6882812895e-10,1.016972672769e-8,
            3.2609105057896e-7,8.73831549662236e-6,1.92469359688114e-4,
            .00341633176601234,.0477187487981741,.509493365439983,
            4.01167376017935,22.2748192424622,82.4890327440241,
            190.494320172743,255.466879624362 };
 double c__[26] = { 1e-14,2e-14,0.,-4e-14,-1.1e-13,-1.1e-13,
            1.2e-13,7e-13,1.4e-12,1.11e-12,-2.45e-12,-1.223e-11,-2.896e-11,
            -4.361e-11,-1.954e-11,1.5142e-10,7.8081e-10,2.87935e-9,
            1.045317e-8,4.342656e-8,2.27083e-7,1.57817791e-6,1.527877872e-5,
            2.2510873571e-4,.00627824030274,.79833170337777 };
 double exp30 = 10686474581524.4;
 double xmax = 46.499;
 double sys059 = 1.7e308;

 double ret_val;

 int i__;
 double y, z__;
 double q1, q2, q3, x1, x2;

//    *ierr = 0;
 if (x < 0.) {
   goto L4;
 }
//L1:
 y = fabs(x);
 z__ = y * .125;
 if (z__ > 1.) {
   goto L5;
 }
 if (z__ <= 0.) {
   goto L3;
 }
 x2 = z__ * 4. * z__ - 2.;
 q3 = 0.;
 q2 = 0.;
 for (i__ = 1; i__ <= 19; ++i__) {
   q1 = q2;
   q2 = q3;
/* L2: */
   q3 = x2 * q2 - q1 + a[i__ - 1];
 }
 ret_val = (q3 - q1) * .5;
 goto L10;
L3:
 ret_val = 1.;
 goto L10;
L4:
// *ierr = 1;
 goto L9;
L5:
 z__ = 1. / z__;
 x2 = z__ + z__;
 q3 = 0.;
 q2 = 0.;
 for (i__ = 1; i__ <= 26; ++i__) {
   q1 = q2;
   q2 = q3;
/* L6: */
   q3 = x2 * q2 - q1 + c__[i__ - 1];
    }
 q2 = (q3 - q1) * .5;
 x1 = 1. / sqrt(y);
 if (y < xmax) {
   goto L7;
    }
// *ierr = 65;
 ret_val = sys059;
 goto L9;
L7:
 if (y > 30.) {
   goto L8;
 }
 ret_val = exp(y) * x1 * q2;
 goto L10;
L8:
 ret_val = x1 * exp30 * (q2 * exp(y - 30.));
L9:
// if (*ierr != 0) {
//   utsf11_c(ierr, &c__28);
// }
// if (*ierr == 1) {
//   goto L1;
// }
L10:
 return ret_val;
}


void __global__ MakeComplex_ucc1_nn(Complex* mas,double dd1,int N_1out)
{
	int i = blockIdx.x;
	if (i>=N_1out)
		return;
	double c1 = 2.0 * 3.1415926 * (double)i * dd1;
	mas[i] = MakeComplex(cos(c1),sin(c1));
}

void __global__ MakeComplex_ucc2_nn(Complex* mas,Complex cc,double dd2,int N_1in,int N_2in,int N_1out,int N_2out,double dd)
{
	int i = blockIdx.x;
	if (i>=N_2out)
		return;
	double sin1;
	double cos1;
	
	//Complex cctest = MakeComplex(1.0/(double)(N_1out*N_2out),0.0);//TEST!!!
	Complex cctest = MakeComplex(((double)N_1out * (double)N_2out / (double)( N_1in*N_2in*N_1out*N_2out)),0.0);
	//double dd = k_1start/k_1step + k_2start/k_2step;
	//cc = ComplexMul(cc,MakeComplex(cos( - pi * dd),sin(- pi * dd)));
	sincos((3.1415926*( 2.0  * (double)i * dd2 - dd)),&sin1,&cos1);
	mas[i] =ComplexMul(cctest,MakeComplex(cos1,sin1));
}

void __global__  Make_Plan(Complex* out,Complex* mas,int N_1out,int N_2out,int N_1in,int N_2in)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;

	if (x>=N_1out || y>=N_2out)
		return;
	int idx = y*N_1out + x;
			if (x>=N_1in){
				out[idx].x = 0.0;
				out[idx].y = 0.0;
			}
			else
			{
				if (y>=N_2in){ 
				out[idx].x = 0.0;
				out[idx].y = 0.0;
				}
				else{
				out[idx] = mas[N_1in*y+x];
				if (x%2 != y%2)
				{
				out[idx].x = -out[idx].x;
				out[idx].y = -out[idx].y;
				}
				}
			}
}

void __global__  Mul_Last(Complex* unif,Complex* ucc2_nn,Complex* ucc1_nn,int N_1out,int N_2out)
{
	__shared__ Complex cx[256*3];
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	if (x>=N_1out || y>=N_2out)
		return;
	int idx = y*N_1out + x;
	int ind = threadIdx.x + threadIdx.y*blockDim.x;
	cx[ind] = unif[idx];
	__syncthreads();
	cx[ind+256] = ucc1_nn[x];
	__syncthreads();
	cx[ind+2*256] = ucc2_nn[y];
	__syncthreads();

	cx[ind] = ComplexMul(ComplexMul(cx[ind],cx[ind+2*256]),cx[ind+256]);
	unif[idx]  = cx[ind];
}
void cuda_calculate_class::cuda_get_IFFT2D_V2C_using_CUDAIFFT(Complex u[],int  N_1in, int  N_2in, int N_1out, int N_2out, double k_1start, double k_1step, double k_2start, double k_2step,Complex* uout)
{
	Complex *dev_u;
	checkCudaErrors(cudaMalloc((void **) &dev_u, sizeof(Complex) * N_1in* N_2in));
	checkCudaErrors(cudaMemcpy(dev_u, u, sizeof(Complex) *N_1in* N_2in, cudaMemcpyHostToDevice));

	int threadNum1 = 256;
	dim3 blockSize1 = dim3(threadNum1, 1, 1); 
	int ivx1 = N_1out/threadNum1;
	if(ivx1*threadNum1 != N_1out) ivx1++;
    dim3 gridSize1 = dim3(ivx1, N_2out, 1);

	int threadNum = 1024;
	int ivx = N_1out/threadNum;
	if(ivx*threadNum != N_1out) ivx++;
	dim3 blockSize = dim3(threadNum, 1, 1); 
    dim3 gridSize = dim3(ivx, N_2out, 1);

	 //time
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//time
	double dd1 = k_1start / (N_1out * k_1step);
	MakeComplex_ucc1_nn<<<N_1out,1>>>(ucc1_nn_dev,dd1,N_1out);

	Complex cc = MakeComplex(((double)N_1out * (double)N_2out / (double)( N_1in*N_2in*N_1out*N_2out)),0.0);
	double dd = k_1start/k_1step + k_2start/k_2step;
	cc = ComplexMul(cc,MakeComplex(cos( - pi * dd),sin(- pi * dd)));
	double dd2 = k_2start / ( (double)N_2out * k_2step);
	MakeComplex_ucc2_nn<<<N_2out, 1>>>(ucc2_nn_dev,cc,dd2,N_1in,N_2in,N_1out,N_2out,dd);	
	Make_Plan<<<gridSize,blockSize>>>(dev_uni_n2_linear,dev_u,N_1out,N_2out,N_1in,N_2in);
	checkCudaErrors(cufftExecZ2Z(plan2D, (Complex *)dev_uni_n2_linear, (Complex *)dev_uni_n2_linear, CUFFT_INVERSE));
	Mul_Last<<<gridSize1,blockSize1>>>(dev_uni_n2_linear,ucc2_nn_dev,ucc1_nn_dev,N_1out,N_2out);
	//time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("IFFT time		 %f ms\n", time);
	//time
	checkCudaErrors(cudaMemcpy(uout, dev_uni_n2_linear,  sizeof(Complex) *N_2out* N_1out, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(dev_u));
}
static __device__ double device_get_kaiser(double alpha, int n, int N)
{
	double pi = 3.1415926;
	double w = device_I0(pi * alpha * sqrt(1.0-(2.0 *double(n)/double(N-1) -1.0)*(2.0 * double(n)/double(N-1) -1.0)))/(device_I0(pi * alpha)); 
	return w;
}

static __device__ double device_I0(double x)
{
 double a[19] = { 0.,4.2e-19,3.132e-17,2.06305e-15,
            1.1989083e-13,6.0968928e-12,2.6882812895e-10,1.016972672769e-8,
            3.2609105057896e-7,8.73831549662236e-6,1.92469359688114e-4,
            .00341633176601234,.0477187487981741,.509493365439983,
            4.01167376017935,22.2748192424622,82.4890327440241,
            190.494320172743,255.466879624362 };
 double c__[26] = { 1e-14,2e-14,0.,-4e-14,-1.1e-13,-1.1e-13,
            1.2e-13,7e-13,1.4e-12,1.11e-12,-2.45e-12,-1.223e-11,-2.896e-11,
            -4.361e-11,-1.954e-11,1.5142e-10,7.8081e-10,2.87935e-9,
            1.045317e-8,4.342656e-8,2.27083e-7,1.57817791e-6,1.527877872e-5,
            2.2510873571e-4,.00627824030274,.79833170337777 };
 double exp30 = 10686474581524.4;
 double xmax = 46.499;
 double sys059 = 1.7e308;

 double ret_val;

 int i__;
 double y, z__;
 double q1, q2, q3, x1, x2;

//    *ierr = 0;
 if (x < 0.) {
   goto L4;
 }
//L1:
 y = fabs(x);
 z__ = y * .125;
 if (z__ > 1.) {
   goto L5;
 }
 if (z__ <= 0.) {
   goto L3;
 }
 x2 = z__ * 4. * z__ - 2.;
 q3 = 0.;
 q2 = 0.;
 for (i__ = 1; i__ <= 19; ++i__) {
   q1 = q2;
   q2 = q3;
/* L2: */
   q3 = x2 * q2 - q1 + a[i__ - 1];
 }
 ret_val = (q3 - q1) * .5;
 goto L10;
L3:
 ret_val = 1.;
 goto L10;
L4:
// *ierr = 1;
 goto L9;
L5:
 z__ = 1. / z__;
 x2 = z__ + z__;
 q3 = 0.;
 q2 = 0.;
 for (i__ = 1; i__ <= 26; ++i__) {
   q1 = q2;
   q2 = q3;
/* L6: */
   q3 = x2 * q2 - q1 + c__[i__ - 1];
    }
 q2 = (q3 - q1) * .5;
 x1 = 1. / sqrt(y);
 if (y < xmax) {
   goto L7;
    }
// *ierr = 65;
 ret_val = sys059;
 goto L9;
L7:
 if (y > 30.) {
   goto L8;
 }
 ret_val = exp(y) * x1 * q2;
 goto L10;
L8:
 ret_val = x1 * exp30 * (q2 * exp(y - 30.));
L9:
// if (*ierr != 0) {
//   utsf11_c(ierr, &c__28);
// }
// if (*ierr == 1) {
//   goto L1;
// }
L10:
 return ret_val;

}
static __device__ double device_round(double number)
{
	return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}
void __global__  cuda_parallel_first_mas(Complex *out,Complex *in,double k_start,double k_step,double k_stop,double fi_span,double fi_start,double fi_step,double *dev_h,int N_tapsd2,int N_k,int N_fi)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	
	if (x>=N_fi || y>=N_k)
		return;
	double dKdr = 2*k_start + ((2*k_stop*cos(fi_span/2.0)-2*k_start)/((double)N_k-1))*(double)y;
	double dPhi = (fi_start + fi_step*x);
	double d = dKdr/2.0/cos((double)dPhi);
	out[y+x*N_k] = device_get_u_inter_for_first(d,k_start,k_step,N_k,in, dev_h, N_tapsd2,x,N_fi); 
}

void __global__  cuda_parallel_second_mas(Complex *out,Complex *in,double k_start,double fi_span,double fi_start,double fi_step,int N_tapsd2,double temp_alpha,int N_k,int N_fi,double *h,double k_stop)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	
	if (x>=N_k || y>=N_fi)
		return;

	 double dKdr = 2*k_start + ((2*k_stop*cos(fi_span/2.0)-2*k_start)/((double)N_k-1))*(double)x;
	 double dKcr = 2*k_start*sin(-fi_span/2.0) + ((2*k_start*sin(fi_span/2.0)-2*k_start*sin(-fi_span/2.0))/((double)N_fi-1))*(double)y;
	 double d = atan(dKcr/dKdr);
	 out[y*N_k+x]  = ComplexMul(device_get_u_inter_for_second(d,fi_start,fi_step,N_fi, in , h, N_tapsd2,x,N_k),ComplexMul(MakeComplex(device_get_kaiser(temp_alpha,y,N_fi),0.0),MakeComplex(device_get_kaiser(temp_alpha,x,N_k),0.0)));
}

bool cuda_calculate_class::Cuda_ConvertZ2Z(int nCols,int nRows,int N_1out, int N_2out,double dFStart, double dFStop, double dAzStart, double dAzStop,Complex *zArrayin,Complex *zArrayout)
{
	int N_k = nCols;//740;//37;//16;//1000; 
	int N_fi = nRows;//820;//41;//16;//1000;

	NEWROW = N_1out;
	NEWCOL = N_2out;
	init_plan(NEWROW,NEWCOL);
	double F_start = dFStart;//8.2;
	double F_stop = dFStop;//10.2;
	double k_start = 2 * pi * F_start / 0.3;
	double k_stop = 2* pi * F_stop/0.3;
	double k_step = (k_stop-k_start) / ((double)N_k-1);
	double fi_degspan = dAzStop - dAzStart;//20.0;
	double fi_start = - (fi_degspan/2) * (pi/180.0);
	double fi_stop = (fi_degspan/2) * (pi/180.0);
	double fi_span = fi_stop-fi_start;
	double fi_step = (fi_stop-fi_start) / ((double)N_fi-1);
	//Complex *v2cc_lin_Complex = new Complex[N_k*N_fi];
	//linear_init_mas_UUSIG(0.0,F_start,F_stop,N_k,N_fi,fi_degspan,v2cc_lin_Complex);	
	const int N_tapsd2 = 7;
	const int L= 2000;
	double alpha = 2.0;
	const int N_ptapsd2 = N_tapsd2 * L;
	const int ad = 2* N_ptapsd2;
	double h[ ad ];
	for (int i=0;i< ad;i++)
	{
		h[i]=get_h(alpha,i, ad+1,L);
	}
	cudaError_t error;
	double* dev_h;
    checkCudaErrors(cudaMalloc((void **) &dev_h, sizeof(double) * ad));
    checkCudaErrors(cudaMemcpy( dev_h, h, sizeof(double) *ad, cudaMemcpyHostToDevice));
	//time
	cudaEvent_t start3, stop3;
	float time3;
    cudaEventCreate(&start3);
	cudaEventCreate(&stop3);
	cudaEventRecord(start3, 0);
	//time
	//POINT A 320ms
	//time
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//time
    //1
	int devID = 0;
	cudaDeviceProp deviceProp;
	error = cudaGetDeviceProperties(&deviceProp, devID);
	if (deviceProp.major < 2)
	{
		cuda_error = "convertZ2Z compute capability error";
		return false;
	}
    Complex *dev_v2cc_lin_Complex_out;
	Complex *dev_v2cc_lin_Complex;
	Complex *v2cc_lin_Complex_out=new Complex[ N_k* N_fi];
	gpuErrchk(cudaMalloc((void **) &dev_v2cc_lin_Complex_out, sizeof(Complex) * N_k* N_fi));
	gpuErrchk(cudaMalloc((void **) &dev_v2cc_lin_Complex, sizeof(Complex) * N_k* N_fi));
	gpuErrchk(cudaMemcpy(dev_v2cc_lin_Complex, zArrayin, sizeof(Complex) *N_k* N_fi, cudaMemcpyHostToDevice));
	int threadNum_first = 1024;
	int ivx_first = N_fi/threadNum_first;
	if(ivx_first*threadNum_first != N_fi) ivx_first++;
	dim3 blockSize_first = dim3(threadNum_first, 1, 1); 
	dim3 gridSize_first = dim3(ivx_first, N_k, 1);

	cuda_parallel_first_mas<<<gridSize_first,blockSize_first>>>(dev_v2cc_lin_Complex_out,dev_v2cc_lin_Complex,k_start,k_step,k_stop,fi_span,fi_start,fi_step,dev_h,N_tapsd2,N_k,N_fi);
	gpuErrchk( cudaPeekAtLastError() );
    //end 1

	 //time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("cuda_convert_auproc: fisrt mas		 %f ms\n", time);
	//time
	//POINT B (B-A=846ms)
	//time
	cudaEvent_t start1, stop1;
	float time1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, 0);
	//time
	//2
	int threadNum_sec = 1024;
	int ivx_sec =N_k/threadNum_sec;
	if(ivx_sec*threadNum_sec != N_k) ivx_sec++;
	dim3 blockSize_sec = dim3(threadNum_sec, 1, 1); 
	dim3 gridSize_sec = dim3(ivx_sec, N_fi, 1);
	double temp_alpha = 2.0;

	cuda_parallel_second_mas<<<gridSize_sec,blockSize_sec>>>(dev_v2cc_lin_Complex,dev_v2cc_lin_Complex_out,k_start,fi_span,fi_start,fi_step,N_tapsd2,temp_alpha,N_k,N_fi,dev_h,k_stop);
	gpuErrchk( cudaPeekAtLastError() );
	//time
	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&time1, start1, stop1);
	printf ("cuda_convert_auproc: second mas		 %f ms\n", time1);
	//time
	//POINT C (C-B=1035ms)
	//IFFT
	//POINT D (D-C=45ms)	
	//time
    cudaEvent_t start2, stop2;
	float time2;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord(start2, 0);
	//time
	// cuda_get_IFFT2D_V2C_using_CUDAIFFT
	double k_1start = 0.3;
	double k_1step = 0.02;
	double k_2start = 12.4;
	double k_2step = 0.01;
	int threadNum1 = 256;
	dim3 blockSize1 = dim3(threadNum1, 1, 1); 
	int ivx1 = NEWROW/threadNum1;
	if(ivx1*threadNum1 != NEWROW) ivx1++;
    dim3 gridSize1 = dim3(ivx1, NEWCOL, 1);

	int threadNum = 1024;
	int ivx = NEWROW/threadNum;
	if(ivx*threadNum != NEWROW) ivx++;
	dim3 blockSize = dim3(threadNum, 1, 1); 
    dim3 gridSize = dim3(ivx, NEWCOL, 1);
	double dd1 = k_1start / (NEWROW * k_1step);
	MakeComplex_ucc1_nn<<<NEWROW,1>>>(ucc1_nn_dev,dd1,NEWROW);
	gpuErrchk( cudaPeekAtLastError() );
	Complex cc = MakeComplex(((double)NEWROW * (double)NEWCOL / (double)( N_k*N_fi*NEWROW*NEWCOL)),0.0);
	double dd = k_1start/k_1step + k_2start/k_2step;
	cc = ComplexMul(cc,MakeComplex(cos( - pi * dd),sin(- pi * dd)));
	double dd2 = k_2start / ( (double)NEWCOL * k_2step);
	MakeComplex_ucc2_nn<<<NEWCOL, 1>>>(ucc2_nn_dev,cc,dd2,N_k,N_fi,NEWROW,NEWCOL,dd);	
	gpuErrchk( cudaPeekAtLastError() );
	Make_Plan<<<gridSize,blockSize>>>(dev_uni_n2_linear,dev_v2cc_lin_Complex,NEWROW,NEWCOL,N_k,N_fi);
	gpuErrchk( cudaPeekAtLastError() );
	checkCudaErrors(cufftExecZ2Z(plan2D, (Complex *)dev_uni_n2_linear, (Complex *)dev_uni_n2_linear, CUFFT_INVERSE));
	Mul_Last<<<gridSize1,blockSize1>>>(dev_uni_n2_linear,ucc2_nn_dev,ucc1_nn_dev,NEWROW,NEWCOL);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk(cudaMemcpy(zArrayout, dev_uni_n2_linear,  sizeof(Complex) *NEWCOL* NEWROW, cudaMemcpyDeviceToHost));
	//end cuda_get_IFFT2D_V2C_using_CUDAIFFT
	//time
	cudaEventRecord(stop2, 0);
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&time2, start2, stop2);
	printf ("cuda_convert_auproc: IFFT2D		 %f ms\n", time2);
	//time
	//POINT E (E-D=1155 ms)
	//time
	cudaEventRecord(stop3, 0);
	cudaEventSynchronize(stop3);
	cudaEventElapsedTime(&time3, start3, stop3);
	printf ("cuda_convert_auproc: full time		 %f ms\n", time3);
	//time
	//FULL TIME 3638
	gpuErrchk(cudaFree(dev_v2cc_lin_Complex_out));
	gpuErrchk(cudaFree(dev_v2cc_lin_Complex));
	gpuErrchk(cudaFree(dev_h));
	return true;
	cuda_free();
}
void cuda_calculate_class::linear_init_mas_UUSIG(double FI0,double F_start, double F_stop, int N_k, int N_fi, double fi_degspan,Complex *uusig)
{
	double k_start = 2 * pi * F_start / 0.3;
	double k_stop = 2* pi * F_stop/0.3;
	double *k_mas = new double[N_k];
	for (int i=0;i<= N_k-1; i++)
	{
		k_mas[i] = k_start + (k_stop - k_start) * (double)i / ((double)N_k - 1.0 );
	}
	//double k_step = ( k_stop - k_start ) / ((double)N_k-1.0);
	double fi_start = - (fi_degspan/2) * (pi/180.0);
	double fi_stop = (fi_degspan/2) * (pi/180.0);
	//double fi_span = fi_stop-fi_start;
	double*  fi_mas=new double[N_fi];
	for (int i=0;i<= N_fi-1; i++)
	{
		fi_mas[i]= fi_start + (fi_stop-fi_start)* (double)i / ((double)N_fi-1) + FI0*pi/180.0;
	}
	double x0=0.5;
	double y0=0.5;
	double x1=-0.5;
	double y1=-.5;
	////double x0=0.0;
	////double y0=0.0;
	//double x1=1.0;
	//double y1=0.5;
	//double x1=0.0;
	//double y1=0.0;

	/*double x0=1.0;
	double y0=1.0;*/

	for (int i=0; i<=N_k-1;i++)
	{
		for (int j=0;j<=N_fi-1;j++)
		{
			//uusig[N_k*j+i] = MakeComplex(1.0,0.0);
			/*uusig[N_k*j+i] = MakeComplex(cos( - 2.0 * k_mas[i] * y0 -  2.0 * k_mas[0] * x0 * fi_mas[j]),
				sin(- 2.0 * k_mas[i] * y0 -  2.0 * k_mas[0] * x0 * fi_mas[j]));*/
			uusig[N_k*j+i] = MakeComplex(cos( - 2.0 * k_mas[i] * y0 * cos(fi_mas[j]) -  2.0 * k_mas[i] * x0 * sin(fi_mas[j])),
				sin(- 2.0 * k_mas[i] * y0 * cos(fi_mas[j]) -  2.0 * k_mas[i] * x0 * sin(fi_mas[j])));
			////test
			uusig[N_k*j+i].x = uusig[N_k*j+i].x + MakeComplex(cos( - 2.0 * k_mas[i] * y1 * cos(fi_mas[j]) -  2.0 * k_mas[i] * x1 * sin(fi_mas[j])),
				sin(- 2.0 * k_mas[i] * y1 * cos(fi_mas[j]) -  2.0 * k_mas[i] * x1 * sin(fi_mas[j]))).x;
			uusig[N_k*j+i].y = uusig[N_k*j+i].y + MakeComplex(cos( - 2.0 * k_mas[i] * y1 * cos(fi_mas[j]) -  2.0 * k_mas[i] * x1 * sin(fi_mas[j])),
				sin(- 2.0 * k_mas[i] * y1 * cos(fi_mas[j]) -  2.0 * k_mas[i] * x1 * sin(fi_mas[j]))).y;
			////test
		}
	}
	delete k_mas;
	delete fi_mas;
}

cuda_calculate_class::cuda_calculate_class()
{
}

Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}
Complex MakeComplex(double a, double b)
{
	Complex c;
	c.x = a;
	c.y = b;
	return c;
}