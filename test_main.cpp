#include "stdio.h"
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <time.h>

struct double2 {double x; double y;}; 
extern "C"
{
#include "../../shared_folder/cuda_dll_module.h"
}
int main()
{
	//double a=ShortBeep(1,5);
    //printf("%lf    \n",a);

	//double test1 = get_kaiser_CUDA(2,1,101);
	//printf("get_kaiser test 1 0.016   %lf    \n",test1);

	//double test2 = get_kaiser_CUDA(2,50,101);
	//printf("get_kaiser test 2 1   %lf    \n",test2);

	//double test3 = get_kaiser_CUDA(2,5,25);
	//printf("get_kaiser test 3 0.343   %lf    \n",test3);

	//double test4 = get_h_CUDA(2,7,14,3.3);
	//printf("get_h test 1 0.946   %lf    \n",test4);

	//double test6 = get_h_CUDA(2,6,15,3.3);
	//printf("get_h test 3 0.807               %lf    \n",test6);

	//double test7 = get_h_CUDA(2,7,15,3.3);
	//printf("get_h test 4 1                   %lf    \n",test7);

	//double test8 = get_h_CUDA(2,8,15,3.3);
	//printf("get_h test 4 0.807               %lf    \n",test8);

	Init_CUDA(2048,1024);
	

	/*double test9 =  test_iterpol2();
	printf("get_u_inter imag test 1         %lf    \n",test9);

	double test10 =  test_iterpol1();
	printf("get_u_inter real test 2         %lf    \n",test10);*/

	//double test11 = test_get_ICFFT();
	//printf("get_ICFFT real test		        %lf    \n",test11);

	//double test12 = test_get_IFFT2D_V2C();
	//printf("get_IFFT2D_V2C real test		    %lf    \n",test12);

	//double test13 = test_get_CFFT2D_V2C();
	//printf("get_CFFT2D_V2C real test		   %lf    \n",test13);

	//how_to_opt();
	//printf("test cuda function: multiplication\n");
	//
	//double test181 = cuda_test_get_IFFT2D_V2C();
	//printf("cuda_test_get_IFFT2D_V2C real test		  %lf    \n",test181);

	//double test1811 = cuda_test_get_IFFT2D_V2C_using_CUDAIFFT();
	//printf("cuda_test_get_IFFT2D_V2C_using_CUDAIFFT real test	  %lf    \n",test1811);

//	double test18111 = test_init_mas_UUSIG();
	//printf("test_init_mas_UUSIG() imag test	  %lf    \n",test18111);

	//double test181111 =test_get_u_inter_for_vct();
	//printf("test_get_u_inter_for_vct real test	-0.722    %lf    \n",test181111);

	//double test1811111 =test_get_u_inter_for_vct2();
	//printf("test_get_u_inter_for_vct real test	0.692    %lf    \n",test1811111);
	
	printf("init complete\n");
	double test1811112 =test_Cuda_ConvertZ2Z();
//	printf("test_cuda_convert_auproc() test    %lf    \n",test1811112);

	printf("done\n");
	/*
	Init_CUDA();
	int nRow=8192;
	int nCol=8192;
	double2* mas=new double2 [nCol*nRow];
	for (int i=0;i<nRow*nCol;i++)
	{
		mas[i].x=1.0;
		mas[i].y=0.0;
	}
	Transform_CUDA(nCol,nRow,(double*)mas);
	clock_t t15 = clock();
	Transform_CUDA(nCol,nRow,(double*)mas);
	clock_t t16 = clock();
    printf("%f \n",((double) (t16-t15) / (double)CLOCKS_PER_SEC));
	*/
   // for (unsigned int i = 0; i < nCol*nRow; ++i)
   // {
    //  printf("%f \n",sqrt(mas[i].y*mas[i].y+mas[i].x*mas[i].x)); 
    //}
	Free_CUDA();
	char aa[10];
	scanf(aa,"%d");
}




