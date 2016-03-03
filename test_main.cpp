#include "stdio.h"
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <time.h>

struct double2 {double x; double y;};
struct float2 {float x; float y;};


extern "C"
{
#include "../../shared_folder/cuda_dll_module.h"
}
int main()
{
    int NEWCOL=2048;
    int NEWROW=1024;
    int OLDCOL = 740;
    int OLDROW = 820;
	/*double2 *masin=new double2[OLDCOL*OLDROW];
	double2 *masout=new double2[NEWCOL*NEWROW];*/
	//bool a = Cuda_ConvertZ2Z(OLDROW,OLDCOL,NEWROW,NEWCOL,8.2,10.2,40.0,60.0,(double*)masin,(double*)masout,true);
	/*bool a = SetArrayZ2Z(OLDROW,OLDCOL,8.2,10.2,40.0,60.0,(double*)masin);
	bool b = CalcZ2Z(NEWROW,NEWCOL,8.2,10.2,40.0,60.0,(double*)masout);*/

	//float2 *masin=new float2[OLDCOL*OLDROW];
	//float2 *masout=new float2[NEWCOL*NEWROW];
	
		float2 *masin=new float2[OLDCOL*OLDROW];
		float2 *masout=new float2[NEWCOL*NEWROW];
		double2 *d_masin=new double2[OLDCOL*OLDROW];
		double2 *d_masout=new double2[NEWCOL*NEWROW];


		for (int k=0;k<OLDCOL*OLDROW;k++)
		{
			masin[k].x = rand() % 20+20;
			d_masin[k].x = rand() % 20+20;

			masin[k].y = rand() % 20+20;
			d_masin[k].y = rand() % 20+20;
		}
	bool d_a = SetArrayZ2Z(OLDROW,OLDCOL,8.2,10.2,40.0,60.0,(double*)d_masin,false);
	bool d_b = CalcZ2Z(NEWROW,NEWCOL,8.2,10.2,40.0,60.0,(double*)d_masout);	

	bool a = SetArrayC2C(OLDROW,OLDCOL,8.2,10.2,40.0,60.0,(float*)masin,false);
	bool b = CalcC2C(NEWROW,NEWCOL,8.2,10.2,40.0,60.0,(float*)masout);


	int count = 0;
	for (int j=0;j<NEWCOL*NEWROW;j++){
		if ((float)masout[j].x != (float)d_masout[j].x)
		{
		
			count++;
		}
	}
	printf("total %d\n", NEWCOL*NEWROW);
	printf("count %d\n",count);
	//bool b = CalcC2C(NEWROW,NEWCOL,8.2,10.2,40.0,60.0,(float*)masout);
	printf("done\n");
	delete masin;
	delete masout;

	delete d_masin;
	delete d_masout;
	char aa[10];
	scanf(aa,"%d");
}