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
	double2 *masin=new double2[OLDCOL*OLDROW];
	double2 *masout=new double2[NEWCOL*NEWROW];
	//bool a = Cuda_ConvertZ2Z(OLDROW,OLDCOL,NEWROW,NEWCOL,8.2,10.2,40.0,60.0,(double*)masin,(double*)masout);
	bool a = SetArrayZ2Z(OLDROW,OLDCOL,8.2,10.2,40.0,60.0,(double*)masin);
	bool b = CalcZ2Z(NEWROW,NEWCOL,8.2,10.2,40.0,60.0,(double*)masout);

	//float2 *masin=new float2[OLDCOL*OLDROW];
	//float2 *masout=new float2[NEWCOL*NEWROW];
	//bool a = Cuda_ConvertC2C(OLDROW,OLDCOL,NEWROW,NEWCOL,8.2,10.2,40.0,60.0,(float*)masin,(float*)masout);
	printf("done\n");

	char aa[10];
	scanf(aa,"%d");
}