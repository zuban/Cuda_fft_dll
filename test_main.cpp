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
    int NEWCOL=2048;
    int NEWROW=1024;
    int OLDCOL = 740;
    int OLDROW = 820;
	double2 *mas=new double2[NEWCOL*NEWROW];
	bool a = Cuda_ConvertZ2Z(OLDROW,OLDCOL,NEWROW,NEWCOL,8.2,10.2,40.0,60.0,(double*)mas);
	printf("done\n");

	char aa[10];
	scanf(aa,"%d");
}




