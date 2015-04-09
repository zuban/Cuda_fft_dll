# Cuda fft dll
##error handling. Для получения кода ошибки и номера строки: 

1. cuda_get_error() для получения кода ошибки;
2. cuda_get_error_str() для получения строки ошибки

## API

1. bool SetArrayZ2Z(int nCols,int nRows,double dFStart, double dFStop, double dAzStart, double dAzStop,double *zArrayin,bool Device);   Device == true GeForce, false Tesla
Возвращает true в случае успеха, false в случае неудачи. Записывает ошибку в error
2. bool CalcZ2Z(int N_1out, int N_2out,double dFStart, double dFStop, double dAzStart, double dAzStop,double *zArrayout); Возвращает true в случае успеха, false в случае неудачи. Записывает ошибку в error
3. bool SetArrayC2C(int nCols,int nRows,float dFStart, float dFStop, float dAzStart, float dAzStop,float *zArrayin,bool Device);
4. bool CalcC2C(int N_1out, int N_2out,float dFStart, float dFStop, float dAzStart, float dAzStop,float *zArrayout);
5. double get_xstart();
6. double get_xstop();
7. double get_zstart();
8. double get_zstop();
9. float f_get_xstart();
10. float f_get_xstop();
11. float f_get_zstart();
12. float f_get_zstop();
