# Cuda fft dll
##error handling. Для получения кода ошибки и номера строки: 

1. cuda_get_error() для получения кода ошибки;
2. cuda_get_error_str() для получения строки ошибки

## API

1. ConvertZ2Z(int N_1in, int N_2in, double FI, double2 *Array) Выходной массив записывается в Array
Возвращает true в случае успеха, false в случае неудачи. Записывает ошибку в error
2. Init_Plan(int N_1out, int N_2out) Возвращает true в случае успеха, false в случае неудачи. Записывает ошибку в error
3. Cuda_Free() Высвобождает ресурсы
