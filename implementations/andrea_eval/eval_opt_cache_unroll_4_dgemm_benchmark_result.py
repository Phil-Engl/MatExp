import plot

# Performance results

columns = ["data/scaled_once_non_triangular/dense_004.txt", "data/scaled_once_non_triangular/dense_008.txt", "data/scaled_once_non_triangular/dense_016.txt", "data/scaled_once_non_triangular/dense_032.txt", "data/scaled_once_non_triangular/dense_048.txt", "data/scaled_once_non_triangular/dense_064.txt", "data/scaled_once_non_triangular/dense_080.txt", "data/scaled_once_non_triangular/dense_096.txt", "data/scaled_once_non_triangular/dense_112.txt", "data/scaled_once_non_triangular/dense_128.txt", "data/scaled_once_non_triangular/dense_positive_004.txt", "data/scaled_once_non_triangular/dense_positive_008.txt", "data/scaled_once_non_triangular/dense_positive_016.txt", "data/scaled_once_non_triangular/dense_positive_032.txt", "data/scaled_once_non_triangular/dense_positive_048.txt", "data/scaled_once_non_triangular/dense_positive_064.txt", "data/scaled_once_non_triangular/dense_positive_080.txt", "data/scaled_once_non_triangular/dense_positive_096.txt", "data/scaled_once_non_triangular/dense_positive_112.txt", "data/scaled_once_non_triangular/dense_positive_128.txt", ] # input matrices
rows = ["baseline_eigen   ", "implementations/eval_opt_cache_unroll_4_dgemm.c", ] # function names

runtime_results = [
[1933.38, 5400.98, 18358.7, 65567.4, 165393, 338350, 828806, 1.41211e+06, 2.11472e+06, 3.12966e+06, 1773.13, 4828.43, 16051.4, 75902.1, 195096, 406187, 842425, 1.45267e+06, 1.89947e+06, 3.20469e+06, ], 
[30129.9, 45519.7, 135467, 643855, 2.14773e+06, 3.72663e+06, 7.29676e+06, 1.44273e+07, 1.84269e+07, 2.93328e+07, 28651.2, 43546.1, 126557, 701305, 2.33242e+06, 3.73306e+06, 7.78175e+06, 1.44294e+07, 1.69732e+07, 3.02939e+07, ], 
]
perf_results = [
[3.43647, 7.02002, 14.5028, 27.051, 35.1688, 40.1692, 31.7539, 35.7803, 37.8013, 38.023, 3.52992, 7.4285, 15.0561, 25.9581, 33.2155, 37.3328, 33.6716, 34.7811, 37.6472, 37.1328, ], 
[0.220512, 0.832936, 1.96545, 2.75475, 2.70829, 3.64706, 3.60679, 3.50209, 4.33818, 4.05687, 0.218455, 0.82368, 1.90959, 2.80944, 2.77832, 4.06212, 3.64518, 3.50156, 4.21311, 3.92816, ], 
]
flops = [
[6644, 37915, 266253, 1773660, 5816680, 13591245, 26317861, 50525574, 79939356, 118999216, 6259, 35868, 241671, 1970271, 6480215, 15164118, 28365844, 50525567, 71509848, 118999230, ], 
[6644, 37915, 266253, 1773660, 5816680, 13591245, 26317861, 50525574, 79939356, 118999216, 6259, 35868, 241671, 1970271, 6480215, 15164118, 28365844, 50525567, 71509848, 118999230, ], 
]

plot.make_all_plots(columns, rows, runtime_results, perf_results, flops)

