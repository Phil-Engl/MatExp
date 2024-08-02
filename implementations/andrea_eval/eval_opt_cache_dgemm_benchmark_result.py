import plot

# Performance results

columns = ["data/scaled_once_non_triangular/dense_004.txt", "data/scaled_once_non_triangular/dense_008.txt", "data/scaled_once_non_triangular/dense_016.txt", "data/scaled_once_non_triangular/dense_032.txt", "data/scaled_once_non_triangular/dense_048.txt", "data/scaled_once_non_triangular/dense_064.txt", "data/scaled_once_non_triangular/dense_080.txt", "data/scaled_once_non_triangular/dense_096.txt", "data/scaled_once_non_triangular/dense_112.txt", "data/scaled_once_non_triangular/dense_128.txt", "data/scaled_once_non_triangular/dense_positive_004.txt", "data/scaled_once_non_triangular/dense_positive_008.txt", "data/scaled_once_non_triangular/dense_positive_016.txt", "data/scaled_once_non_triangular/dense_positive_032.txt", "data/scaled_once_non_triangular/dense_positive_048.txt", "data/scaled_once_non_triangular/dense_positive_064.txt", "data/scaled_once_non_triangular/dense_positive_080.txt", "data/scaled_once_non_triangular/dense_positive_096.txt", "data/scaled_once_non_triangular/dense_positive_112.txt", "data/scaled_once_non_triangular/dense_positive_128.txt", ] # input matrices
rows = ["baseline_eigen   ", "implementations/eval_opt_cache_dgemm.c", ] # function names

runtime_results = [
[2101.76, 5764.28, 19975.2, 71751.9, 183636, 373551, 898598, 1.5567e+06, 2.32653e+06, 3.41667e+06, 1900.47, 5083.92, 17078.9, 80531.3, 210908, 438516, 705554, 1.50859e+06, 2.06894e+06, 3.46109e+06, ], 
[31804.6, 48023.8, 141017, 687235, 2.2922e+06, 3.93944e+06, 7.84039e+06, 1.54504e+07, 1.9971e+07, 3.16116e+07, 30771.3, 45664, 133230, 757141, 2.3406e+06, 3.95895e+06, 8.13227e+06, 1.55195e+07, 1.81583e+07, 3.19346e+07, ], 
]
perf_results = [
[3.18258, 6.57758, 13.3292, 24.7193, 31.675, 36.3839, 29.2877, 32.4569, 34.3599, 34.829, 3.31707, 7.05519, 14.1503, 24.4659, 30.7254, 34.5805, 40.2037, 33.492, 34.5635, 34.382, ], 
[0.210315, 0.789505, 1.88809, 2.58086, 2.53759, 3.45005, 3.3567, 3.27019, 4.00278, 3.76441, 0.204866, 0.785476, 1.81394, 2.60225, 2.76861, 3.83034, 3.48806, 3.25562, 3.93813, 3.72634, ], 
]
flops = [
[6689, 37915, 266253, 1773660, 5816680, 13591245, 26317861, 50525574, 79939356, 118999216, 6304, 35868, 241671, 1970271, 6480215, 15164118, 28365844, 50525567, 71509848, 118999230, ], 
[6689, 37915, 266253, 1773660, 5816680, 13591245, 26317861, 50525574, 79939356, 118999216, 6304, 35868, 241671, 1970271, 6480215, 15164118, 28365844, 50525567, 71509848, 118999230, ], 
]

plot.make_all_plots(columns, rows, runtime_results, perf_results, flops)

