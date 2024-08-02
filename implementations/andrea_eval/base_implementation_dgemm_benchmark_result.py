import plot

# Performance results

columns = ["data/scaled_once_non_triangular/dense_004.txt", "data/scaled_once_non_triangular/dense_008.txt", "data/scaled_once_non_triangular/dense_016.txt", "data/scaled_once_non_triangular/dense_032.txt", "data/scaled_once_non_triangular/dense_048.txt", "data/scaled_once_non_triangular/dense_064.txt", "data/scaled_once_non_triangular/dense_080.txt", "data/scaled_once_non_triangular/dense_096.txt", "data/scaled_once_non_triangular/dense_112.txt", "data/scaled_once_non_triangular/dense_128.txt", "data/scaled_once_non_triangular/dense_positive_004.txt", "data/scaled_once_non_triangular/dense_positive_008.txt", "data/scaled_once_non_triangular/dense_positive_016.txt", "data/scaled_once_non_triangular/dense_positive_032.txt", "data/scaled_once_non_triangular/dense_positive_048.txt", "data/scaled_once_non_triangular/dense_positive_064.txt", "data/scaled_once_non_triangular/dense_positive_080.txt", "data/scaled_once_non_triangular/dense_positive_096.txt", "data/scaled_once_non_triangular/dense_positive_112.txt", "data/scaled_once_non_triangular/dense_positive_128.txt", ] # input matrices
rows = ["baseline_eigen   ", "implementations/base_implementation_dgemm.c", ] # function names

runtime_results = [
[2101.35, 5723.22, 19784.4, 71970.1, 184176, 372324, 907317, 1.57852e+06, 2.33539e+06, 3.43003e+06, 1864.51, 5117.13, 17133.2, 85785.5, 215143, 426300, 937083, 1.57729e+06, 2.04832e+06, 3.42053e+06, ], 
[28017, 44741.5, 143771, 696529, 2.3811e+06, 3.61625e+06, 7.79777e+06, 1.53724e+07, 1.9689e+07, 3.18561e+07, 27153.9, 44314.8, 136760, 814008, 2.40784e+06, 4.00054e+06, 8.57544e+06, 1.53782e+07, 1.81617e+07, 3.13984e+07, ], 
]
perf_results = [
[3.16511, 6.62599, 13.4581, 24.6445, 31.5823, 36.5038, 29.0062, 32.0081, 34.2295, 34.6933, 3.36066, 7.01076, 14.1058, 22.9675, 30.1205, 35.5715, 30.2704, 32.0332, 34.9114, 34.7898, ], 
[0.237392, 0.84758, 1.85197, 2.54644, 2.44286, 3.75838, 3.37505, 3.28678, 4.0601, 3.73552, 0.230759, 0.809548, 1.76717, 2.42047, 2.6913, 3.79052, 3.3078, 3.28552, 3.93741, 3.78997, ], 
]
flops = [
[6651, 37922, 266260, 1773667, 5816687, 13591252, 26317868, 50525581, 79939363, 118999223, 6266, 35875, 241678, 1970278, 6480222, 15164125, 28365851, 50525574, 71509855, 118999237, ], 
[6651, 37922, 266260, 1773667, 5816687, 13591252, 26317868, 50525581, 79939363, 118999223, 6266, 35875, 241678, 1970278, 6480222, 15164125, 28365851, 50525574, 71509855, 118999237, ], 
]

plot.make_all_plots(columns, rows, runtime_results, perf_results, flops)

