import plot

# Performance results

columns = ["data/scaled_once_non_triangular/dense_004.txt", "data/scaled_once_non_triangular/dense_008.txt", "data/scaled_once_non_triangular/dense_016.txt", "data/scaled_once_non_triangular/dense_032.txt", "data/scaled_once_non_triangular/dense_048.txt", "data/scaled_once_non_triangular/dense_064.txt", "data/scaled_once_non_triangular/dense_080.txt", "data/scaled_once_non_triangular/dense_096.txt", "data/scaled_once_non_triangular/dense_112.txt", "data/scaled_once_non_triangular/dense_128.txt", "data/scaled_once_non_triangular/dense_positive_004.txt", "data/scaled_once_non_triangular/dense_positive_008.txt", "data/scaled_once_non_triangular/dense_positive_016.txt", "data/scaled_once_non_triangular/dense_positive_032.txt", "data/scaled_once_non_triangular/dense_positive_048.txt", "data/scaled_once_non_triangular/dense_positive_064.txt", "data/scaled_once_non_triangular/dense_positive_080.txt", "data/scaled_once_non_triangular/dense_positive_096.txt", "data/scaled_once_non_triangular/dense_positive_112.txt", "data/scaled_once_non_triangular/dense_positive_128.txt", ] # input matrices
rows = ["baseline_eigen   ", "implementations/eval_opt_simd_dgemm.c", ] # function names

runtime_results = [
[2089.48, 5788.41, 20012.3, 72802.9, 183582, 370329, 689043, 1.55776e+06, 2.32993e+06, 3.41422e+06, 1865.12, 5304.38, 17131.6, 80903.6, 215289, 433707, 697012, 1.48733e+06, 2.05439e+06, 3.42937e+06, ], 
[31397.8, 47906.2, 142373, 689827, 2.41424e+06, 3.70847e+06, 7.74278e+06, 1.53193e+07, 1.95879e+07, 3.12102e+07, 30318.1, 46089, 134445, 752145, 2.398e+06, 3.996e+06, 8.15622e+06, 1.53368e+07, 1.80402e+07, 3.01798e+07, ], 
]
perf_results = [
[3.14911, 6.50594, 13.2533, 24.3062, 31.6342, 36.6563, 38.1576, 32.4112, 34.2883, 34.8348, 3.3215, 6.7137, 14.047, 24.3027, 30.0572, 34.9262, 40.6596, 33.946, 34.7838, 34.6809, ], 
[0.209569, 0.786099, 1.86292, 2.56523, 2.4055, 3.6605, 3.39571, 3.29576, 4.07849, 3.81073, 0.204333, 0.772679, 1.78993, 2.61409, 2.6985, 3.79072, 3.47468, 3.292, 3.96114, 3.94083, ], 
]
flops = [
[6580, 37659, 265229, 1769564, 5807464, 13574861, 26292261, 50488710, 79889180, 118933680, 6195, 35612, 240647, 1966175, 6470999, 15147734, 28340244, 50488703, 71459672, 118933694, ], 
[6580, 37659, 265229, 1769564, 5807464, 13574861, 26292261, 50488710, 79889180, 118933680, 6195, 35612, 240647, 1966175, 6470999, 15147734, 28340244, 50488703, 71459672, 118933694, ], 
]

plot.make_all_plots(columns, rows, runtime_results, perf_results, flops)

