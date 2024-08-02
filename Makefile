PATH_TO_EIGEN =/usr/local/include/eigen-3.4.0/
CC=g++ # or clang
CFLAGS=-I $(PATH_TO_EIGEN)
FLAGS=-O3 -fno-tree-vectorize -ffp-contract=off -Wall -march=native -ffast-math # for clang: -std=c++17 -lstdc++
DEBUG_FLAGS=-ggdb3 -O0 -Wall

LIBS=-lm -lcblas -lopenblas

# change this if you want test optimizations in other folders
IMPLEMENTATION_FOLDER=base_implementation
OPTIMIZATION_FOLDER=implementation

IMPLEMENTATION_FILE=dummy # set this 

BENCHMARK_OUT = benchmark.out
TEST_OUT = test.out
EIGEN_TEST_OUT = eigen_test.out
BENCHMARK_FUNCS_OUT = benchmark_functions.out

BENCHMARK_OBJECTS = benchmark/eigen_wrapper.o main.o comp.o

EIGEN_TEST_OBJECTS = benchmark/eigen_wrapper.o 

MAT_EXP_OBJECTS = ${IMPLEMENTATION_FOLDER}/matrix_exponential.o ${IMPLEMENTATION_FOLDER}/onenorm.o ${IMPLEMENTATION_FOLDER}/matrix_operations.o ${IMPLEMENTATION_FOLDER}/LU_wrapper.o

BENCHMARK_FUNCTION_OBJECTS = func_benchmark/base_func_benchmark.o
BENCHMARK_BASELINE_OBJECTS = func_benchmark/baseline_benchmark.o
JINJA_OBJECTS= bm_implementations/benchmark_optimized.o



benchmark-optimized: $(BENCHMARK_FUNCTION_OBJECTS) $(JINJA_OBJECTS)
	$(CC) $(FLAGS) -o optimize_version_benchmark.out  $(BENCHMARK_FUNCTION_OBJECTS) $(JINJA_OBJECTS) $(LIBS) 

benchmark-optimized-inlined: 
	$(CC) $(FLAGS) -o optimize_version_benchmark_in.out func_benchmark/base_func_benchmark_inlined.cpp $(LIBS)

bm_implementations/benchmark_optimized.o: bm_implementations/benchmark_optimized.c
	$(CC) $(FLAGS) -c -o bm_implementations/benchmark_optimized.o bm_implementations/benchmark_optimized.c

benchmark-jinja: $(IMPLEMENTATION_FILE).o $(BENCHMARK_OBJECTS)
	$(CC) -o $(IMPLEMENTATION_FILE)_benchmark.out $(CFLAGS) $(BENCHMARK_OBJECTS) $(IMPLEMENTATION_FILE).o $(LIBS) $(FLAGS) -lfmt


benchmark-function-base: $(BENCHMARK_BASELINE_OBJECTS) $(MAT_EXP_OBJECTS)
	$(CC) -o $(BENCHMARK_FUNCS_OUT) $(CFLAGS) $(BENCHMARK_BASELINE_OBJECTS) $(MAT_EXP_OBJECTS) $(LIBS) $(FLAGS)

benchmark: $(BENCHMARK_OBJECTS) $(MAT_EXP_OBJECTS)
	$(CC) -o $(BENCHMARK_OUT) $(CFLAGS) $(BENCHMARK_OBJECTS) $(MAT_EXP_OBJECTS) $(LIBS) $(FLAGS) -lfmt

test: ${IMPLEMENTATION_FOLDER}/test.c $(MAT_EXP_OBJECTS)
	$(CC) -o $(TEST_OUT) ${IMPLEMENTATION_FOLDER}/test.c $(MAT_EXP_OBJECTS) $(LIBS) $(FLAGS)

eigen_test: $(EIGEN_TEST_OBJECTS) $(MAT_EXP_OBJECTS) eigen_test.c 
	$(CC) -o $(EIGEN_TEST_OUT) $(CFLAGS) eigen_test.c $(EIGEN_TEST_OBJECTS) $(MAT_EXP_OBJECTS) $(LIBS) $(FLAGS)

flop_count_main: $(MAT_EXP_OBJECTS) ${IMPLEMENTATION_FOLDER}/utils.o flop_count_main.cpp
	$(CC) -o flop_count_main.out flop_count_main.cpp ${IMPLEMENTATION_FOLDER}/utils.o $(MAT_EXP_OBJECTS) $(LIBS) $(FLAGS)

flop_count_single: $(IMPLEMENTATION_FILE).o flop_count_main.cpp base_implementation/utils.o
	$(CC) -o flop_count_main.out flop_count_main.cpp $(IMPLEMENTATION_FILE).o base_implementation/utils.o $(LIBS) $(FLAGS)


debug_benchmark: benchmark/eigen_wrapper.cpp main.cpp comp.cpp ${IMPLEMENTATION_FOLDER}/matrix_exponential.c ${IMPLEMENTATION_FOLDER}/onenorm.c ${IMPLEMENTATION_FOLDER}/matrix_operations.c
	$(CC) -g -o debug_$(BENCHMARK_OUT) $(CFLAGS) \
	benchmark/eigen_wrapper.cpp main.cpp comp.cpp \
	${IMPLEMENTATION_FOLDER}/matrix_exponential.c ${IMPLEMENTATION_FOLDER}/onenorm.c ${IMPLEMENTATION_FOLDER}/matrix_operations.c \
	$(LIBS) $(DEBUG_FLAGS) -lfmt

debug_test: ${IMPLEMENTATION_FOLDER}/test.c ${IMPLEMENTATION_FOLDER}/matrix_exponential.c ${IMPLEMENTATION_FOLDER}/onenorm.c ${IMPLEMENTATION_FOLDER}/matrix_operations.c
	$(CC) -g -o debug_$(TEST_OUT) ${IMPLEMENTATION_FOLDER}/test.c ${IMPLEMENTATION_FOLDER}/matrix_exponential.c ${IMPLEMENTATION_FOLDER}/onenorm.c ${IMPLEMENTATION_FOLDER}/matrix_operations.c \
	$(LIBS) $(DEBUG_FLAGS)

debug_eigen_test: benchmark/eigen_wrapper.cpp eigen_test.c ${IMPLEMENTATION_FOLDER}/matrix_exponential.c ${IMPLEMENTATION_FOLDER}/onenorm.c ${IMPLEMENTATION_FOLDER}/matrix_operations.c
	$(CC) -g -o debug_$(EIGEN_TEST_OUT) $(CFLAGS) benchmark/eigen_wrapper.cpp \
	eigen_test.c ${IMPLEMENTATION_FOLDER}/matrix_exponential.c ${IMPLEMENTATION_FOLDER}/onenorm.c ${IMPLEMENTATION_FOLDER}/matrix_operations.c \
	$(LIBS) $(DEBUG_FLAGS)

benchmark/eigen_wrapper.o: benchmark/eigen_wrapper.h benchmark/eigen_wrapper.cpp 
	$(CC) $(CFLAGS) $(FLAGS) -o benchmark/eigen_wrapper.o -c benchmark/eigen_wrapper.cpp 

${IMPLEMENTATION_FOLDER}/LU_wrapper.o: ${IMPLEMENTATION_FOLDER}/LU_wrapper.h ${IMPLEMENTATION_FOLDER}/LU_wrapper.cpp
	$(CC) $(CFLAGS) $(FLAGS) -o ${IMPLEMENTATION_FOLDER}/LU_wrapper.o -c ${IMPLEMENTATION_FOLDER}/LU_wrapper.cpp


main.o: benchmark/eigen_wrapper.h benchmark/utils.h benchmark/tsc_x86.h main.cpp
	$(CC) $(CFLAGS) $(FLAGS) -c main.cpp

comp.o: comp.cpp
	$(CC) $(CFLAGS) $(FLAGS) -c comp.cpp

${IMPLEMENTATION_FOLDER}/matrix_exponential.o: ${IMPLEMENTATION_FOLDER}/matrix_exponential.c
	$(CC) $(CFLAGS) $(FLAGS) -o ${IMPLEMENTATION_FOLDER}/matrix_exponential.o -c ${IMPLEMENTATION_FOLDER}/matrix_exponential.c $(LIBS)

${IMPLEMENTATION_FOLDER}/onenorm.o: ${IMPLEMENTATION_FOLDER}/onenorm.c
	$(CC) $(CFLAGS) $(FLAGS) -o ${IMPLEMENTATION_FOLDER}/onenorm.o -c ${IMPLEMENTATION_FOLDER}/onenorm.c $(LIBS)

${IMPLEMENTATION_FOLDER}/matrix_operations.o: ${IMPLEMENTATION_FOLDER}/matrix_operations.c
	$(CC) $(CFLAGS) $(FLAGS) -o ${IMPLEMENTATION_FOLDER}/matrix_operations.o -c ${IMPLEMENTATION_FOLDER}/matrix_operations.c $(LIBS)

$(IMPLEMENTATION_FILE).o: $(IMPLEMENTATION_FILE).c
	$(CC) $(CFLAGS) $(FLAGS) -o $(IMPLEMENTATION_FILE).o -c $(IMPLEMENTATION_FILE).c $(LIBS)

clean: 
	rm -f *.o *.out $(IMPLEMENTATION_FOLDER)/*.o
