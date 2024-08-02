import sys
from jinja2 import Template

l_input = int(sys.argv[1]) #linear unrollings (n*n)
c_input = int(sys.argv[2]) #column wise unrollings (eg at 4x4 matrix, use 4 for full unrolling)
r_input = int(sys.argv[3]) #row wise unrolling (e.g at 4x4 matrix use 1 for full unrolling)
dgemm_on = sys.argv[4].lower() == 'true'
inlined = sys.argv[5].lower() == 'true'

templates=["matrix_operations_template.c", "onenorm_template.c", "eval_functions_template.c", "matrix_exponential_template.c"]

if(not inlined):
    fname = "../bm_implementations/benchmark_optimized.c" 


    with open('includes_template.c') as includes:
        template = Template(includes.read())
    includes.close()

    #first overwrite
    f = open(fname, "w") 
    f.write(template.render(l = l_input, c=c_input, r=r_input, dgemm=dgemm_on, dbg=0, flop_count=False, benchmark=True))
    f.close()


    f = open(fname, "a")

    for templ in templates:
        with open(templ) as current:
            template = Template(current.read())
        current.close()
        f.write(template.render(l = l_input, c=c_input, r=r_input, dgemm=dgemm_on, loop_order='ikj', lu_blas=False, flop_count=False))
    f.close()
else:
    fname = "../func_benchmark/base_func_benchmark_inlined.cpp" 


    with open('includes_template.c') as includes:
        template = Template(includes.read())
    includes.close()

    #first overwrite
    f = open(fname, "w") 
    f.write(template.render(l = l_input, c=c_input, r=r_input, dgemm=dgemm_on, dbg=0, flop_count=False, benchmark=False))
    f.close()


    f = open(fname, "a")

    for templ in templates:
        with open(templ) as current:
            template = Template(current.read())
        current.close()
        f.write(template.render(l = l_input, c=c_input, r=r_input, dgemm=dgemm_on, loop_order='ikj', lu_blas=False, flop_count=False))
    
    with open("benchmark_inlined_template.cpp") as current:
        template = Template(current.read())
    current.close()
    f.write(template.render())
    
    f.close()




