import sys
from jinja2 import Template

l_input = int(sys.argv[1]) #linear unrollings (n*n)
c_input = int(sys.argv[2]) #column wise unrollings (eg at 4x4 matrix, use 4 for full unrolling)
r_input = int(sys.argv[3]) #row wise unrolling (e.g at 4x4 matrix use 1 for full unrolling)
dgemm_on = sys.argv[4].lower() == 'true' 
debug = int(sys.argv[5])
loop = sys.argv[6].lower() == 'ikj' #either ikj or kij
LU_BLAS = sys.argv[7].lower() == 'true'
fc = sys.argv[8].lower() == 'true'
profiling = sys.argv[9].lower() == 'true'
implementation_name = sys.argv[10].lower()

templates=["matrix_operations_template.c", "onenorm_template.c", "eval_functions_template.c", "matrix_exponential_template.c"]

fname = ""
if(not profiling):
    fname = "../implementations/" + implementation_name 
else:
    fname = "../profiling/" + implementation_name

testn = ""
if(debug == 1):
    testn = "../test_" + implementation_name

# +"_unrolled_" + str(l_input) + "_"
# if(dgemm_on):
#     fname = fname + "with_blas.c"
# else:
#     fname = fname + "without_blas.c"

with open('includes_template.c') as includes:
    template = Template(includes.read())
includes.close()

#first overwrite
f = open(fname, "w") 
f.write(template.render(l = l_input, c=c_input, r=r_input, dgemm=dgemm_on, dbg=debug, flop_count=fc, benchmark=False))
f.close()

if(debug == 1):
    d = open(testn, "w")
    d.write(template.render(l = l_input, c=c_input, r=r_input, dgemm=dgemm_on, dbg=debug, flop_count=fc, benchmark=False))
    d.close()

f = open(fname, "a")
if(debug == 1):
    d = open(testn, "a")

for templ in templates:
    with open(templ) as current:
        template = Template(current.read())
    current.close()
    f.write(template.render(l = l_input, c=c_input, r=r_input, dgemm=dgemm_on, loop_order=loop, lu_blas=LU_BLAS, flop_count=fc, benchmark=False))
    if(debug == 1):
        d.write(template.render(l = l_input, c=c_input, r=r_input, dgemm=dgemm_on, loop_order=loop, lu_blas=LU_BLAS, flop_count=fc, benchmark=False))


if(profiling):
    with open("perf_template.c") as profile_template:
        template = Template(profile_template.read())
    profile_template.close()
    f.write(template.render())

f.close()

if(debug == 1):
    with open("testing_template.c") as testing_template:
        template = Template(testing_template.read())
    testing_template.close()
    d.write(template.render())
    d.close()


