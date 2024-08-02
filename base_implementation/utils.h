
#ifndef DEBUG
    #define DEBUG 0
#endif


#ifndef COUNT_FLOPS
    #define COUNT_FLOPS 0 // only set this to 1 if you execute flop_count_main.cpp
    #define GET_FLOP_COUNT() do{if(COUNT_FLOPS){printf("FLOP COUNT: %ld flops\n", get_flop_count());}}while(0)
    #define FLOP_COUNT_INC(x, str) do{if(COUNT_FLOPS){inc_flop(x); printf("FLOP_COUNT: %s added %ld flops\n", str, x);}}while(0)
    #define RESET_FLOP_COUNT() do{if(COUNT_FLOPS){reset_flop_count();}}while(0)
#endif


void inc_flop(long c);

long get_flop_count();

void reset_flop_count();