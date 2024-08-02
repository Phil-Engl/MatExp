#include "utils.h"

long flop_count = 0;

void inc_flop(long c){
    flop_count += c;
}

long get_flop_count(){
    return flop_count;
}

void reset_flop_count(){
    flop_count = 0;
}