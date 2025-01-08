#ifndef INIT_H
#define INIT_H

#include <stdlib.h>
#include <stdio.h>

extern const int MAT_SIZE;
extern const int OUTPUT_SIZE;
extern const int MAX_LINE_LEN;

extern double LEARNING_RATE;
extern int NUM_H_LAYERS;
extern int *NUM_H_LAYER_NODES;

void init(int argc, char **argv);

#endif
