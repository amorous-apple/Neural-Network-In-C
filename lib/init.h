#ifndef INIT_H
#define INIT_H

#include <stdlib.h>
#include <stdio.h>

extern int MAT_SIZE;
extern int OUTPUT_SIZE;
extern int MAX_LINE_LEN;

extern int NUM_H_LAYERS;
extern int *NUM_H_LAYER_NODES;

void init(int argc, char **argv);

#endif
