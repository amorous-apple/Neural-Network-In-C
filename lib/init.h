#ifndef INIT_H
#define INIT_H

#include <stdlib.h>
#include <stdio.h>
#include "utils_mat.h"
#include "utils_files.h"

extern const int MAT_SIZE;
extern const int OUTPUT_SIZE;
extern const int MAX_LINE_LEN;
extern const int TEST_DATA_SIZE;
extern const int TRAINING_DATA_SIZE;

extern int NUM_H_LAYERS;
extern int *NUM_LAYER_NODES;

extern double LEARNING_RATE;
extern double LEARNING_RATE_TERM;
extern int NUM_EPOCHS;
extern int BATCH_SIZE;

extern const double REGULARIZATION_PARAMETER;
extern double REGULARIZATION_TERM;

void init(int argc, char **argv);
Mat **init_trainingData(int *labels);

#endif
