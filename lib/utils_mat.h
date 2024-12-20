#ifndef UTILS_MAT_H
#define UTILS_MAT_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

typedef struct {
    double **values;
    int rows;
    int cols;
} Mat;

Mat *mat_init(int rows, int cols);
void mat_free(Mat *m);
void mat_print(Mat *m);
void mat_printI(Mat *m);
char intToASCII(int val);
void mat_populate(Mat *m, double val);

#endif
