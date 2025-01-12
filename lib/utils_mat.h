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
void mat_populate2(Mat *m);
void mat_populate_rand(Mat *m);
double calc_random(double lowerBound, double upperBound);

void mat_flatten(Mat **m);
void mat_unflatten(Mat **m, int cols);
Mat *mat_multiply(Mat *B, Mat *A);
Mat *mat_multiply_scalar1(double n, Mat *M);
Mat *mat_add1(Mat *dest, Mat *src);
Mat *mat_sub1(Mat *dest, Mat *src);
Mat *mat_sub2(Mat *dest, Mat *src);
Mat *schur_product1(Mat *A, Mat *B);
Mat *schur_product2(Mat *A, Mat *B);
Mat *mat_transpose2(Mat *M);

int maxIndex(Mat *output);


#endif
