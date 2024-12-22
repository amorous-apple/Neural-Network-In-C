#include "activationFunctions.h"

// Running the ReLu activation function on input
double relu(double input) {
    if (input < 0) {
        return 0;
    } else {
        return input;
    }
}

// Running the sigmoid function in input
double sigmoid(double input) { return 1.0 / (1 + exp(-input)); }

// Running the ReLu function on all values of an input matrix m
Mat *apply(double (*ptr)(double), Mat *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->values[i][j] = ptr(m->values[i][j]);
        }
    }
    return m;
}
