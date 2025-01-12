#include "activationFunctions.h"

// Running the ReLu activation function on input
double relu(double input) {
    if (input < 0) {
        return 0;
    } else {
        return input;
    }
}

// Running the derivative of the ReLu function on input
double drelu(double input) {
    if (input < 0) {
        return 0;
    } else {
        return 1;
    }
}

// Running the sigmoid function on input
double sigmoid(double input) { return 1.0 / (1 + exp(-input)); }

// Running the derivative of the sigmoid function on input
double dsigmoid(double input) {
    // double val = exp(input) / pow(exp(input) + 1, 2);
    // if (isnan(val)) {
    //     printf("Nan from input of %lf\n", input);
    //     exit(EXIT_FAILURE);
    // }

    double val = sigmoid(input) * (1 - sigmoid(input));
    return val;
}

// Running the ReLu function on all values of an input matrix m
Mat *apply1(double (*ptr)(double), Mat *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->values[i][j] = ptr(m->values[i][j]);
        }
    }
    return m;
}

// Creating a Mat* with ReLu applied to all values of Mat *m
Mat *apply2(double (*ptr)(double), Mat *m) {
    Mat *result = mat_init(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result->values[i][j] = ptr(m->values[i][j]);
        }
    }
    return result;
}
