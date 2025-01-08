#include "backprop.h"

// Calculating the error for the last layer assuming a quadratic cost function
Mat *error_output(double (*actFnct)(double), Mat *input, Mat **weights,
                  Mat **biases, int label) {
    // Mat *errorOutput = mat_init(OUTPUT_SIZE, 1);
    Network *net = net_init(weights, biases);

    Mat *preOutput = prePropagate(actFnct, input, net);
    printf("preOutput:\n");
    mat_print(preOutput);

    Mat *output = mat_init(OUTPUT_SIZE, 1);
    Mat *derivOutput = mat_init(OUTPUT_SIZE, 1);
    Mat *expectedVal = mat_init(OUTPUT_SIZE, 1);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (i == label) {
            expectedVal->values[i][0] = 1.0;
        } else {
            expectedVal->values[i][0] = 0.0;
        }
        output->values[i][0] = sigmoid(preOutput->values[i][0]);
        derivOutput->values[i][0] = dsigmoid(preOutput->values[i][0]);
    }
    printf("Output:\n");
    mat_print(output);
    printf("derivOutput:\n");
    mat_print(derivOutput);
    printf("expectedVal:\n");
    mat_print(expectedVal);

    Mat *errorOutput = schur_product(mat_sub(output, expectedVal), derivOutput);

    net_free(net);
    mat_free(preOutput);
    mat_free(output);
    mat_free(derivOutput);
    mat_free(expectedVal);

    return errorOutput;
};
