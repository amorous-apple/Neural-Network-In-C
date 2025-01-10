#include "backprop.h"

// Calculating the error for the last layer assuming a quadratic cost function
Mat *error_output(double (*actFnct)(double), Mat *input, Network *net,
                  int label) {
    propagate(actFnct, input, net);

    Mat *preOutput = net->preLayers[NUM_H_LAYERS];
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

    mat_free(preOutput);
    mat_free(output);
    mat_free(derivOutput);
    mat_free(expectedVal);

    return errorOutput;
};

// Calculating all of the errors assuming a quadratic cost function
Mat **calc_errors(double (*actFnct)(double), Mat *input, Mat **weights,
                  Mat **biases, int label) {
    Mat **errors = malloc((NUM_H_LAYERS + 1) * sizeof(Mat *));
    if (errors == NULL) {
        perror("Error allocating memory for **errors\n");
        exit(EXIT_FAILURE);
    }

    Network *net = net_init(weights, biases);
    errors[NUM_H_LAYERS] = error_output(actFnct, input, net, label);

    // printf("derivVals: \n");
    // mat_print(net->hiddenLayers[0]);

    for (int i = NUM_H_LAYERS - 1; i >= 0; i--) {
        errors[i] = schur_product(
            mat_multiply(mat_transpose(weights[i + 1]), errors[i + 1]),
            net->layers[i]);
    }

    net_free(net);

    return errors;
}
