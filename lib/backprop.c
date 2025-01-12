#include "backprop.h"

// Calculating the error for the last layer assuming a quadratic cost function
Mat *error_output(double (*actFnct)(double), Mat *input, Network *net,
                  int label) {
    propagate(actFnct, input, net);

    Mat *preOutput = net->preLayers[NUM_H_LAYERS];
    Mat *output = net->layers[NUM_H_LAYERS];

    Mat *expectedVal = mat_init(OUTPUT_SIZE, 1);
    Mat *derivOutput = mat_init(OUTPUT_SIZE, 1);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (i == label) {
            expectedVal->values[i][0] = 1.0;
        } else {
            expectedVal->values[i][0] = 0.0;
        }
        derivOutput->values[i][0] = dsigmoid(preOutput->values[i][0]);
    }
    // printf("preOutput:\n");
    // mat_print(preOutput);
    // printf("Output:\n");
    // mat_print(output);
    // printf("derivOutput:\n");
    // mat_print(derivOutput);
    // printf("expectedVal:\n");
    // mat_print(expectedVal);

    Mat *errorOutput =
        schur_product1(mat_sub2(output, expectedVal), derivOutput);

    mat_free(expectedVal);
    mat_free(derivOutput);

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
        Mat *derivOutput = mat_init(NUM_LAYER_NODES[i], 1);
        for (int j = 0; j < NUM_LAYER_NODES[i]; j++) {
            derivOutput->values[j][0] =
                dsigmoid(net->preLayers[i]->values[j][0]);
        }
        Mat *Tweights = mat_transpose2(weights[i + 1]);
        errors[i] =
            schur_product1(mat_multiply(Tweights, errors[i + 1]), derivOutput);

        mat_free(Tweights);
        mat_free(derivOutput);
    }

    net_free(net);

    return errors;
}

// Updating the weights and biases based on a starting point and the number of
// inputs to be used (a single batch of training)
void update_weights(double (*actFnct)(double), Mat **inputs, Mat **weights,
                    Mat **biases, int *labels, int startingIndex) {
    Mat ***batchErrors = malloc(BATCH_SIZE * sizeof(Mat **));
    if (batchErrors == NULL) {
        perror("Error allocating memory for batchErrors\n");
        exit(EXIT_FAILURE);
    }

// Calculating and storing all of the errors
#pragma omp parallel for
    for (int i = 0; i < BATCH_SIZE; i++) {
        int index = startingIndex + i;
        batchErrors[i] =
            calc_errors(actFnct, inputs[index], weights, biases, labels[index]);
    }

    // Averaging the errors for a batch
    for (int i = 1; i < BATCH_SIZE; i++) {
        for (int j = 0; j < NUM_H_LAYERS + 1; j++) {
            mat_add1(batchErrors[0][j], batchErrors[i][j]);
        }
    }
    // Weighting the average by the learning rate
    for (int i = 0; i < NUM_H_LAYERS + 1; i++) {
        mat_multiply_scalar1(((-1.0 * LEARNING_RATE) / BATCH_SIZE),
                             batchErrors[0][i]);
    }

    // printf("OG Biases: \n");
    // mat_print(biases[1]);
    // printf("OG Weights: \n");
    // mat_print(weights[1]);
    // printf("Weighted bias dependence: \n");
    // mat_print(batchErrors[0][1]);

    // Updating the biases
    for (int i = 0; i < NUM_H_LAYERS + 1; i++) {
        mat_add1(biases[i], batchErrors[0][i]);
    }
    // printf("New Biases: \n");
    // mat_print(biases[1]);
}

// Training the weights and biases
void traininator(double (*actFnct)(double), Mat **inputs, Mat **weights,
                 Mat **biases, int *labels) {
    int num_batches = TRAINING_DATA_SIZE / BATCH_SIZE;
    // int num_batches = 1;

    for (int i = 0; i < NUM_EPOCHS; i++) {
        for (int j = 0; j < num_batches; j++) {
            // printf("Batch %d / %d\n", j + 1, num_batches);
            int startingIndex = j * BATCH_SIZE;
            update_weights(actFnct, inputs, weights, biases, labels,
                           startingIndex);
        }
        test_weights(weights, biases);
    }
}
