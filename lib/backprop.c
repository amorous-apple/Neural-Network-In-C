#include "backprop.h"

// Calculating the error for the last layer assuming a quadratic cost function
Mat *error_output(double (*actFnct)(double), Mat *input, Network *net,
                  int label, Mat *errorOutMat) {
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

    // // errorOutput based on the quadratic cost function
    // schur_product1(mat_subExt(output, expectedVal, errorOutMat),
    // derivOutput);

    // errorOutput based on cross-entropy
    mat_subExt(output, expectedVal, errorOutMat);

    mat_free(expectedVal);
    mat_free(derivOutput);

    return errorOutMat;
};

// Calculating all of the errors assuming a quadratic cost function
Mat **calc_errors(double (*actFnct)(double), Mat *input, Network *net,
                  int label, Mat **errors) {
    error_output(actFnct, input, net, label, errors[NUM_H_LAYERS]);

    for (int i = NUM_H_LAYERS - 1; i >= 0; i--) {
        Mat *derivOutput = mat_init(NUM_LAYER_NODES[i], 1);
        for (int j = 0; j < NUM_LAYER_NODES[i]; j++) {
            derivOutput->values[j][0] =
                dsigmoid(net->preLayers[i]->values[j][0]);
        }
        Mat *Tweights = mat_transpose2(net->weights[i + 1]);
        mat_multiplyExt(Tweights, errors[i + 1], errors[i]);
        schur_product1(errors[i], derivOutput);

        mat_free(derivOutput);
        mat_free(Tweights);
        // mat_free(prod);
    }

    return errors;
}

// Updating the weights and biases based on a starting point in the data set
void update_weights(double (*actFnct)(double), Mat **inputs, Mat **weights,
                    Mat **biases, int *labels, int startingIndex,
                    Network **nets, Mat ***batchErrors) {
// Calculating and storing all of the errors
#pragma omp parallel for
    for (int i = 0; i < BATCH_SIZE; i++) {
        int index = startingIndex + i;
        batchErrors[i] = calc_errors(actFnct, inputs[index], nets[i],
                                     labels[index], batchErrors[i]);
    }

    double learningConst = -LEARNING_RATE / BATCH_SIZE;

    // Updating the weights for the input layer
    Mat **weightsShift = init_weightsZ();
#pragma omp parallel for
    for (int i = 0; i < BATCH_SIZE; i++) {
        int index = startingIndex + i;

        Mat *inputT = mat_transpose2(inputs[index]);
        Mat *prod = mat_multiply(batchErrors[i][0], inputT);
        mat_add1(weightsShift[0], prod);

        mat_free(inputT);
        mat_free(prod);
    }
    mat_multiply_scalar1(learningConst, weightsShift[0]);
    mat_add1(weights[0], weightsShift[0]);
    mat_free(weightsShift[0]);

// Updating the remaining weights
#pragma omp parallel for
    for (int i = 1; i < NUM_H_LAYERS + 1; i++) {
        for (int j = 0; j < BATCH_SIZE; j++) {
            Mat *Ttemp = mat_transpose2(nets[j]->layers[i - 1]);
            Mat *prod = mat_multiply(batchErrors[j][i], Ttemp);
            mat_add1(weightsShift[i], prod);

            mat_free(Ttemp);
            mat_free(prod);
        }
    }
    for (int i = 1; i < NUM_H_LAYERS + 1; i++) {
        mat_free(weightsShift[i]);
    }
    free(weightsShift);

    // Averaging the errors for a batch
    for (int i = 1; i < BATCH_SIZE; i++) {
        for (int j = 0; j < NUM_H_LAYERS + 1; j++) {
            mat_add1(batchErrors[0][j], batchErrors[i][j]);
        }
    }
    // Weighting the average by the learning rate
    for (int i = 1; i < NUM_H_LAYERS + 1; i++) {
        mat_multiply_scalar1(learningConst, batchErrors[0][i]);
    }

    // Updating the biases
    for (int i = 0; i < NUM_H_LAYERS + 1; i++) {
        mat_add1(biases[i], batchErrors[0][i]);
    }
}

// Training the weights and biases
void traininator(double (*actFnct)(double), Mat **inputs, Mat **weights,
                 Mat **biases, int *labels) {
    int num_batches = TRAINING_DATA_SIZE / BATCH_SIZE;

    FILE *testData = openInputFile("./data/mnist_test.csv");

    // Loading all of the data into an array of matrices and labels
    int *trainingLabels = init_labels(TEST_DATA_SIZE);

    Mat **trainingInputs = malloc(TEST_DATA_SIZE * sizeof(Mat *));
    if (trainingInputs == NULL) {
        perror("Error allocating memory for trainingInputs\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < TEST_DATA_SIZE; i++) {
        trainingInputs[i] = dataToMat(testData, &trainingLabels[i]);
    }
    fclose(testData);

    int *guesses = malloc(TEST_DATA_SIZE * sizeof(int));
    if (guesses == NULL) {
        perror("Error allocating memory for guesses\n");
        exit(EXIT_FAILURE);
    }

    Network **nets = malloc(BATCH_SIZE * sizeof(Network *));
    if (nets == NULL) {
        perror("Error allocating memory for nets\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < BATCH_SIZE; i++) {
        nets[i] = net_init(weights, biases);
    }

    Mat ***batchErrors = malloc(BATCH_SIZE * sizeof(Mat **));
    if (batchErrors == NULL) {
        perror("Error allocating memory for batchErrors\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < BATCH_SIZE; i++) {
        batchErrors[i] = malloc((NUM_H_LAYERS + 1) * sizeof(Mat *));
        if (batchErrors[i] == NULL) {
            perror("Error allocating memory for **errors\n");
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < NUM_H_LAYERS + 1; j++) {
            batchErrors[i][j] = mat_init(NUM_LAYER_NODES[j], 1);
        }
    }

    for (int i = 0; i < NUM_EPOCHS; i++) {
        printf("Epoch %d / %d\n", i + 1, NUM_EPOCHS);
        for (int j = 0; j < num_batches; j++) {
            int startingIndex = j * BATCH_SIZE;
            update_weights(actFnct, inputs, weights, biases, labels,
                           startingIndex, nets, batchErrors);
            // if ((j + 1) % 500 == 0) {
            //     printf("Batch %d / %d\n", j + 1, num_batches);
            //     test_weights(weights, biases, trainingInputs, trainingLabels,
            //                  guesses);
            // }
        }
        test_weights(weights, biases, trainingInputs, trainingLabels, guesses);
    }

    free(trainingLabels);
    for (int i = 0; i < TEST_DATA_SIZE; i++) {
        mat_free(trainingInputs[i]);
    }
    free(trainingInputs);
    free(guesses);

    // Freeing all of the nets and errors
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < NUM_H_LAYERS + 1; j++) {
            mat_free(batchErrors[i][j]);
        }
        free(batchErrors[i]);
        net_free(nets[i]);
    }
    free(nets);
    free(batchErrors);
}
