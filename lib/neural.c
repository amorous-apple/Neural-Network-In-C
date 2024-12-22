#include "neural.h"

// Initializing matrices to store all of the hidden layer values
Mat **init_h_layers() {
    Mat **h_layers = malloc(NUM_H_LAYERS * sizeof(Mat *));
    if (h_layers == NULL) {
        perror("Error allocating memory for hidden layers\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < NUM_H_LAYERS; i++) {
        h_layers[i] = mat_init(NUM_H_LAYER_NODES[i], 1);
    }

    return h_layers;
}

// Initializing matrices to store all of the weights
Mat **init_weights() {
    Mat **weights = malloc((NUM_H_LAYERS + 1) * sizeof(Mat *));
    if (weights == NULL) {
        perror("Error allocating memory for weights\n");
        exit(EXIT_FAILURE);
    }

    // Weights connecting the input to the first hidden layer
    weights[0] = mat_init(NUM_H_LAYER_NODES[0], MAT_SIZE * MAT_SIZE);

    for (int i = 1; i < NUM_H_LAYERS; i++) {
        weights[i] = mat_init(NUM_H_LAYER_NODES[i], NUM_H_LAYER_NODES[i - 1]);
    }

    // Weights connecting the last hidden layer to the output
    weights[NUM_H_LAYERS] =
        mat_init(OUTPUT_SIZE, NUM_H_LAYER_NODES[NUM_H_LAYERS - 1]);

    // Initializing the weights with random values
    for (int i = 0; i < NUM_H_LAYERS + 1; i++) {
        mat_populate_rand(weights[i]);
    }

    return weights;
}

Mat **init_biases() {
    Mat **biases = malloc((NUM_H_LAYERS + 1) * sizeof(Mat *));
    if (biases == NULL) {
        perror("Error allocating memory for biases\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < NUM_H_LAYERS; i++) {
        biases[i] = mat_init(NUM_H_LAYER_NODES[i], 1);
    }
    biases[NUM_H_LAYERS] = mat_init(OUTPUT_SIZE, 1);

    for (int i = 0; i < NUM_H_LAYERS + 1; i++) {
        mat_populate(biases[i], 0);
        // mat_populate_rand(biases[i]);
    }

    return biases;
}

// Calculating the output of the neural network
Mat *propagate(Mat *input, Mat **hidden_layers, Mat **weights, Mat **biases) {
    hidden_layers[0] =
        relu_mat(mat_add(mat_multiply(weights[0], input), biases[0]));

    for (int i = 1; i < NUM_H_LAYERS; i++) {
        hidden_layers[i] = relu_mat(
            mat_add(mat_multiply(weights[i], hidden_layers[i - 1]), biases[i]));
    }

    Mat *output = mat_init(OUTPUT_SIZE, 1);
    output = relu_mat(mat_add(
        mat_multiply(weights[NUM_H_LAYERS], hidden_layers[NUM_H_LAYERS - 1]),
        biases[NUM_H_LAYERS]));
    return output;
}
