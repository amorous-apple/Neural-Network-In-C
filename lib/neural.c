#include "neural.h"

// Initializing a network struct to contain the neural network data
Network *net_init(Mat **weights, Mat **biases) {
    Network *net = malloc(sizeof(Network));
    if (net == NULL) {
        perror("Error allocating memory for net\n");
        exit(EXIT_FAILURE);
    }

    Mat **hiddenLayers = init_h_layers();

    net->hiddenLayers = hiddenLayers;
    net->weights = weights;
    net->biases = biases;

    return net;
}

// Freeing all of the memory taken by a network
void net_free(Network *net) {
    for (int i = 0; i < NUM_H_LAYERS; i++) {
        mat_free(net->hiddenLayers[i]);
    }
    // for (int i = 0; i < NUM_H_LAYERS + 1; i++) {
    //     mat_free(net->weights[i]);
    //     mat_free(net->biases[i]);
    // }
    free(net);
}

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

// Calculating the output of the neural network using an activation function,
// the input, and the weights
Mat *propagate(double (*actFnct)(double), Mat *input, Network *net) {
    net->hiddenLayers[0] =
        apply(actFnct,
              (mat_add(mat_multiply(net->weights[0], input), net->biases[0])));

    for (int i = 1; i < NUM_H_LAYERS; i++) {
        net->hiddenLayers[i] = apply(
            actFnct,
            mat_add(mat_multiply(net->weights[i], net->hiddenLayers[i - 1]),
                    net->biases[i]));
    }

    Mat *output = mat_init(OUTPUT_SIZE, 1);
    output = apply(actFnct,
                   (mat_add(mat_multiply(net->weights[NUM_H_LAYERS],
                                         net->hiddenLayers[NUM_H_LAYERS - 1]),
                            net->biases[NUM_H_LAYERS])));
    return output;
}
