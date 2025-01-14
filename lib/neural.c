#include "neural.h"

// Initializing a network struct to contain the neural network data
Network *net_init(Mat **weights, Mat **biases) {
    Network *net = malloc(sizeof(Network));
    if (net == NULL) {
        perror("Error allocating memory for net\n");
        exit(EXIT_FAILURE);
    }

    net->preLayers = init_layers_empty();
    net->layers = init_layers_empty();
    net->weights = weights;
    net->biases = biases;

    return net;
}

// Freeing all of the memory taken by a network
void net_free(Network *net) {
    for (int i = 0; i < NUM_H_LAYERS + 1; i++) {
        mat_free(net->preLayers[i]);
        mat_free(net->layers[i]);
    }
    free(net->preLayers);
    free(net->layers);
    free(net);
}

// Free a network with empty layers
void net_free_empty(Network *net) {
    free(net->preLayers);
    free(net->layers);
    free(net);
}

// Freeing the values stored in a Network *'s layers
void net_free_layerVals(Network *net) {
    for (int i = 0; i < NUM_H_LAYERS + 1; i++) {
        mat_free(net->preLayers[i]);
        mat_free(net->layers[i]);
    }
}

// Initializing matrices to store all of the layer outputs/ preOutputs
Mat **init_layers() {
    Mat **layers = malloc((NUM_H_LAYERS + 1) * sizeof(Mat *));
    if (layers == NULL) {
        perror("Error allocating memory for layers\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < NUM_H_LAYERS + 1; i++) {
        layers[i] = mat_init(NUM_LAYER_NODES[i], 1);
    }

    return layers;
}

// Initializing pointers to store the layers
Mat **init_layers_empty() {
    Mat **layers = malloc((NUM_H_LAYERS + 1) * sizeof(Mat *));
    if (layers == NULL) {
        perror("Error allocating memory for layers\n");
        exit(EXIT_FAILURE);
    }

    return layers;
}

// Initializing matrices with random numbers to store all of the weights
Mat **init_weights() {
    Mat **weights = malloc((NUM_H_LAYERS + 1) * sizeof(Mat *));
    if (weights == NULL) {
        perror("Error allocating memory for weights\n");
        exit(EXIT_FAILURE);
    }

    // Weights connecting the input to the first hidden layer
    weights[0] = mat_init(NUM_LAYER_NODES[0], MAT_SIZE * MAT_SIZE);

    for (int i = 1; i < NUM_H_LAYERS + 1; i++) {
        weights[i] = mat_init(NUM_LAYER_NODES[i], NUM_LAYER_NODES[i - 1]);
    }

    // Initializing the weights with random values
    for (int i = 0; i < NUM_H_LAYERS + 1; i++) {
        mat_populate_rand(weights[i]);
    }

    return weights;
}

// Initializing weights with 0 (for backpropagation)
Mat **init_weightsZ() {
    Mat **weights = malloc((NUM_H_LAYERS + 1) * sizeof(Mat *));
    if (weights == NULL) {
        perror("Error allocating memory for weights\n");
        exit(EXIT_FAILURE);
    }

    // Weights connecting the input to the first hidden layer
    weights[0] = mat_init(NUM_LAYER_NODES[0], MAT_SIZE * MAT_SIZE);

    for (int i = 1; i < NUM_H_LAYERS + 1; i++) {
        weights[i] = mat_init(NUM_LAYER_NODES[i], NUM_LAYER_NODES[i - 1]);
    }

    // Initializing the weights with random values
    for (int i = 0; i < NUM_H_LAYERS + 1; i++) {
        mat_populate(weights[i], 0.0);
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
        biases[i] = mat_init(NUM_LAYER_NODES[i], 1);
    }
    biases[NUM_H_LAYERS] = mat_init(OUTPUT_SIZE, 1);

    for (int i = 0; i < NUM_H_LAYERS + 1; i++) {
        mat_populate_rand(biases[i]);
    }

    return biases;
}

// Calculating the output of the neural network using an activation function,
// the input, and the weights
Mat *propagate(double (*actFnct)(double), Mat *input, Network *net) {
    net->preLayers[0] =
        mat_add1(mat_multiply(net->weights[0], input), net->biases[0]);
    net->layers[0] = apply2(actFnct, net->preLayers[0]);

    for (int i = 1; i < NUM_H_LAYERS + 1; i++) {
        net->preLayers[i] = mat_add1(
            mat_multiply(net->weights[i], net->layers[i - 1]), net->biases[i]);
        net->layers[i] = apply2(actFnct, net->preLayers[i]);
    }

    Mat *output = net->layers[NUM_H_LAYERS];
    return output;
}

int *init_labels(int dataSize) {
    int *labels = malloc(dataSize * sizeof(int));
    if (labels == NULL) {
        perror("Error allocating memory for labels\n");
        exit(EXIT_FAILURE);
    }
    return labels;
}

// Testing the percent error yielded by the given weights and biases when
// running the test data (a 'closed' function that does not require the inputs
// to be supplied)
void test_weightsClosed(Mat **weights, Mat **biases) {
    FILE *testData = openInputFile("./data/mnist_test.csv");

    // Loading all of the data into an array of matrices and labels
    int *labels = init_labels(TEST_DATA_SIZE);

    Mat **inputs = malloc(TEST_DATA_SIZE * sizeof(Mat *));
    if (inputs == NULL) {
        perror("Error allocating memory for inputs\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < TEST_DATA_SIZE; i++) {
        inputs[i] = dataToMat(testData, &labels[i]);
    }

    // printf("Test input read from file.\n");

    int *guesses = malloc(TEST_DATA_SIZE * sizeof(int));
#pragma omp parallel
    {
        Network *net = net_init(weights, biases);
#pragma omp for
        for (int i = 0; i < TEST_DATA_SIZE; i++) {
            propagate(sigmoid, inputs[i], net);
            Mat *output = net->layers[NUM_H_LAYERS];
            guesses[i] = maxIndex(output);
            net_free_layerVals(net);
        }
        net_free_empty(net);
    }

    // Checking guesses
    int numWrong = 0;
    for (int i = 0; i < TEST_DATA_SIZE; i++) {
        if (labels[i] != guesses[i]) {
            // printf("Error at line %d\n", i + 2);
            // printf("Label: %d\n", labels[i]);
            // printf("Guess: %d\n", guesses[i]);
            // mat_unflatten(&inputs[i], MAT_SIZE);
            // mat_printI(inputs[i]);
            numWrong++;
        }
    }

    double percentWrong = ((double)numWrong / TEST_DATA_SIZE) * 100;
    printf("Percent wrong: %lf %%\n", percentWrong);

    fclose(testData);
    free(labels);
    for (int i = 0; i < TEST_DATA_SIZE; i++) {
        mat_free(inputs[i]);
    }
    free(inputs);
    free(guesses);
}
