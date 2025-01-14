#include "init.h"

const int MAT_SIZE = 28;
const int OUTPUT_SIZE = 10;
const int MAX_LINE_LEN = 5000;
const int TEST_DATA_SIZE = 10000;
const int TRAINING_DATA_SIZE = 60000;

int NUM_H_LAYERS;
int *NUM_LAYER_NODES;

double LEARNING_RATE = 0.3;
int NUM_EPOCHS = 500;
int BATCH_SIZE = 16;

void init(int argc, char **argv) {
    if (argc == 1) {
        NUM_H_LAYERS = 1;
        int num_layer_nodes[] = {300, OUTPUT_SIZE};
        NUM_LAYER_NODES = malloc((NUM_H_LAYERS + 1) * sizeof(int));
        for (int i = 0; i < NUM_H_LAYERS + 1; i++) {
            NUM_LAYER_NODES[i] = num_layer_nodes[i];
        }
        return;
    } else if (argc == 2 || argc != atoi(argv[1]) + 2) {
        printf("Invalid numbers of command lines arguments!\n");
        printf(
            "Usage: ./bigbrain <num of h-layers> <num of h-layer "
            "1 nodes> <num of h-layer 2 nodes> ... <num of h-layer n "
            "nodes>\n");
        exit(EXIT_FAILURE);
    } else {
        NUM_H_LAYERS = atoi(argv[1]);
        NUM_LAYER_NODES = malloc(NUM_H_LAYERS * sizeof(int));

        for (int i = 0; i < NUM_H_LAYERS; i++) {
            NUM_LAYER_NODES[i] = atoi(argv[i + 2]);
        }
        NUM_LAYER_NODES[NUM_H_LAYERS] = OUTPUT_SIZE;
        return;
    }
}

Mat **init_trainingData(int *labels) {
    FILE *trainingDataFile = openInputFile("./data/mnist_train.csv");
    Mat **trainingData = malloc(TRAINING_DATA_SIZE * sizeof(Mat *));
    if (trainingData == NULL) {
        perror("Error allocating memory for training data\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < TRAINING_DATA_SIZE; i++) {
        trainingData[i] = dataToMat(trainingDataFile, &labels[i]);
    }
    fclose(trainingDataFile);
    return trainingData;
}
