#include "init.h"

int MAT_SIZE = 28;
int OUTPUT_SIZE = 10;
int MAX_LINE_LEN = 5000;
int NUM_H_LAYERS;
int *NUM_H_LAYER_NODES;

void init(int argc, char **argv) {
    if (argc == 1) {
        NUM_H_LAYERS = 1;
        int pre_h_layer_nodes[] = {300};
        NUM_H_LAYER_NODES = malloc(NUM_H_LAYERS * sizeof(int));
        for (int i = 0; i < NUM_H_LAYERS; i++) {
            NUM_H_LAYER_NODES[i] = pre_h_layer_nodes[i];
        }
        return;
    } else if (argc == 2) {
        printf("Invalid numbers of command lines arguments!\n");
        printf(
            "Usage: ./bigbrain <number of hidden layers> <num of hidden layer "
            "1 nodes> <num of hidden layer 2 nodes> ... <num of hidden layer n "
            "nodes>\n");
        exit(EXIT_FAILURE);
    } else {
        NUM_H_LAYERS = atoi(argv[1]);
        NUM_H_LAYER_NODES = malloc(NUM_H_LAYERS * sizeof(int));

        for (int i = 0; i < NUM_H_LAYERS; i++) {
            NUM_H_LAYER_NODES[i] = atoi(argv[i + 2]);
        }
        return;
    }
}
