#include "init.h"

const int MAT_SIZE = 28;
const int OUTPUT_SIZE = 10;
const int MAX_LINE_LEN = 5000;

double LEARNING_RATE = 0.1;
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
    } else if (argc == 2 || argc != atoi(argv[1]) + 2) {
        printf("Invalid numbers of command lines arguments!\n");
        printf(
            "Usage: ./bigbrain <num of h-layers> <num of h-layer "
            "1 nodes> <num of h-layer 2 nodes> ... <num of h-layer n "
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
