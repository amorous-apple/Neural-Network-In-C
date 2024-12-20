#include "utils_files.h"

FILE *openDataFile(char *fileName) {
    FILE *pfile = fopen(fileName, "r");
    if (pfile == NULL) {
        printf("Error opening dataFile %s\n", fileName);
        exit(EXIT_FAILURE);
    }
    return pfile;
}
