CC = gcc
CFLAGS = -Wall -Wextra -fopenmp -g
LDFLAGS = -lm -fopenmp

# Define the executable name and build directory
EXEC = bigbrain
BUILD_DIR = build

# List of source files and corresponding object files in the build directory
SRC = main.c $(wildcard lib/*.c)
OBJ = $(patsubst %.c, $(BUILD_DIR)/%.o, $(SRC))

# Default rule to create the executable
$(EXEC): $(OBJ)
	@$(CC) $(OBJ) -o $(EXEC) $(LDFLAGS)

# Rule to compile each .c file into the corresponding .o file in the build directory
$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@) #Creates the build directory
	@$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -rf $(BUILD_DIR) $(EXEC)
