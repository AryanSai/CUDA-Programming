#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

//MNIST has 70,000 images, we are training on 10,000 images
#define TRAIN_SIZE 10000
//test on 1000 images
#define DEV_SIZE 1000
//test on 1000 images
#define TEST_SIZE 100

#define INPUT_FEATURES 784
#define HIDDEN_FEATURES 256
#define OUTPUT_SIZE 10

#define BATCH_SIZE 32
#define EPOCHS 20
#define LEARNING_RATE 0.001

#define SIZE sizeof(float)

typedef struct {
    float *weights1;
    float *weights2;
    float *bias1;
    float *bias2;
    float *grad_weights1;
    float *grad_weights2;
    float *grad_bias1;
    float *grad_bias2;
} NeuralNetwork;

// CUDA_CHECK macro to print error information
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// load batch image data
void load_images(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, SIZE, size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// load batch labels
void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading labels: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// kaiming init func for weights
void initialize_weights(float *weights, int size) {
    float scale = sqrtf(2.0f / size);
    for (int i = 0; i < size; i++)
        weights[i] = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
}

// basic init for biases
void initialize_bias(float *bias, int size) {
    for (int i = 0; i < size; i++)
        bias[i] = 0.0f;
}

// CUDA kernel for matrix multiplication (A @ B)
__global__ void matmul_a_b_kernel(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
            sum += A[row * n + i] * B[i * k + col];
        C[row * k + col] = sum;
    }
}

// CUDA kernel for matrix multiplication (A @ B.T)
__global__ void matmul_a_bt_kernel(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
            sum += A[row * n + i] * B[col * n + i];
        C[row * k + col] = sum;
    }
}

// CUDA kernel for matrix multiplication (A.T @ B)
__global__ void matmul_at_b_kernel(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < m; ++i)
            sum += A[i * n + row] * B[i * k + col];
        C[row * k + col] = sum;
    }
}

// CUDA kernel for ReLU activation
__global__ void relu_kernel(float *x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        x[idx] = fmaxf(0.0f, x[idx]);
}

// CUDA kernel for bias addition
__global__ void bias_add_kernel(float *x, float *bias, int batch_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / size;
    int i = idx % size;
    if (b < batch_size && i < size)
        x[idx] += bias[i];
}

// CUDA kernel for softmax
// softmax(x_i) = exp(x_i) / sum(exp(x_j) for j in range(N))
// where x is the input vector and N is the number of elements in the vector.
__global__ void softmax_kernel(float *x, int batch_size, int size) {
    int b = blockIdx.x;
    if (b < batch_size) {
        float max_val = x[b * size];
        for (int i = 1; i < size; ++i) 
            max_val = fmaxf(max_val, x[b * size + i]);
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            x[b * size + i] = expf(x[b * size + i] - max_val);
            sum += x[b * size + i];
        }
        for (int i = 0; i < size; ++i) 
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
    }
}

void forward(NeuralNetwork *nn, float *d_input, float *d_hidden, float *d_output, int batch_size) {
    // 1024 threads/blocks
    dim3 block_size(32, 32);

    // just enough blocks + threads for our naive matmul kernel
    dim3 grid_size((HIDDEN_FEATURES + block_size.x - 1) / block_size.x, (batch_size + block_size.y - 1) / block_size.y);

    //input * weights1 = hidden
    matmul_a_b_kernel<<<grid_size, block_size>>>(d_input, nn->weights1, d_hidden, batch_size, INPUT_FEATURES, HIDDEN_FEATURES);
    CUDA_CHECK(cudaGetLastError());

    //add bias1 (one bias term for each neuron)
    //hidden = hidden + bias1
    bias_add_kernel<<<(batch_size * HIDDEN_FEATURES + 255) / 256, 256>>>(d_hidden, nn->bias1, batch_size, HIDDEN_FEATURES);
    CUDA_CHECK(cudaGetLastError());

    //ReLU on hidden
    relu_kernel<<<(batch_size * HIDDEN_FEATURES + 255) / 256, 256>>>(d_hidden, batch_size * HIDDEN_FEATURES);
    CUDA_CHECK(cudaGetLastError());

    //hidden * weights2 = output
    grid_size.x = (OUTPUT_SIZE + block_size.x - 1) / block_size.x;
    grid_size.y = (batch_size + block_size.y - 1) / block_size.y;
    matmul_a_b_kernel<<<grid_size, block_size>>>(d_hidden, nn->weights2, d_output, batch_size, HIDDEN_FEATURES, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    //output = output + bias2
    bias_add_kernel<<<(batch_size * OUTPUT_SIZE + 255) / 256, 256>>>(d_output, nn->bias2, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    //softmax
    softmax_kernel<<<batch_size, 1>>>(d_output, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

// Modify cross_entropy_loss to work with batches (w/out softmax because we already do this in the forward pass)
float cross_entropy_loss(float *output, int *labels, int batch_size) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        total_loss -= logf(fmaxf(output[b * OUTPUT_SIZE + labels[b]], 1e-7f));
    }
    return total_loss / batch_size;
}

// Add this CUDA kernel to zero out gradients
__global__ void zero_grad_kernel(float *grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 0.0f;
    }
}

// CUDA kernel for computing output gradients
__global__ void compute_output_gradients_kernel(float *grad_output, float *output, int *labels, int batch_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];
        }
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;
    }
}

// CUDA kernel for updating gradients
__global__ void update_gradients_kernel(float *grad_weights, float *grad_bias, float *grad_layer, float *prev_layer, int batch_size, int prev_size, int curr_size) {
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < curr_size && j < prev_size) {
        float grad_w_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            grad_w_sum += grad_layer[b * curr_size + i] * prev_layer[b * prev_size + j];
        }
        atomicAdd(&grad_weights[i * prev_size + j], grad_w_sum);

        if (j == 0) {
            float grad_b_sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                grad_b_sum += grad_layer[b * curr_size + i];
            }
            atomicAdd(&grad_bias[i], grad_b_sum);
        }
    }
}

//dderivative of relu 
__global__ void drelu_kernel(float *x, float *d_ReLU_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_ReLU_out[idx] = x[idx] > 0.0f ? 1.0f : 0.0f;
    }
}

//element-wise product of d_dX2 and d_grad_hidden
__global__ void multiply_gradients_kernel(float *grad1, float *grad2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad1[idx] *= grad2[idx];
    }
}

// Modified backward function using CUDA kernels
// shape rotating is on par with the visual example (excalidraw diagram) in the mnist-cuda git repo (also found in "assets")
void backward(NeuralNetwork *nn, float *d_input, float *d_hidden, float *d_output, int *d_labels, int batch_size) {
    // Initialize gradients to zero using CUDA kernel
    zero_grad_kernel<<<(HIDDEN_FEATURES * INPUT_FEATURES + 256 - 1) / 256, 256>>>(nn->grad_weights1, HIDDEN_FEATURES * INPUT_FEATURES);
    CUDA_CHECK(cudaGetLastError());

    zero_grad_kernel<<<(OUTPUT_SIZE * HIDDEN_FEATURES + 256 - 1) / 256, 256>>>(nn->grad_weights2, OUTPUT_SIZE * HIDDEN_FEATURES);
    CUDA_CHECK(cudaGetLastError());

    zero_grad_kernel<<<(HIDDEN_FEATURES + 256 - 1) / 256, 256>>>(nn->grad_bias1, HIDDEN_FEATURES);
    CUDA_CHECK(cudaGetLastError());

    zero_grad_kernel<<<(OUTPUT_SIZE + 256 - 1) / 256, 256>>>(nn->grad_bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());
    
    // Compute gradients for output layer
    float *d_grad_output;
    CUDA_CHECK(cudaMalloc(&d_grad_output, batch_size * OUTPUT_SIZE * SIZE));
    compute_output_gradients_kernel<<<(batch_size + 255) / 256, 256>>>(d_grad_output, d_output, d_labels, batch_size);
    CUDA_CHECK(cudaGetLastError());

    // Update gradients for weights2 (W2.grad = grad_output.T @ hidden)
    dim3 block_size(32, 32);
    dim3 grid_size((HIDDEN_FEATURES + block_size.x - 1) / block_size.x, (OUTPUT_SIZE + block_size.y - 1) / block_size.y);
    matmul_at_b_kernel<<<grid_size, block_size>>>(d_hidden, d_grad_output, nn->grad_weights2, batch_size, HIDDEN_FEATURES, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Update gradients for bias2
    update_gradients_kernel<<<grid_size, block_size>>>(nn->grad_weights2, nn->grad_bias2, d_grad_output, d_hidden, batch_size, HIDDEN_FEATURES, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Compute dX2 (gradient of loss w.r.t. input of second layer)
    float *d_dX2;
    CUDA_CHECK(cudaMalloc(&d_dX2, batch_size * HIDDEN_FEATURES * SIZE));
    grid_size.x = (HIDDEN_FEATURES + block_size.x - 1) / block_size.x;
    grid_size.y = (batch_size + block_size.y - 1) / block_size.y;
    matmul_a_bt_kernel<<<grid_size, block_size>>>(d_grad_output, nn->weights2, d_dX2, batch_size, OUTPUT_SIZE, HIDDEN_FEATURES);
    CUDA_CHECK(cudaGetLastError());

    // Compute d_ReLU_out (element-wise multiplication with ReLU derivative)
    float *d_grad_hidden;
    CUDA_CHECK(cudaMalloc(&d_grad_hidden, batch_size * HIDDEN_FEATURES * SIZE));
    drelu_kernel<<<(batch_size * HIDDEN_FEATURES + 255) / 256, 256>>>(d_hidden, d_grad_hidden, batch_size * HIDDEN_FEATURES);
    CUDA_CHECK(cudaGetLastError());


    multiply_gradients_kernel<<<(batch_size * HIDDEN_FEATURES + 255) / 256, 256>>>(d_dX2, d_grad_hidden, batch_size * HIDDEN_FEATURES);
    CUDA_CHECK(cudaGetLastError());

    // Update gradients for weights1 (W1.grad = d_ReLU_out.T @ input)
    grid_size.x = (INPUT_FEATURES + block_size.x - 1) / block_size.x;
    grid_size.y = (HIDDEN_FEATURES + block_size.y - 1) / block_size.y;
    matmul_at_b_kernel<<<grid_size, block_size>>>(d_input, d_dX2, nn->grad_weights1, batch_size, INPUT_FEATURES, HIDDEN_FEATURES);
    CUDA_CHECK(cudaGetLastError());

    // Update gradients for bias1
    update_gradients_kernel<<<grid_size, block_size>>>(nn->grad_weights1, nn->grad_bias1, d_dX2, d_input, batch_size, INPUT_FEATURES, HIDDEN_FEATURES);
    CUDA_CHECK(cudaGetLastError());

    // Free allocated memory
    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_dX2));
    CUDA_CHECK(cudaFree(d_grad_hidden));

    CUDA_CHECK(cudaDeviceSynchronize());
}

// Stochastic Gradient Descent (SGD)
// weights -= (learning rate) * (gradient of weights wrt loss)
__global__ void update_weights_kernel(float *weights, float *grad_weights, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        weights[idx] -= LEARNING_RATE * grad_weights[idx];
}

void update_weights(NeuralNetwork *nn) {
    int block_size = 256;
    int grid_size;

    // Update weights1
    grid_size = (HIDDEN_FEATURES * INPUT_FEATURES + block_size - 1) / block_size;
    update_weights_kernel<<<grid_size, block_size>>>(nn->weights1, nn->grad_weights1, HIDDEN_FEATURES * INPUT_FEATURES);
    CUDA_CHECK(cudaGetLastError());

    // Update weights2
    grid_size = (OUTPUT_SIZE * HIDDEN_FEATURES + block_size - 1) / block_size;
    update_weights_kernel<<<grid_size, block_size>>>(nn->weights2, nn->grad_weights2, OUTPUT_SIZE * HIDDEN_FEATURES);
    CUDA_CHECK(cudaGetLastError());

    // Update bias1
    grid_size = (HIDDEN_FEATURES + block_size - 1) / block_size;
    update_weights_kernel<<<grid_size, block_size>>>(nn->bias1, nn->grad_bias1, HIDDEN_FEATURES);
    CUDA_CHECK(cudaGetLastError());

    // Update bias2
    grid_size = (OUTPUT_SIZE + block_size - 1) / block_size;
    update_weights_kernel<<<grid_size, block_size>>>(nn->bias2, nn->grad_bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

// Modified train function to work with CUDA
void train(NeuralNetwork *nn, float *X_train, int *y_train) {
    float *d_X_train, *d_hidden, *d_output;
    int *d_y_train;

    //allocate memory on device
    CUDA_CHECK(cudaMalloc(&d_X_train, TRAIN_SIZE * INPUT_FEATURES * SIZE));
    CUDA_CHECK(cudaMalloc(&d_hidden, BATCH_SIZE * HIDDEN_FEATURES * SIZE));
    CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * SIZE));
    CUDA_CHECK(cudaMalloc(&d_y_train, TRAIN_SIZE * sizeof(int)));

    //copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_X_train, X_train, TRAIN_SIZE * INPUT_FEATURES * SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_train, y_train, TRAIN_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    int NUM_BATCHES = TRAIN_SIZE / BATCH_SIZE;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;
        
        // Zero out gradients at the beginning of each epoch
        zero_grad_kernel<<<(HIDDEN_FEATURES * INPUT_FEATURES + 256 - 1) / 256, 256>>>(nn->grad_weights1, HIDDEN_FEATURES * INPUT_FEATURES);
        zero_grad_kernel<<<(OUTPUT_SIZE * HIDDEN_FEATURES + 256 - 1) / 256, 256>>>(nn->grad_weights2, OUTPUT_SIZE * HIDDEN_FEATURES);
        zero_grad_kernel<<<(HIDDEN_FEATURES + 256 - 1) / 256, 256>>>(nn->grad_bias1, HIDDEN_FEATURES);
        zero_grad_kernel<<<(OUTPUT_SIZE + 256 - 1) / 256, 256>>>(nn->grad_bias2, OUTPUT_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int batch = 0; batch < NUM_BATCHES; batch++) {
            int start_idx = batch * BATCH_SIZE;
            
            forward(nn, &d_X_train[start_idx * INPUT_FEATURES], d_hidden, d_output, BATCH_SIZE);

            float *h_output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * SIZE);
            CUDA_CHECK(cudaMemcpy(h_output, d_output, BATCH_SIZE * OUTPUT_SIZE * SIZE, cudaMemcpyDeviceToHost));

            float loss = cross_entropy_loss(h_output, &y_train[start_idx], BATCH_SIZE);
            total_loss += loss;

            for (int i = 0; i < BATCH_SIZE; i++) {
                int predicted = 0;
                for (int j = 1; j < OUTPUT_SIZE; j++) {
                    if (h_output[i * OUTPUT_SIZE + j] > h_output[i * OUTPUT_SIZE + predicted])
                        predicted = j;
                }
                if (predicted == y_train[start_idx + i])
                    correct++;
            }

            free(h_output);

            backward(nn, &d_X_train[start_idx * INPUT_FEATURES], d_hidden, d_output, &d_y_train[start_idx], BATCH_SIZE);

            update_weights(nn);

            if ((batch + 1) % 100 == 0 || (epoch == 0 && batch == 0)) {
                printf("Epoch %d/%d, Iter %d/%d, Loss: %.4f, Accuracy: %.2f%%\n", 
                       epoch + 1, EPOCHS, batch + 1, NUM_BATCHES, total_loss / (batch + 1), 
                       100.0f * correct / ((batch + 1) * BATCH_SIZE));
            }
        }
        
        printf("Epoch %d/%d completed, Loss: %.4f, Accuracy: %.2f%%\n", 
            epoch + 1, EPOCHS, total_loss / NUM_BATCHES, 100.0f * correct / TRAIN_SIZE);
    }
    
    CUDA_CHECK(cudaFree(d_X_train));
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_y_train));
}

void test(NeuralNetwork *nn, float *X_test, int *y_test, int num_samples) {
    float *d_X_test, *d_hidden, *d_output;
    int *d_y_test;
    int correct=0;

    CUDA_CHECK(cudaMalloc(&d_X_test, TEST_SIZE * INPUT_FEATURES * SIZE));
    CUDA_CHECK(cudaMalloc(&d_hidden, BATCH_SIZE * HIDDEN_FEATURES * SIZE));
    CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * SIZE));
    CUDA_CHECK(cudaMalloc(&d_y_test, TEST_SIZE * sizeof(int)));

    // copy test data to device
    CUDA_CHECK(cudaMemcpy(d_X_test, X_test, TEST_SIZE * INPUT_FEATURES * SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_test, y_test, TEST_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    for (int i = 0; i < num_samples; i++) {
        // pick a random index to test
        int random_index = rand() % TEST_SIZE;

        // forward pass for a single sample
        forward(nn, &d_X_test[random_index * INPUT_FEATURES], d_hidden, d_output, 1);

        // copy output to host
        float *h_output = (float *)malloc(OUTPUT_SIZE * SIZE);
        CUDA_CHECK(cudaMemcpy(h_output, d_output, OUTPUT_SIZE * SIZE, cudaMemcpyDeviceToHost));

        //find the predicted label
        int predicted_label = 0;
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (h_output[j] > h_output[predicted_label])
                predicted_label = j;
        }

        // print the results
        printf("Test Sample %d: Prediction: %d, Ground Truth: %d\n", i + 1, predicted_label, y_test[random_index]);
        
        if(predicted_label==y_test[random_index])
            correct++;

        // free allocated memory for output
        free(h_output);
    }

    printf("\nTest Accuracy: %.2f%%\n", 100.0f * correct / num_samples);

    // free device memory
    CUDA_CHECK(cudaFree(d_X_test));
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_y_test));
}

// initialize function to allocate memory
void initialize_nn(NeuralNetwork *nn) {
    CUDA_CHECK(cudaMalloc(&nn->weights1, HIDDEN_FEATURES * INPUT_FEATURES * SIZE));
    CUDA_CHECK(cudaMalloc(&nn->weights2, OUTPUT_SIZE * HIDDEN_FEATURES * SIZE));
    CUDA_CHECK(cudaMalloc(&nn->bias1, HIDDEN_FEATURES * SIZE));
    CUDA_CHECK(cudaMalloc(&nn->bias2, OUTPUT_SIZE * SIZE));
    CUDA_CHECK(cudaMalloc(&nn->grad_weights1, HIDDEN_FEATURES * INPUT_FEATURES * SIZE));
    CUDA_CHECK(cudaMalloc(&nn->grad_weights2, OUTPUT_SIZE * HIDDEN_FEATURES * SIZE));
    CUDA_CHECK(cudaMalloc(&nn->grad_bias1, HIDDEN_FEATURES * SIZE));
    CUDA_CHECK(cudaMalloc(&nn->grad_bias2, OUTPUT_SIZE * SIZE));

    //allocate temporary host memory
    float *h_weights1 = (float *)malloc(HIDDEN_FEATURES * INPUT_FEATURES * SIZE);
    float *h_weights2 = (float *)malloc(OUTPUT_SIZE * HIDDEN_FEATURES * SIZE);
    float *h_bias1 = (float *)malloc(HIDDEN_FEATURES * SIZE);
    float *h_bias2 = (float *)malloc(OUTPUT_SIZE * SIZE);

    //initialize weights and biases on the host
    initialize_weights(h_weights1, HIDDEN_FEATURES * INPUT_FEATURES);
    initialize_weights(h_weights2, OUTPUT_SIZE * HIDDEN_FEATURES);
    initialize_bias(h_bias1, HIDDEN_FEATURES);
    initialize_bias(h_bias2, OUTPUT_SIZE);

    //copy initialized values to device
    CUDA_CHECK(cudaMemcpy(nn->weights1, h_weights1, HIDDEN_FEATURES * INPUT_FEATURES * SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->weights2, h_weights2, OUTPUT_SIZE * HIDDEN_FEATURES * SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias1, h_bias1, HIDDEN_FEATURES * SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias2, h_bias2, OUTPUT_SIZE * SIZE, cudaMemcpyHostToDevice));

    // Free temporary host memory
    free(h_weights1);
    free(h_weights2);
    free(h_bias1);
    free(h_bias2);
}

int main() {
    srand(time(NULL));

    NeuralNetwork nn;
    initialize_nn(&nn); //initialise the weights and biases matrices

    //load the train, dev and test data
    float *X_train = (float *)malloc(TRAIN_SIZE * INPUT_FEATURES * SIZE);
    int *y_train = (int *)malloc(TRAIN_SIZE * sizeof(int));
    float *X_dev = (float *)malloc(DEV_SIZE * INPUT_FEATURES * SIZE);
    int *y_dev = (int *)malloc(DEV_SIZE * sizeof(int));
    float *X_test = (float *)malloc(TEST_SIZE * INPUT_FEATURES * SIZE);
    int *y_test = (int *)malloc(TEST_SIZE * sizeof(int));

    load_images("mnist_data/X_train.bin", X_train, TRAIN_SIZE * INPUT_FEATURES);
    load_labels("mnist_data/y_train.bin", y_train, TRAIN_SIZE);
    load_images("mnist_data/X_test.bin", X_dev, DEV_SIZE * INPUT_FEATURES);
    load_labels("mnist_data/y_test.bin", y_dev, DEV_SIZE);
    load_images("mnist_data/X_test.bin", X_test, TEST_SIZE * INPUT_FEATURES);
    load_labels("mnist_data/y_test.bin", y_test, TEST_SIZE);
    
    printf("\n-------------An MNIST Multi-Layer Perceptron in CUDA!-------------\n");

    printf("\n-------------Time to train the Model!-------------\n");
    train(&nn, X_train, y_train);

    printf("\n-------------The MNIST Multi-Layer Perceptron is ready!-------------\n");

    printf("\n-------------Let us test the Model on a few samples and see!-------------\n");
    test(&nn, X_test, y_test, 5); //test the model

    CUDA_CHECK(cudaFree(nn.weights1));
    CUDA_CHECK(cudaFree(nn.weights2));
    CUDA_CHECK(cudaFree(nn.bias1));
    CUDA_CHECK(cudaFree(nn.bias2));
    CUDA_CHECK(cudaFree(nn.grad_weights1));
    CUDA_CHECK(cudaFree(nn.grad_weights2));
    CUDA_CHECK(cudaFree(nn.grad_bias1));
    CUDA_CHECK(cudaFree(nn.grad_bias2));
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
