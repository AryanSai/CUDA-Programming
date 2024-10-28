#include <stdio.h>
#include <cuda_runtime.h>

// Example 3x2 @ 2x4 = 3x4 -> (M x K) @ (K x N) = (M x N)
// A = [[1, 2], 
//      [3, 4], 
//      [5, 6]]

// B = [[7, 8, 9, 10],
//      [11, 12, 13, 14]]

// C = A * B = [[1*7 + 2*11, 1*8 + 2*12, 1*9 + 2*13, 1*10 + 2*14],
//              [3*7 + 4*11, 3*8 + 4*12, 3*9 + 4*13, 3*10 + 4*14],
//              [5*7 + 6*11, 5*8 + 6*12, 5*9 + 6*13, 5*10 + 6*14]]

// C = [[29, 32, 35, 38],
//      [65, 72, 79, 86],
//      [101, 112, 123, 134]]

#define N 1000

// function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

void matMul_cpu(float *a, float *b, float *c, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int l = 0; l < cols; l++) {
                sum += a[i * cols + l] * b[l * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

__global__ void matMul_kernel(float *a, float *b, float *c, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (row < rows && col < cols) {
        for (int i = 0; i < N; i++) {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

float matMul_gpu(float *h_a, float *h_b, float *h_c, int rows, int cols) {
    float *d_a, *d_b, *d_c;
    size_t size = rows * cols * sizeof(float);

    // allocate device memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // copy data to GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // set up execution configuration
    const dim3 blockSize(16, 16);
    const dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    double start_time = get_time();
    matMul_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, rows, cols);
    cudaDeviceSynchronize();  // wait for GPU to finish
    double end_time = get_time();

    // copy from GPU to CPU
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return end_time - start_time;
}

int main() {
    printf("Matrix Multiplication\n");
    int rows = N, cols = N;  
    size_t size = rows * cols * sizeof(float);
    srand(time(NULL));

    // allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // initialize the matrices with random values
    init_matrix(h_a, rows, cols);
    init_matrix(h_b, rows, cols);

    double start_time = get_time();
    matMul_cpu(h_a, h_b, h_c, rows, cols);
    double end_time = get_time();
    double cpu_time = end_time - start_time;
    printf("CPU Time: %f\n", cpu_time);

    double gpu_time = matMul_gpu(h_a, h_b, h_c, rows, cols);
    printf("GPU Time: %f\n", gpu_time);

    printf("Speedup: %fx\n", cpu_time / gpu_time);
    
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
