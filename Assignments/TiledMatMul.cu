#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1000
#define TILE_SIZE 32

// Function to measure execution time
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
        for (int j = 0; j < cols; j++) {
            float sum = 0.0f;
            for (int l = 0; l < N; l++) {
                sum += a[i * N + l] * b[l * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

__global__ void matMul_kernel(float *a, float *b, float *c, int rows, int cols) {
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // t < number of blocks
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (t * TILE_SIZE + threadIdx.x < N && row < rows) {
            tile_a[threadIdx.y][threadIdx.x] = a[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (t * TILE_SIZE + threadIdx.y < N && col < cols) {
            tile_b[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < rows && col < cols) {
        c[row * N + col] = sum;
    }
}

float matMul_gpu(float *h_a, float *h_b, float *h_c, int rows, int cols) {
    float *d_a, *d_b, *d_c;
    size_t size_a = rows * N * sizeof(float);
    size_t size_b = N * cols * sizeof(float);
    size_t size_c = rows * cols * sizeof(float);

    // Allocate device memory
    cudaMalloc((void **)&d_a, size_a);
    cudaMalloc((void **)&d_b, size_b);
    cudaMalloc((void **)&d_c, size_c);

    // Copy data to GPU
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    // Set up execution configuration
    const dim3 blockSize(TILE_SIZE, TILE_SIZE);
    const dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    double start_time = get_time();
    matMul_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, rows, cols);
    cudaDeviceSynchronize();  // Wait for GPU to finish
    double end_time = get_time();

    // Copy result from GPU to CPU
    cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return end_time - start_time;
}

int main() {
    printf("Tiled Matrix Multiplication\n");
    int rows = N, cols = N;  
    size_t size = rows * cols * sizeof(float);
    srand(time(NULL));

    // Allocate host memory
    float *h_a = (float *)malloc(rows * N * sizeof(float));
    float *h_b = (float *)malloc(N * cols * sizeof(float));
    float *h_c = (float *)malloc(rows * cols * sizeof(float));

    // Initialize the matrices with random values
    init_matrix(h_a, rows, N);
    init_matrix(h_b, N, cols);

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
