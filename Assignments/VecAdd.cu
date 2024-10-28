#include <stdio.h>
#include <cuda_runtime.h>

#define N 10000000

// function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void init_vector(float *vec) {
    for (unsigned int i = 0; i < N; i++) 
        vec[i] = (float) rand() / RAND_MAX;
}

void vecAdd_cpu(float *a, float *b, float *c) {
    for (unsigned int i = 0; i < N; i++) 
        c[i] = a[i] + b[i];
}

__global__ void vecAdd_kernel(float *a, float *b, float *c) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) 
        c[i] = a[i] + b[i];
}

float vecAdd_gpu(float *h_a, float *h_b, float *h_c, size_t size) {
    float *d_a, *d_b, *d_c;

    // allocate device memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);


    // copy data to GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // perform addition
    const unsigned int BLOCK_SIZE = 256;
    // N = 1024, BLOCK_SIZE = 256, num_blocks = 4
    // (N + BLOCK_SIZE - 1) / BLOCK_SIZE = ( (1024 + 256 - 1) / 256 ) = 1279 / 256 = 4 rounded 
    // N = 1025, BLOCK_SIZE = 256, num_blocks = 4
    // (N + BLOCK_SIZE - 1) / BLOCK_SIZE = ( (1025 + 256 - 1) / 256 ) = 1280 / 256 = 4 rounded 
    const unsigned int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double start_time = get_time();
    vecAdd_kernel<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c);
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
    printf("Vector addition\n");
    size_t size = N * sizeof(float);
    srand(time(NULL));

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // Initialize the vectors with random values
    init_vector(h_a);
    init_vector(h_b);

    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 5; i++) {
        double start_time = get_time();
        vecAdd_cpu(h_a, h_b, h_c);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 5.0;
    printf("CPU Time: %f\n", cpu_avg_time);

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 5; i++) {
        gpu_total_time += vecAdd_gpu(h_a, h_b, h_c, size);
    }
    double gpu_avg_time = gpu_total_time / 5.0;
    printf("GPU Time: %f\n", gpu_avg_time);

    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);
    
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}