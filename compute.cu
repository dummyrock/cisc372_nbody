#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

#define TILE_SIZE 16

static double *dPos = NULL;
static double *dVel = NULL;
static double *dMass = NULL;
static double *dAccels = NULL;
static int deviceReady = 0;

static void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static void freeDeviceMemory(void) {
    if (dPos != NULL) cudaFree(dPos);
    if (dVel != NULL) cudaFree(dVel);
    if (dMass != NULL) cudaFree(dMass);
    if (dAccels != NULL) cudaFree(dAccels);

    dPos = NULL;
    dVel = NULL;
    dMass = NULL;
    dAccels = NULL;
    deviceReady = 0;
}

__global__ void computeAccelMatrix(const double *pos, const double *mass, double *accels, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double iPos[TILE_SIZE][3];
    __shared__ double jPos[TILE_SIZE][3];
    __shared__ double jMass[TILE_SIZE];

    if (threadIdx.x == 0) {
        if (i < n) {
            size_t pi = (size_t)i * 3ULL;
            iPos[threadIdx.y][0] = pos[pi + 0];
            iPos[threadIdx.y][1] = pos[pi + 1];
            iPos[threadIdx.y][2] = pos[pi + 2];
        } else {
            iPos[threadIdx.y][0] = 0.0;
            iPos[threadIdx.y][1] = 0.0;
            iPos[threadIdx.y][2] = 0.0;
        }
    }

    if (threadIdx.y == 0) {
        if (j < n) {
            size_t pj = (size_t)j * 3ULL;
            jPos[threadIdx.x][0] = pos[pj + 0];
            jPos[threadIdx.x][1] = pos[pj + 1];
            jPos[threadIdx.x][2] = pos[pj + 2];
            jMass[threadIdx.x] = mass[j];
        } else {
            jPos[threadIdx.x][0] = 0.0;
            jPos[threadIdx.x][1] = 0.0;
            jPos[threadIdx.x][2] = 0.0;
            jMass[threadIdx.x] = 0.0;
        }
    }

    __syncthreads();

    if (i >= n || j >= n) return;

    size_t out = ((size_t)i * (size_t)n + (size_t)j) * 3ULL;
    if (i == j) {
        accels[out + 0] = 0.0;
        accels[out + 1] = 0.0;
        accels[out + 2] = 0.0;
        return;
    }

    double dx = iPos[threadIdx.y][0] - jPos[threadIdx.x][0];
    double dy = iPos[threadIdx.y][1] - jPos[threadIdx.x][1];
    double dz = iPos[threadIdx.y][2] - jPos[threadIdx.x][2];

    double magnitudeSq = dx * dx + dy * dy + dz * dz;
    double magnitude = sqrt(magnitudeSq);
    double accelmag = -1.0 * GRAV_CONSTANT * jMass[threadIdx.x] / magnitudeSq;

    accels[out + 0] = accelmag * dx / magnitude;
    accels[out + 1] = accelmag * dy / magnitude;
    accels[out + 2] = accelmag * dz / magnitude;
}

__global__ void sumAndUpdate(const double *accels, double *pos, double *vel, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;

    size_t row = (size_t)i * (size_t)n * 3ULL;
    for (int j = 0; j < n; j++) {
        size_t idx = row + (size_t)j * 3ULL;
        ax += accels[idx + 0];
        ay += accels[idx + 1];
        az += accels[idx + 2];
    }

    size_t p = (size_t)i * 3ULL;
    vel[p + 0] += ax * INTERVAL;
    vel[p + 1] += ay * INTERVAL;
    vel[p + 2] += az * INTERVAL;

    pos[p + 0] += vel[p + 0] * INTERVAL;
    pos[p + 1] += vel[p + 1] * INTERVAL;
    pos[p + 2] += vel[p + 2] * INTERVAL;
}

static void initDeviceMemory(void) {
    size_t vecBytes = sizeof(double) * (size_t)NUMENTITIES * 3ULL;
    size_t massBytes = sizeof(double) * (size_t)NUMENTITIES;
    size_t accelBytes = sizeof(double) * (size_t)NUMENTITIES * (size_t)NUMENTITIES * 3ULL;

    checkCuda(cudaMalloc((void **)&dPos, vecBytes), "cudaMalloc dPos");
    checkCuda(cudaMalloc((void **)&dVel, vecBytes), "cudaMalloc dVel");
    checkCuda(cudaMalloc((void **)&dMass, massBytes), "cudaMalloc dMass");
    checkCuda(cudaMalloc((void **)&dAccels, accelBytes), "cudaMalloc dAccels");

    checkCuda(cudaMemcpy(dPos, hPos, vecBytes, cudaMemcpyHostToDevice), "copy hPos->dPos");
    checkCuda(cudaMemcpy(dVel, hVel, vecBytes, cudaMemcpyHostToDevice), "copy hVel->dVel");
    checkCuda(cudaMemcpy(dMass, mass, massBytes, cudaMemcpyHostToDevice), "copy mass->dMass");

    atexit(freeDeviceMemory);
    deviceReady = 1;
}

// compute: Updates the positions and locations of the objects in the system based on gravity.
// Parameters: None
// Returns: None
// Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute() {
    if (!deviceReady) {
        initDeviceMemory();
    }

    dim3 block2d(TILE_SIZE, TILE_SIZE);
    dim3 grid2d((NUMENTITIES + TILE_SIZE - 1) / TILE_SIZE,
                (NUMENTITIES + TILE_SIZE - 1) / TILE_SIZE);

    computeAccelMatrix<<<grid2d, block2d>>>(dPos, dMass, dAccels, NUMENTITIES);
    checkCuda(cudaGetLastError(), "launch computeAccelMatrix");
    checkCuda(cudaDeviceSynchronize(), "sync computeAccelMatrix");

    int block1d = 256;
    int grid1d = (NUMENTITIES + block1d - 1) / block1d;
    sumAndUpdate<<<grid1d, block1d>>>(dAccels, dPos, dVel, NUMENTITIES);
    checkCuda(cudaGetLastError(), "launch sumAndUpdate");
    checkCuda(cudaDeviceSynchronize(), "sync sumAndUpdate");

    size_t vecBytes = sizeof(double) * (size_t)NUMENTITIES * 3ULL;
    checkCuda(cudaMemcpy(hPos, dPos, vecBytes, cudaMemcpyDeviceToHost), "copy dPos->hPos");
    checkCuda(cudaMemcpy(hVel, dVel, vecBytes, cudaMemcpyDeviceToHost), "copy dVel->hVel");
}
