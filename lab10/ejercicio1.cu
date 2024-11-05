#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <curl/curl.h>
#include <cuda_runtime.h>

// Instrucciones de compilación:
// 1. Instalar libcurl: !apt-get install libcurl4-openssl-dev
// 2. Compilar código: !nvcc -o ejercicio1 ejercicio1.cu -lcurl
// 3. Ejecutar código: !./ejercicio1

using namespace std;

// Función para escribir los datos recibidos
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Macro para verificar errores de CUDA
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Kernel para calcular la suma
__global__ void reduceSumOptimized(float* input, float* output, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Memoria compartida
    sdata[tid] = (i < N ? input[i] : 0) + (i + blockDim.x < N ? input[i + blockDim.x] : 0);
    __syncthreads();

    // Reducción en memoria compartida
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result of this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Kernel para ordenar usando ordanamiento en paralelo con Bitonic Sort
__global__ void bitonicSort(float* data, int N) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;

    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            unsigned int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k) == 0 && data[tid] > data[ixj]) {
                    float temp = data[tid];
                    data[tid] = data[ixj];
                    data[ixj] = temp;
                }
                if ((tid & k) != 0 && data[tid] < data[ixj]) {
                    float temp = data[tid];
                    data[tid] = data[ixj];
                    data[ixj] = temp;
                }
            }
            __syncthreads();
        }
    }
}

// Kernel para calcular cuantiles (mínimo, Q1, mediana, Q3, máximo)
__global__ void computeQuantiles(float* data, int N, float* quantiles) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Mínimo
        quantiles[0] = data[0];

        // Máximo
        quantiles[4] = data[N - 1];

        // Mediana
        if (N % 2 == 1) {
            quantiles[2] = data[N / 2];
        } else {
            quantiles[2] = (data[N / 2 - 1] + data[N / 2]) / 2.0f;
        }

        // Q1 y Q3
        int mid = N / 2;
        if (mid % 2 == 1) {
            quantiles[1] = data[mid / 2];
            quantiles[3] = data[mid + mid / 2];
        } else {
            quantiles[1] = (data[mid / 2 - 1] + data[mid / 2]) / 2.0f;
            quantiles[3] = (data[mid + mid / 2 - 1] + data[mid + mid / 2]) / 2.0f;
        }
    }
}

int main() {
    // Inicializar libcurl
    CURL* curl;
    CURLcode res;
    string readBuffer;

    curl = curl_easy_init();
    if(curl) {
        // ID de la hoja de Google y GID
        string sheet_id = "1QvEpnYnAtH9mYsoq5yqj4VazvpqxICCanV5ZuOmi0yA";
        string gid = "0";
        string url = "https://docs.google.com/spreadsheets/d/" + sheet_id + "/export?format=csv&gid=" + gid;

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        res = curl_easy_perform(curl);
        if(res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() falló: %s\n", curl_easy_strerror(res));
            return 1;
        }
        curl_easy_cleanup(curl);
    } else {
        cout << "Error al inicializar libcurl." << endl;
        return 1;
    }

    // Procesar el contenido CSV almacenado en readBuffer
    vector<float> pressures;
    stringstream sstream(readBuffer);
    string line;
    bool firstLine = true;

    while (getline(sstream, line)) {
        if (firstLine) { firstLine = false; continue; }  // Skip header
        stringstream ss(line);
        vector<string> row;
        string value;
        while (getline(ss, value, ',')) { row.push_back(value); }

        if (row.size() >= 3) {
            // La presión está en la tercera columna (índice 2)
            pressures.push_back(stof(row[2]));
        }
    }

    int N = pressures.size();
    if (N == 0) {
        cout << "No se encontraron datos" << endl;
        return 1;
    }
    cout << "Count: " << N << endl;

    // Copiando la data a la GPU
    float *d_pressures, *d_intermediate, *d_quantiles;
    cudaCheckError(cudaMalloc((void**)&d_pressures, N * sizeof(float)));
    cudaCheckError(cudaMemcpy(d_pressures, pressures.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMalloc((void**)&d_intermediate, N / 2 * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_quantiles, 5 * sizeof(float)));

    // Reducción de la suma total
    int blockSize = 256;
    int gridSize = (N + blockSize * 2 - 1) / (blockSize * 2);
    reduceSumOptimized<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_pressures, d_intermediate, N);
    cudaCheckError(cudaDeviceSynchronize());

    // Copiar resultado de la reducción a la CPU
    vector<float> h_intermediate(gridSize);
    cudaCheckError(cudaMemcpy(h_intermediate.data(), d_intermediate, gridSize * sizeof(float), cudaMemcpyDeviceToHost));
    float totalSum = 0;
    for (float val : h_intermediate) {
        totalSum += val;
    }
    float mean = totalSum / N;

    // Ordenar los datos usando Bitonic Sort
    int numThreads = 256;
    int numBlocks = (N + numThreads - 1) / numThreads;
    bitonicSort<<<numBlocks, numThreads>>>(d_pressures, N);
    cudaCheckError(cudaDeviceSynchronize());

    // Calcular cuantiles
    computeQuantiles<<<1, 1>>>(d_pressures, N, d_quantiles);
    cudaCheckError(cudaDeviceSynchronize());

    // Copiar cuantiles a la CPU
    float h_quantiles[5];
    cudaCheckError(cudaMemcpy(h_quantiles, d_quantiles, 5 * sizeof(float), cudaMemcpyDeviceToHost));

    // Imprimir resultados
    cout << "Suma total: " << totalSum << endl;
    cout << "Media: " << mean << endl;
    cout << "Mínimo: " << h_quantiles[0] << endl;
    cout << "Q1: " << h_quantiles[1] << endl;
    cout << "Mediana: " << h_quantiles[2] << endl;
    cout << "Q3: " << h_quantiles[3] << endl;
    cout << "Máximo: " << h_quantiles[4] << endl;

    // Liberar memoria
    cudaFree(d_pressures);
    cudaFree(d_intermediate);
    cudaFree(d_quantiles);

    return 0;
}
