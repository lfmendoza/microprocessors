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
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Kernel para calcular la suma (reducción)
__global__ void reduceSum(float* input, float* output, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    float sum = 0;
    if (i < N)
        sum += input[i];
    if (i + blockDim.x < N)
        sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Reducción en memoria compartida
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Escribir resultado del bloque
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Kernel para ordenar usando Bubble Sort
__global__ void bubbleSortKernel(float* data, int N) {
    // Hilo realiza el ordenamiento
    if (threadIdx.x == 0 && blockIdx.x == 0){
        for (int i = 0; i < N - 1; ++i){
            for (int j = 0; j < N - i - 1; ++j){
                if (data[j] > data[j + 1]){
                    float temp = data[j];
                    data[j] = data[j + 1];
                    data[j + 1] = temp;
                }
            }
        }
    }
}

// Kernel para calcular cuantiles (mínimo, Q1, mediana, Q3, máximo)
__global__ void computeQuantiles(float* data, int N, float* quantiles) {
    if (threadIdx.x == 0 && blockIdx.x == 0){
        // Mínimo
        quantiles[0] = data[0];
        // Máximo
        quantiles[4] = data[N - 1];

        // Mediana
        if (N % 2 == 1){
            quantiles[2] = data[N / 2];
        } else {
            quantiles[2] = (data[N / 2 - 1] + data[N / 2]) / 2.0f;
        }

        // Q1 y Q3
        int mid = N / 2;
        if (mid % 2 == 1){
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
        string sheet_id = "1QvEpnYnAtH9mYsoq5yqj4VazvpqxICCanV5ZuOmi0yA"; // Reemplaza con tu ID
        string gid = "0";
        
        // URL de la hoja de Google en formato CSV
        string url = "https://docs.google.com/spreadsheets/d/" + sheet_id + "/export?format=csv&gid=" + gid;
       
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        // Seguir redirecciones si las hay
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        // Especificar la función de escritura
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        // Pasar string para almacenar los datos
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        // Realizar la solicitud curl
        res = curl_easy_perform(curl);
        // Verificar errores
        if(res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() falló: %s\n", curl_easy_strerror(res));
            return 1;
        }
        // Limpiar
        curl_easy_cleanup(curl);
    } else {
        cout << "Error al inicializar libcurl." << endl;
        return 1;
    }

    // Procesar el contenido CSV almacenado en readBuffer
    vector<vector<string>> datos;
    stringstream sstream(readBuffer);
    string linea;

    // Variable para almacenar las presiones
    vector<float> pressures;

    // Saltar el encabezado
    bool firstLine = true;

    // Leer cada línea del CSV
    while (getline(sstream, linea)) {
        // Saltar la primera línea de encabezado
        if (firstLine) {
            firstLine = false;
            continue;
        }

        stringstream ss(linea);
        string valor;
        vector<string> fila;

        // Separar los valores por comas
        while (getline(ss, valor, ',')) {
            fila.push_back(valor);
        }
        datos.push_back(fila);

        // La presión está en la tercera columna (índice 2)
        if (fila.size() >= 3) {
            float pressureValue = stof(fila[2]);
            pressures.push_back(pressureValue);
        }
    }

    // Número de elementos
    int N = pressures.size();

    // Mostrar conteo
    cout << "Conteo: " << N << endl;

    // Copiar datos a memoria de dispositivo
    float* d_pressures;
    cudaCheckError(cudaMalloc((void**)&d_pressures, N * sizeof(float)));
    cudaCheckError(cudaMemcpy(d_pressures, pressures.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Calcular suma usando reducción
    int blockSize = 256;
    int gridSize = (N + blockSize * 2 - 1) / (blockSize * 2);

    float* d_intermediate;
    cudaCheckError(cudaMalloc((void**)&d_intermediate, gridSize * sizeof(float)));

    size_t sharedMemSize = blockSize * sizeof(float);
    reduceSum<<<gridSize, blockSize, sharedMemSize>>>(d_pressures, d_intermediate, N);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Copiar resultados intermedios al host y sumar
    vector<float> h_intermediate(gridSize);
    cudaCheckError(cudaMemcpy(h_intermediate.data(), d_intermediate, gridSize * sizeof(float), cudaMemcpyDeviceToHost));

    float totalSum = 0.0f;
    for (int i = 0; i < gridSize; ++i) {
        totalSum += h_intermediate[i];
    }

    // Calcular media
    float mean = totalSum / N;

    // Ordenar datos en el dispositivo
    bubbleSortKernel<<<1,1>>>(d_pressures, N);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Calcular cuantiles
    float* d_quantiles;
    cudaCheckError(cudaMalloc((void**)&d_quantiles, 5 * sizeof(float)));

    computeQuantiles<<<1,1>>>(d_pressures, N, d_quantiles);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Copiar cuantiles al host
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
