#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <curl/curl.h>


// Instrucciones de compilación:
// 1. Installar culr: !apt-get install libcurl4-openssl-dev
// 2. Compilar código: !nvcc -o ejercicio1 ejercicio1.cu -lcurl
// 3. Ejecutar código: !./ejercicio1

using namespace std;


// Función de devolución de llamada para escribir los datos recibidos
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
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
        
        // URL de la hoja de Google en formato CSV
        string url = "https://docs.google.com/spreadsheets/d/" + sheet_id + "/export?format=csv&gid=" + gid;
       
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        // Seguir redirecciones si las hay
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        // Especificar la función de escritura
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        // Pasar nuestro string para almacenar los datos
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);


        // Realizar la solicitud
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


    // Leer cada línea del CSV
    while (getline(sstream, linea)) {
        stringstream ss(linea);
        string valor;
        vector<string> fila;


        // Separar los valores por comas
        while (getline(ss, valor, ',')) {
            fila.push_back(valor);
        }
        datos.push_back(fila);
    }


    // Imprimir los datos
    for (size_t i = 0; i < datos.size(); ++i) {
        for (size_t j = 0; j < datos[i].size(); ++j) {
            cout << datos[i][j] << "\t";
        }
        cout << endl;
    }


    return 0;
}
