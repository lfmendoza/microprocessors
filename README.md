### Optimizaciones utilizando CUDA

El código ha sido optimizado para aprovechar al máximo el rendimiento de la GPU en el procesamiento de grandes volúmenes de datos. A continuación, se detallan las optimizaciones realizadas para mejorar la eficiencia y velocidad del procesamiento.

1. Reducción Paralela con Memoria Compartida

   **Descripción**: La reducción (suma de todos los elementos) se realiza utilizando el kernel reduceSumOptimized, que ha sido optimizado para reducir el número de accesos a la memoria compartida y los puntos de sincronización.

   **Ventajas**: Al utilizar un patrón de reducción en memoria compartida, el kernel reduce el uso de la memoria global, lo que acelera el acceso a los datos y mejora la velocidad de procesamiento.

   **Implementación**: Cada bloque de hilos suma una sección de los datos en paralelo en la memoria compartida. La suma de cada bloque se reduce a un único valor que luego se combina en el host para obtener la suma total.

2. Ordenamiento Paralelo con Bitonic Sort

   **Descripción**: Para realizar el ordenamiento, se implementa el algoritmo Bitonic Sort, que es eficiente en GPUs y permite ordenar los datos en paralelo, especialmente si el tamaño de los datos es potencia de 2.

   **Ventajas**: Bitonic Sort es un algoritmo de ordenamiento paralelo que funciona bien en GPUs, ya que divide el ordenamiento en fases que pueden ejecutarse simultáneamente en múltiples hilos. Esto es significativamente más rápido que el ordenamiento secuencial (como el Bubble Sort).

   **Implementación**: Cada hilo compara e intercambia valores en paralelo. Esto asegura que los datos se ordenen de manera eficiente sin depender de algoritmos de ordenamiento secuenciales ineficientes.

3. Cálculo de Cuantiles

   **Descripción**: Después del ordenamiento, el kernel computeQuantiles calcula directamente los valores mínimos, Q1, mediana, Q3 y máximos usando los índices de los datos ordenados.

   **Ventajas**: Gracias al ordenamiento previo, los cuantiles se pueden calcular en tiempo constante accediendo directamente a los índices correspondientes en el conjunto de datos ordenado. Esto permite calcular los valores sin tener que iterar sobre el conjunto de datos.

   **Implementación**: El kernel calcula el mínimo, máximo, mediana, primer y tercer cuartil accediendo a los valores en sus posiciones correspondientes en los datos ordenados. Esto asegura una alta eficiencia en el cálculo de estos valores estadísticos.
