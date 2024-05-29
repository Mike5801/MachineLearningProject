# A01610836 TC3002B Proyecto M2 
El espacio de trabajo se encuentra en el archivo "A01610836_Project.ipynb" de este repositorio 

## Información del dataset
El dataset seleccionado se obtuvo en Kaggle: <a href="https://www.kaggle.com/datasets/jehanbhathena/weather-dataset">Kaggle Dataset</a>

Este dataset contiene 6862 imágenes de diferentes paisajes o escenarios con diferentes climas.

Las imágenes están clasificadas de la siguiente forma:
- dew: 698 imágenes
- fogsmog: 851 imágenes
- frost: 475 imágenes
- glaze: 639 imágenes
- hail: 591 imágenes
- lighting: 377 imágenes
- rain: 526 imágenes
- rainbow: 232 imágenes
- rime: 1160 imágenes
- sandstorm: 692 imágenes
- snow: 621 imágenes

## Separación del dataset
Link al dataset separado: <a href="https://drive.google.com/drive/folders/14jQl-QiEyWfMf4tNDnXXiczenaPqkY1S?usp=sharing">Google Drive Dataset</a>

El dataset contiene 11 clases. Para cada clase se crearán carpetas de train, validation y test.
- De esta manera, se podrá entrenar, validar y probar el modelo por cada clase
  - Si se juntan todas las imágenes en una sola carpeta y se realiza la separación del dataset, puede existir un sesgo. Es posible que en el set de train no existan imágenes de una clase. Por lo tanto el modelo no estaría preparado en caso de que se presente una imagen de una clase que no ha visto
- Se planea separar el dataset de cada clase en la siguiente división:
  - train dataset: 60%
  - validation dataset: 20%
  - test dataset: 20%
- En este caso el 80% del total de imágenes será utilizado para entrenar al modelo (60% para entrenarlo y 20% para validar su funcionamiento y realizar los ajustes necesarios al modelo con base en los resultados). 
- El restante 20% de las imágenes será utilizado para probar el modelo.
- La selección de las imágenes para cada separación (train, validation y test) se hará de manera aleatoria
  - Si se seleccionan las imágenes de manera secuencial, es posible causar un sesgo al modelo al momento de entrenarlo. 
    - Las imágenes que están al final o en medio pueden ser diferentes a las del inicio. Por lo que, sería recomendable escoger qué imágenes van para cada partición de manera aleatoria
  - Para esto se utilizará un script de python que separará el dataset.
    - El script utilizado que está en el archivo "split_dataset.py" está basado en el siguiente código fuente: <a href="https://www.kaggle.com/code/shuvostp/split-folders-for-train-test-val-split-of-images">Kaggle Split Dataset Script</a>

## Técnicas de escalamiento y Preprocesado
#### El avance de código se encuentra en el archivo de este repositorio: "A01610836_Avance_S2.ipynb"


El escalado de los pixeles de las imágenes se normalizo para que estuviera dentro de un rango de [0, 1].
- De esta manera, es más fácil hacer cálculos, ya que la magnitud sería más pequeña a comparación del rango normal de los pixeles [0, 255]

Para el preprocesado de la imagen, se configuró el ImageDataGenerator de la siguiente forma
- Para saber qué hace cada configuración del ImageDataGenerator se utilizó la siguiente fuente como referencia
<a href="https://towardsdatascience.com/exploring-image-data-augmentation-with-keras-and-tensorflow-a8162d89b844">Medium Data Augmentation</a>

```
train_datagen = ImageDataGenerator(
    rescale = 1./255, # Escalado de la imagen
    rotation_range = 100,
    zoom_range = [0.3, 0.9],
    horizontal_flip = True,
    shear_range = 10,
)
```
Carpeta de Google Drive con algunas imágenes generadas (se generó un batch de 8 imágenes sobre todo el set de train): <a href="https://drive.google.com/drive/folders/1Po9V5vg51P141g5MTXDkWGyPEZo4IZgp?usp=sharing">Augmentation</a>
### Justificación de la configuración utilizada:
  - Rotación de la imagen: es posible utilizar esta configuración mientras la rotación esté por debajo de los 180°.
    - Si es mayor a esta cantidad lo que está en el suelo puede confundirse con el cielo y viceversa. En paisajes con nieve o fondos grises y blancos, la rotación de 180° ocasionaría una confusión con imágenes de smog en el cielo, ya que el color es muy similar. 
  - Zoom: en este caso la configuración es en el rango de [0.3, 0.9]
    - De acuerdo con la referencia revisada, mientras el valor sea menor que 1, el zoom aplicado siempre será un acercamiento de la imagen
    - Siempre y cuando el acercamiento no sea muy extremo, la imagen tendrá las características visuales del paisaje original
      - En un paisaje de smog o nieve, si existe un acercamiento exagerado, puede que la imagen sea un fondo blanco o gris
    - Se plantea evitar el alejamiento en el zoom por las posibles afectaciones que puede ocasionar los pixeles perdidos
      - De manera predeterminada, rellena los pixeles con un pixel fijo que representa el último pixel antes de salirse del marco original de la imagen
        Esto causa ruido en la imagen, por lo que no se incluirá este tipo de Zoom
  - Volteado horizontal: el volteado horizontal sigue manteniendo las mismas características visuales de la imagen. 
    - Esto no representa ruido visual en la imagen para una persona independientemente del paisaje
  - Shear Range: mientras no sea el rango no sea muy pequeño o muy grande aún conserva las características visuales de la imagen original
    - Si se estira demasiado, la imagen deja de ser reconocible por el humano. Si se encoge la imagen, el problema de los pixeles perdidos surge, por lo tanto genera ruido en la imagen
    - Después de probar con la configuración con diferentes valores, se llegó a la conclusión de que un valor de 10 no genera los problemas mencionados.
### Justificación de omisión de parámetros en la configuración:
  - Volteado vertical: debido a que ocasiona el mismo efecto que la rotación de 180°, no se consideró esta configuración
  - "Width shift" y "Height shift": debido a los pixeles perdidos que puede ocasionar la imagen generada no se incluyó este parámetro
  - Brillo: para las imágenes que sean de noche o paisajes blancos, prácticamente generaría un fondo negro o blanco, por lo que se consideró como una fuente de ruido
  - Channel shift: cambiar los valores del canal de la imagen puede ocasionar cambios en los colores. Esto no es óptimo, ya que puede perder las características visuales de la imagen original

## Estado del arte del modelo
### Conceptos
- Capa convolucional (Convolutional layer): es una matriz llamada kernel que es pasada sobre la matriz de entrada para crear el mapa de features de la siguiente capa. Se ejecuta una operación llamada convolución al deslizar el kernel sobre la matriz de entrada. Para sección de la matriz de entrada se calcula la multiplicación de matrices con el Kernel mientras se desliza. [2]
- Funciones de activiación no lineales: Una función de activación es un nodo que está después de una capa convolucional. Esto se utiliza para hacer una transformación no lineal sobre la señal de entrada. ReLU (Rectified Linear Unit) es una función que regresará la entrada si es positiva, de lo contrario regresará cero. [2]
- Capa de agrupación (Pooling): una desventaja de la salida de la capa convolucional es que registra la posición exacta de los features de la entrada. Esto significa que durante cualquier modificación de la matriz como recortes, rotaciones, entre otros, resultará en un mapa de features diferente. Para contrarrestar este problema, se realiza un "down sampling" de la capa convolucional. Esto se logra aplicando una capa de agrupación (pooling). De esta manera, es posible representar los mismos features ante modificaciones pequeñas evitando que la salida cambie. [2]
  - Tipos de pooling [2]

![Tipos de pooling](https://github.com/Mike5801/MachineLearningProject/blob/main/images/TypesPooling.png?raw=true)
- Capa completamente conectada (Fully connected layer): Al final de la red neuronal convolucional, la salida de la última capa de agrupación actúa como una entrada a la capa completamente conectada (Fully Connected Layer). Puede haber más de una de estas capas en la arquitectura. "Completamente conectada" hace referencia a que todos los nodos de la capa uno están conectados con todos los nodos de la segunda capa. [2]

### Marco teoríco

Para encontrar una arquitectura optimizada es necesarop explorar los diferentes factores como número de capas y profundidad, número de features, el tamaño del batch, entre otros, para determinar su impacto en el desempeño de la red. [1]

La arquitectura propuesta para optimizar la clasificación de imágenes es:
- Capa convolucional de entrada: 32 feature maps con un tamaño de 3 x 3, ReLU como función de activación  y el límite de los pesos (max norm) establecido and a 3
- Capa de agrupación (Max Pool layer) con un tamaño de 2 x 2
- Capa convolucional de entrada: 64 feature maps con un tamaño de 3 x 3, ReLU como función de activación  y el límite de los pesos (max norm) establecido and a 3
- Capa de agrupación (Max Pool) con un tamaño 2 x 2
- Dropout establecido a 20%
- Capa convolucional de entrada: 64 feature maps con un tamaño de 3 x 3, ReLU como función de activación  y el límite de los pesos (max norm) establecido and a 3
- Capa de agrupación (Max Pool) con un tamaño 2 x 2
- Flatten layer
- Capa completamente conectada (fully connected layer) con 512 unidades y rectificador como función de activación
- Dropout establecido a 50%
- Capa de salida completamente conectada (fully connected layer) con 10 unidades y SoftMax como función de activación
<div style="text-align: right">[1]</div>

Imagen de la arquitectura propuesta:

![Imagen arquitectura](https://github.com/Mike5801/MachineLearningProject/blob/main/images/Optimized%20Architecture%20CNN.png?raw=true)
<div style="text-align: right">[1]</div>


En la configuración de la CNN para el entrenamiento se utilizó el optimizador Adam con un aprendizaje fijo (fixed-learning) de 0.001 y un batch size de 32 ejemplos. Para la métrica de loss, utilizó el binary cross-entropy (BCE) también llamado log loss. Por último el dropout rate se estableció a 0.2 para evitar el overfitting de la red. [1]

La arquitectura propuesta en [1] obtuvo el mejor porcentaje de accuracy comparado a los otros métodos. Del mismo modo, dicha arquitectura obtuvo un buen desempeño en las métricas de calidad (precision, recall, f1-score) para la gran mayoría de las clases.

Transfer learning es un método de re-utilizar un modelo pre-entrenado con conocimiento para otra tarea. Transfer learning puede ser usado para la clasificación, regresión y problemas de agrupación. Para este caso en partícular, el  artículo utilizara el modelo pre-entrenado de VGG-16 para clasificar imágenes. [2]

Para encontrar las diferencias entre diferentes modelos la metodología de [2] propone lo siguiente: primero implementan una CNN básica para clasificar las imágenes. Posteriormente ajustaran el modelo con imágenes aumentadas. Por último, utilizarán el modelo pre-entrenado VGG-16 para clasificar las imágenes.

Para cada uno de estas fases, calculan el accuracy y el loss del modelo para observar cómo va mejorando dependiendo de los ajustes y adiciones al modelo. [2]

I. CNN Básica
- Tres capas convolucionales con un tamaño de 3 x 3 y ReLU como función de activación
- Una capa de agrupación (Max Pool) de tamaño 2 x 2.
- Capa completamente conectada (fully connected layer) con 2 unidades para determinar si es perro o gato
<div style="text-align: right">[2]</div>

II. Imágenes aumentadas
- Utilizando Keras y la función ImageDataGenerator() con las siguientes características:
  - Zoom aleatorio con un factor de 0.3
  - Rotación aleatoria de 50°
  - Movimiento de la image horizontal y verticalmente por un factor de 0.2
  - Estiramiento por un factor de 0.2
  - Re-escalado de los pixeles a un intervalo normalizado de 0 y 1
<div style="text-align: right">[2]</div>

III. Usando un modelo pre-entrenado para extraer los features de las imágenes aumentadas
- En esta fase importan el modelo pre-entrenado de VGG-16 el cual está entrenado con los pesos de ImageNet
  - ImageNet es un projecto de investigación para desarrollar una base de datos de imágenes con sus anotaciones (imagen y etiqueta). VGG-16 ya ha aprendido features de bajo nivel como espacio, esquinas, rotaciones, brillo, formas. Este conocimiento puede ser transferido para  extraer features para un problema diferente
  - Al importar el modelo pre-entrenado mantienen las siguientes configuraciones
    - include_top = False evita importar la última capa del modelo pre-entrenado que se encarga de la clasificación, ya que se busca clasificar las imágenes del dataset del artículo y no usar las clasificaciones del modelo pre-entrenado
  - Después de extrear los features con el modelo pre-entrenado, se utiliza la capa completamente conectada del modelo básico (fully connected layer) para clasificar las imágenes en dos categorías: perro o gato.
<div style="text-align: right">[2]</div>

Con el primer modelo, después de los primeros 3 epochs el modelo con ya no era capaz de clasificar las imágenes de la separación de test a comparación de la separación de train, indicando un comportamiento de overfitting. [2]

Con el segundo modelo se utilizaron las imágenes aumentadas. La idea principal de usar estas imágenes aumentadas es agregar pequeñas variaciones de las imágenes sin dañar el objeto central para que la red sea más robusta cuando se enfrente al mundo real con situaciones similares. [2]

Por último, al utilizar el modelo pre-entrenado VGG-16 se obtuvieron resultados muy altos en la métrica de accuracy para la separación de test. [2]

Los resultados finales de cada modelo fueron: [2]
| Fase de metodología | Training accuracy | Test accuracy |
| - | - | - |
| CNN Básico | 98.20% | 72.40% |
| CNN con imágenes aumentadas | 81.30% | 89.20% |
| CNN con modelo VGG-16 e imágenes aumentadas | 86.50% | 95.40% |

El primer modelo construido genera un accuracy de 72.40% para el set de test. Después de ajustar el modelo con imágenes aumentadas, el accuracy incrementa a 79.20%. Por último, al añadir el modelo pre-entrenado, el accuracy incrementa a 95.40% [2]

## Implementación del modelo
Para la implementación del modelo, se utilizó la arquitectura descrita en [1].

![Resumen de arquitectura](https://github.com/Mike5801/MachineLearningProject/blob/main/images/my_model_summary.png?raw=true)

Para la compilación del modelo se utilizó la misma configuración especificada en [1].
- Optimizador: Adam
- Loss: Binary crossentropy
- Metricas: Accuracy

Algunas modificaciones que se hicieron a este modelo fueron:
- La última capa completamente conectada se estableció con 11 unidades para representar las 11 clases que tiene el dataset
- El tamaño de entrada de las imágenes fue modificado de 32 x 32 a 150 x 150, ya que la resolución de 32 x 32 causa ruido en las imágenes de los paisajes
  - Para igualar las dimensiones de las imágenes de las separaciones (train, validation y test), se utilizó el ImageDataGenerator para re-escalar las imágenes a 150 x 150. Del mismo modo, se normalizaron los pixeles para que estén en el rango de [0, 1]

### Metodología de entrenamiento del modelo
Para entrenar el modelo, se va a utilizar una metodología muy similar al descrito en el [2] para encontrar las diferencias entre el modelo con la arquitectura optimizada y un modelo que utiliza transfer learning para la extracción de features de las imágenes.

1. Crear un modelo para la clasificación de imágenes siguiendo [1]
2. Entrenar el modelo con las imágenes no aumentadas 
3. Evaluar el modelo con la separación de datos de validación y test
4. Entrenar el modelo con las imágenes aumentadas
5. Evaluar el modelo con la separación de datos de validación y test
6. Entrenar un modelo nuevo utilizando transfer learning y las imágenes aumentadas
7. Evaluar el modelo con la separación de datos de validación y test

Con base en la metodolgía de [2], se designaron los siguientes epochs por cada fase del modelo:
- Fase 1: Modelo de clasificación de imágenes entrenado sin imágenes aumentadas
  - Número de epochs: 30
    - El comportamiento observado en el artículo es que en menos de 30 epochs llega al overfitting
- Fase 2: Modelo de clasificación de imágenes entrenado con imágenes aumentadas
  - Número de epochs: 100
    - El comportamiento observado en el artículo es que 30 epochs no es suficiente para observar los cambios de las métricas. A los 100 epochs, el artículo muestra que ambas métricas (loss y accuracy) para ambas separaciones (train y validation) son casi iguales
- Fase 3: Modelo de clasificación de imágenes utilizando Transfer learning:
  - Número de epochs: 20
    - De acuerdo con el artículo, el modelo es capaz de tener un accuracy mayor al 90% en 20 epochs gracias al modelo pre-entrenado de VGG-16

### Evaluación del modelo
#### Fase 1
Los resultados de las métricas de accuracy y loss con el modelo entrenado sin las imágenes aumentadas fueron:

![Modelo_fase1_acc&loss](https://github.com/Mike5801/MachineLearningProject/blob/main/images/dev_model_stage_1_acc&loss.png?raw=true)

El modelo memorizó los patrones de la separación de train, ocasionando que al evaluar el modelo con la separación de validation no obtenga clasificaciones correctas. Por ello se concluye que para esta fase el modelo tiene overfitting. Esto se observa en las gráficas cuando los valores de train son mejores que los valores de validation.

Este comportamiento era esperado, de acuerdo con los resultados y la metodología descrita en [2].

Al probarlo contra la separación de los datos de test los resultados fueron los siguientes:

| Set | Accuracy | Loss |
|-|-|-|
| Train | 0.9923 | 0.0115 |
| Validation | 0.7180 | 0.2997 |
| Test | 0.7202 | 0.2969 |

Como se observa en la tabla, es posible confirmar el overfitting al comparar las métricas de accuracy y loss entre la separación de datos de train y test. Las métricas de la separación de train son mucho mejores que las obtenidas en test con una diferencia del 27% en accuracy y 0.2854 en loss.

Con respecto a las métricas para saber qué tan bien clasificó las imágenes de test, se presentan a continuación la matriz de confusión y las métricas de precision, recall y f1-score.

![Modelo_fase1_matriz_confusion](https://github.com/Mike5801/MachineLearningProject/blob/main/images/dev_model_stage_1_confmat.png?raw=true)

| class | precision | recall | f1-score |
|-|-|-|-|
| dew | 0.93 | 0.79 | 0.55
| fogsmog | 0.77 | 0.88 | 0.83
| frost | 0.62 | 0.46 | 0.53
| glaze | 0.62 | 0.58 | 0.60
| hail | 0.63 | 0.71 | 0.67
| lightning | 0.82 | 0.92 | 0.87
| rain | 0.66 | 0.58 | 0.62
| rainbow | 0.85 | 0.62 | 0.72
| rime | 0.76 | 0.78 | 0.77
| sandstorm | 0.88 | 0.82 | 0.85
| snow | 0.46 | 0.59 | 0.52

Con base en estas métricas podemos observar que las clases de snow, dew y frost son difíciles de clasificar para este modelo, ya que tienen un f1-score menor a 0.60.

Para hacer más robusto el modelo, en la fase 2 se entrena al modelo con las imágenes aumentadas. De esta manera, el modelo es forzado a ver modificaciones de las imágenes originales, haciendo que sea más difícil memorizar las imágenes de entrada.

#### Fase 2
Los resultados de las métricas de accuracy y loss con el modelo entrenado con las imágenes aumentadas fueron:

![Modelo_fase2_acc&loss](https://github.com/Mike5801/MachineLearningProject/blob/main/images/dev_model_stage_2_acc&loss.png?raw=true)

La diferencia entre las métricas de loss y accuracy ya no es tan grande entre la separación de los datos de train y validation. Esto quiere decir que el modelo sí logró aprender algunos patrones que le permiten clasificar a la imagen. Sin embargo, aún existe una diferencia mayor al 6% de las métricas, por lo que aún se puede considerar que se encuentra en overfitting.

Al probarlo contra la separación de los datos de test los resultados fueron los siguientes:

| Set | Accuracy | Loss |
|-|-|-|
| Train | 0.8001 | 0.0960 |
| Validation | 0.7400 | 0.1361 |
| Test | 0.7587 | 0.1274 |

Como se puede observar en la tabla, las métricas de accuracy y loss para la separación de test son muy cercanas a los valores obtenidos en la etapa de entrenamiento del modelo. Esto fue gracias a que las imágenes aumentadas provocaron que fuera más difícil memorizar las imágenes haciendo que las métricas de accuracy y loss sean menores durante el entrenamiento. Esto quiere decir que el modelo sí fue capaz de aprender algunos patrones en lugar de memorizar las imágenes.

Para saber qué tan bien aprendió patrones para reconocer y clasificar imágenes nuevas, también se obtuvo la matriz de confusión y las métricas precision, recall y f1-score

![Modelo_fase2_matriz_confusion](https://github.com/Mike5801/MachineLearningProject/blob/main/images/dev_model_stage_2_confmat.png?raw=true)

| class | precision | recall | f1-score |
|-|-|-|-|
| dew | 0.81 | 0.91 | 0.86
| fogsmog | 0.81 | 0.91 | 0.86
| frost | 0.57 | 0.57 | 0.57
| glaze | 0.65 | 0.59 | 0.62
| hail | 0.87 | 0.60 | 0.71
| lightning | 0.77 | 0.88 | 0.82
| rain | 0.69 | 0.80 | 0.74
| rainbow | 0.91 | 0.62 | 0.73
| rime | 0.82 | 0.82 | 0.82
| sandstorm | 0.94 | 0.78 | 0.85
| snow | 0.56 | 0.65 | 0.60

Como se puede observar, el modelo es capaz de detectar si una imagen pertenece a una clase de manera correcta con un mayor porcentaje a comparación del modelo de la fase 1. En este caso, la clase que puede ser difícil de clasificar de manera correcta es frost, ya que el valor de f1-score es menor a 0.60.

Para ver qué tan optimizado está la arquitectura propuesta por [1], ahora se implementará un modelo utilizando transfer learning con las mismas características planteadas por [2] en la tercera etapa de su metodología.

#### Fase 3



## Referencias

[1] Aamir M. et al, "An Optimized Architecture of Image Classification Using Convolutional Neural Network", Modern Education and Computer Science PRESS, 2019 <a href="https://www.mecs-press.org/ijigsp/ijigsp-v11-n10/IJIGSP-V11-N10-5.pdf">MECSS</a>

[2] Tammina S., "Transfer learning using VGG-16 with Deep Convolutional Nerual Network for Classifying Images", International Journal of Scientific and Research Publications, Vol. 9, Issue 10, 2019 <a href="https://www.researchgate.net/profile/Srikanth-Tammina/publication/337105858_Transfer_learning_using_VGG-16_with_Deep_Convolutional_Neural_Network_for_Classifying_Images/links/5dc94c3ca6fdcc57503e6ad9/Transfer-learning-using-VGG-16-with-Deep-Convolutional-Neural-Network-for-Classifying-Images.pdf?_sg%5B0%5D=started_experiment_milestone&origin=journalDetail&_rtd=e30%3D">International Journal of Scientific and Research</a>
