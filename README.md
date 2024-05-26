# A01610836 TC3002B Proyecto M2 
Acceso de lectura al espacio de trabajo del proyecto: <a href="https://colab.research.google.com/drive/1LQanzyHwA0hZgATzanzLaqG0XwBa9Tf8?usp=sharing">Google Collab</a>

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
#### Conceptos
- Capa convolucional (Convolutional layer)
- Funciones de activiación no lineales
- Capa de agrupación (Pooling)
- Capa completamente conectada (Fully connected layer)

#### Artículo 1: "An Optimized Architecture of Image Classification Using Convolutional Neural Network"

Este artículo explora diferentes factores como número de capas y profundidad, número de features, el tamaño del batch, entre otros, para determinar su impacto en el desempeño de la red. Esto con el objetivo de encontrar una arquitectura para la clasificación de imágenes optimizada.

##### Descripción del dataset
Para este experimento, el artículo usó el dataset de CIFAR-10. Éste contiene 60,000 imágenes de color con una separación de training de 50,000 imágenes y una separación de test de 10,000 imágenes. Cada imagen tiene un tamaño de 32 x 32 pixeles.

CIFAR-10 está etiquetado con 10 clases: aviones, automóviles, pájaros, gatos, venados, perros, ranas, caballos, barcos y camiones.

##### Arquitectura del modelo
La arquitectura propuesta para optimizar la clasificación de imágenes es:
- Capa convolucional de entrada: 32 feature maps con un tamaño de 3 x 3, relu como función de activación  y el límite de los pesos (max norm) establecido and a 3
- Capa de agrupación (Max Pool layer) con un tamaño de 2 x 2
- Capa convolucional de entrada: 64 feature maps con un tamaño de 3 x 3, relu como función de activación  y el límite de los pesos (max norm) establecido and a 3
- Capa de agrupación (Max Pool) con un tamaño 2 x 2
- Dropout establecido a 20%
- Capa convolucional de entrada: 64 feature maps con un tamaño de 3 x 3, relu como función de activación  y el límite de los pesos (max norm) establecido and a 3
- Capa de agrupación (Max Pool) con un tamaño 2 x 2
- Flatten layer
- Capa completamente conectada (fully connected layer) con 512 unidades y rectificador como función de activación
- Dropout establecido a 50%
- Capa de salida completamente conectada (fully connected layer) con 10 unidades y SoftMax como función de activación

Imagen de la arquitectura propuesta: COLOCAR IMAGEN

##### Resultados
Comparación de la métrica accuracy con otros métodos que se han aplicado al mismo dataset

| Método | Accuracy % |
|--------|------------|
| Logistic Regression (Softmax) | 40.76 |
| KNN classifier | 22.84, 38.6 |
| Patternet | 41.325 |
| SVM | 55.22, 33.54 |
| Fast-learning shallow CNN | 75.86 |
| Arquitectura propuesta | 78.86 |

Rendimiento de la arquitectura propuesta sobre las 10 clases del dataset CIFAR-10

| Name | Precision (Correctness) | Recall (Completeness) | f1-score |
|-|-|-|-|
| Avión | 82.00 | 82.5 | 82.2 |
| Automóvil | 87.83 | 88.8 | 88.3 |
| Pájaro | 79.43 | 61.4 | 69.2 |
| Gato | 62.53 | 61.1 | 61.8 |
| Venado | 67.61 | 80.8 | 73.6 |
| Perro | 71.33 | 68.2 | 69.7 |
| Rana | 82.42 | 84.9 | 83.6 |
| Caballo | 81.72 | 81.4 | 81.5 |
| Barco | 87.48 | 88.1 | 87.7 |
| Camión | 81.69 | 85.7 | 83.6 |

La arquitectura propuesta en este artículo obtuvo el mejor porcentaje de accuracy comparado a los otros métodos. Del mismo modo, dicha arquitectura obtuvo un buen desempeño en las métricas de calidad (precision, recall, f1-score) para la gran mayoría de las clases.

##### Referencia del artículo

Aamir M. et al, "An Optimized Architecture of Image Classification Using Convolutional Neural Network", Modern Education and Computer Science PRESS, 2019

<a href="https://www.mecs-press.org/ijigsp/ijigsp-v11-n10/IJIGSP-V11-N10-5.pdf">Link del artículo</a>



