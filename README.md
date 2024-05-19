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
- Se planea separar el dataset de cada clase en en la siguiente división:
  - train dataset: 60%
  - validation dataset: 20%
  - test dataset: 20%
- En este caso el 80% del total de imágenes será utilizado para entrenar al modelo (60% para entrenarlo y 20% para validar su funcionamiento y realizar los ajustes necesarios al modelo con base en los resultados). 
- El restante 20% de las imágenes será utilizado para probar el modelo.
- La selección de las imágenes para cada separación se (train, validation y test) hará de manera aleatoria
  - Si se seleccionan las imágenes de manera secuencial, es posible causar un sesgo al modelo al momento de entrenarlo. 
    - Las imágenes que están al final o en medio pueden ser diferentes a las del inicio. Por lo que, sería recomendable escoger qué imágenes van para cada partición de manera aleatoria
  - Para esto se utilizará un script de python que separará el dataset.
    - El script utilizado que está en el archivo "split_dataset.py" está basado en el siguiente código fuente: <a href="https://www.kaggle.com/code/shuvostp/split-folders-for-train-test-val-split-of-images">Kaggle Split Dataset Script</a>

## Técnicas de escalamiento y Preprocesado
#### El avance de código se encuentra en el archivo de este repositorio: "A01610836_Avance_S2.ipynb"


El escalado de los pixeles de las imágenes se normalizo para que estuviera dentro de un rango de [0, 1].
- De esta manera, es más fácil hacer cálculos, ya que la magnitud sería más pequeña a comparación del rango normal de los pixeles [0, 255]

Para el preprocesado de la imágen, se configuró el ImageDataGenerator de la siguiente forma
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
  - Rotación de la imágen: es posible utilizar esta configuración mientras la rotación esté por debajo de los 180°.
    - Si es mayor a esta cantidad lo que está en el suelo puede confundirse con el cielo y viceversa. En paisajes con nieve o fondos grises y blancos, la rotación de 180° ocasionaría una confusión con imágenes de smog en el cielo, ya que el color es muy similar. 
  - Zoom: en este caso la configuración es en el rango de [0.3, 0.9]
    - De acuerdo con la referencia revisada, mientras el valor sea menor que 1, el zoom aplicado siempre será un acercamiento de la imagen
    - Siempre y cuando el acercamiento no sea muy extremo, la imágen tendrá las características visuales del paisaje original
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
  - "width shift", "height shift": debido a los pixeles perdidos que puede ocasionar la imagen generada no se incluyó este parámetro
  - Brillo: para las imágenes que sean de noche o paisajes blancos, practicamente generaría un fondo negro o blanco, por lo que se consideró como una fuente de ruido
  - Channel shift: cambiar los valores del canal de la imagen puede ocasionar cambios en los colores. Esto no es óptimo, ya que puede perder las características visuales del a imagen original

