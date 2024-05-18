# A01610836 TC3002B Proyecto M2 

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
    