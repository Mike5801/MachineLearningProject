import splitfolders
import os
path = "/Users/Migue/OneDrive/Escritorio/Documentos de tarea/Universidad/Octavo semestre/Desarrollo de aplicaciones/M2/dataset"
print(os.listdir(path))
splitfolders.ratio(path, seed=1337, output="Splitted-Dataset", ratio=(0.6, 0.2, 0.2))
