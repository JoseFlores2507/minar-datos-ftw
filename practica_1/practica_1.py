#José María Flores San Martin - 1859565
#DataAdquisition

#importamos la libreria de kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

#descarga del dataset 
api.dataset_download_files('gregorut/videogamesales', path='./')

#descomprimir dataset
import zipfile
with zipfile.ZipFile('./videogamesales.zip', 'r') as zipref:
    zipref.extractall('videogamesales')