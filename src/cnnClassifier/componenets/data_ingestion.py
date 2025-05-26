import os 
import zipfile
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
import gdown

from cnnClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config = config



    def download_file(self) -> str:

        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts",exist_ok=True)
            logger.info(f"Downloading file from: {dataset_url} to: {zip_download_dir}")

            file_id= dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)
            logger.info(f"File id: {file_id} and prefix: {prefix}")
        except Exception as e:
            raise e
        

    def extratct_zip_file(self) :
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
            zip_ref.extractall(unzip_path)