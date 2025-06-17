import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config



    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale=1./255,  # Normalize pixel values to [0, 1]
            validation_split=0.3  # Split data into training and validation sets
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation='bilinear'
        )
        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self._valid_generator= valid_datagen.flow_from_directory(
            directory=self.config.training_data,
            subset='validation',
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    



    def evaluation(self):
        self.model=self.load_model(self.config.path_of_mdel)
        self._valid_generator()
        self.score = self.model.evaluate(self._valid_generator)

    def save_score(self):
        score = {'loss':self.score[0], 'accuracy':self.score[1]}
        save_json(Path('artifacts/evaluation/score.json'),data=score)
        
        
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({'loss':self.score[0], 'accuracy':self.score[1]})

            if tracking_url_type_store != 'file':

                mlflow.keras.log_model(self.model, 'model',registered_model_name='VGG16MODEL')
            else:
                mlflow.keras.log_model(self.model, 'model')

           

        

     