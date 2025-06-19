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
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        print("Running model evaluation...")
        self.score = self.model.evaluate(self._valid_generator)
        print(f"Score: {self.score}")
        self.save_score()


    def save_score(self):
        path = Path("scores.json").resolve()
        print(f"Saving scores to: {path}")
        scores = {'loss': self.score[0], 'accuracy': self.score[1]}
        save_json(path, data=scores)

        
        
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

           

        

     