from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.componenets.model_training import Training
from cnnClassifier import logger

STAGE_NAME= "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            training_config = ConfigurationManager().get_training_config()
            training = Training(config=training_config)
            training.get_base_model()
            training.train_valid_generator()
            training.train()
            

        except Exception as e:
            raise e