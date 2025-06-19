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
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
    

   