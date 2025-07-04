from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.componenets.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger


STAGE_NAME='prepare_base_model'



class PrepareBaseModelPipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            config= ConfigurationManager()
            prepare_base_model_config=config.get_prepare_base_model_config()
            prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
            prepare_base_model.get_base_model()
            prepare_base_model.update_base_model()
        except Exception as e:
            raise e
        



if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
    