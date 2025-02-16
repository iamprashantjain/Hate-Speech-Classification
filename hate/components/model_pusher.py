import os
import shutil
import sys
from hate.logger import logging
from hate.exception import CustomException
# from hate.configuration.gcloud_syncer import GCloudSync
from hate.entity.config_entity import ModelPusherConfig
from hate.entity.artifact_entity import ModelPusherArtifacts

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        """
        :param model_pusher_config: Configuration for model pusher
        """
        self.model_pusher_config = model_pusher_config
        # self.gcloud = GCloudSync()
    
    
    # def initiate_model_pusher(self) -> ModelPusherArtifacts:
    #     """
    #         Method Name :   initiate_model_pusher
    #         Description :   This method initiates model pusher.

    #         Output      :    Model pusher artifact
    #     """
    #     logging.info("Entered initiate_model_pusher method of ModelTrainer class")
    #     try:
    #         # Uploading the model to gcloud storage

    #         self.gcloud.sync_folder_to_gcloud(self.model_pusher_config.BUCKET_NAME,
    #                                           self.model_pusher_config.TRAINED_MODEL_PATH,
    #                                           self.model_pusher_config.MODEL_NAME)

    #         logging.info("Uploaded best model to gcloud storage")

    #         # Saving the model pusher artifacts
    #         model_pusher_artifact = ModelPusherArtifacts(
    #             bucket_name=self.model_pusher_config.BUCKET_NAME
    #         )
    #         logging.info("Exited the initiate_model_pusher method of ModelTrainer class")
    #         return model_pusher_artifact

    #     except Exception as e:
    #         raise CustomException(e, sys) from e
    
    
    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        """
            Method Name :   initiate_model_pusher
            Description :   This method initiates model pusher by copying the model to the dataset folder.

            Output      :    Model pusher artifact
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            # Ensure dataset directory exists
            dataset_dir = self.model_pusher_config.DATASET_DIR
            os.makedirs(dataset_dir, exist_ok=True)

            logging.info(f"Dataset directory: {dataset_dir} created/exists")

            # Define target path
            target_model_path = os.path.join(dataset_dir, self.model_pusher_config.MODEL_NAME)
            logging.info(f"Target model path: {target_model_path}")

            # Check if the source model path exists
            if not os.path.exists(self.model_pusher_config.TRAINED_MODEL_PATH):
                raise FileNotFoundError(f"Trained model file not found: {self.model_pusher_config.TRAINED_MODEL_PATH}")
            
            # Copy the trained model to the dataset folder
            shutil.copy(self.model_pusher_config.TRAINED_MODEL_PATH, target_model_path)

            logging.info(f"Copied best model to dataset folder at {target_model_path}")

            # Saving the model pusher artifacts
            model_pusher_artifact = ModelPusherArtifacts(
                bucket_name=dataset_dir  # Using dataset directory instead of bucket
            )
            logging.info("Exited the initiate_model_pusher method of ModelPusher class")
            return model_pusher_artifact

        except Exception as e:
            # Log the exception details
            logging.error(f"Error occurred: {str(e)}")
            raise CustomException(e, sys) from e
