# import os
# import sys
# import keras
# import pickle
# import numpy as np
# import pandas as pd
# from hate.logger import logging
# from hate.exception import CustomException
# from keras.utils import pad_sequences
# from hate.constants import *
# # from hate.ml.model import ModelArchitecture
# # from hate.configuration.gcloud_syncer import GCloudSync
# # from keras.preprocessing.text import Tokenizer
# from sklearn.metrics import confusion_matrix
# from hate.entity.config_entity import ModelEvaluationConfig
# from hate.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts


# class ModelEvaluation:
#     def __init__(self, model_evaluation_config: ModelEvaluationConfig,
#                  model_trainer_artifacts: ModelTrainerArtifacts,
#                  data_transformation_artifacts: DataTransformationArtifacts):
#         """
#         :param model_evaluation_config: Configuration for model eva            model = model_architecture.get_model()
#         data transformation artifact stage
#         :param model_trainer_artifacts: Output reference of model trainer artifact stage
#         """

#         self.model_evaluation_config = model_evaluation_config
#         self.model_trainer_artifacts = model_trainer_artifacts
#         self.data_transformation_artifacts = data_transformation_artifacts
#         # self.gcloud = GCloudSync()


# def get_best_model_from_gcloud(self) -> str:
#     try:
#         logging.info("Entered the get_best_model_from_local method of Model Evaluation class")

#         # Define the artifacts directory path
#         artifacts_dir = "D:/ML Project/Hate-Speech-Classification/hate/artifacts"

#         # Check if the artifacts directory exists
#         if not os.path.exists(artifacts_dir):
#             raise FileNotFoundError(f"The artifacts directory was not found: {artifacts_dir}")

#         # List all subdirectories (timestamped folders) inside the 'artifacts' directory
#         subfolders = [f for f in os.listdir(artifacts_dir) if os.path.isdir(os.path.join(artifacts_dir, f))]

#         if not subfolders:
#             raise FileNotFoundError("No timestamped folders found in the artifacts directory.")

#         # Sort the subfolders to find the latest one based on the timestamp in the folder name
#         subfolders.sort(reverse=True)  # Sorting in descending order to get the latest folder first
#         latest_folder = subfolders[0]

#         # Construct the path to the 'ModelTrainerArtifacts' folder within the latest timestamped folder
#         model_folder_path = os.path.join(artifacts_dir, latest_folder, "ModelTrainerArtifacts")

#         # Check if the folder exists
#         if not os.path.exists(model_folder_path):
#             raise FileNotFoundError(f"ModelTrainerArtifacts folder not found in {model_folder_path}.")

#         # Define the path to the model file (model.h5)
#         model_file_path = os.path.join(model_folder_path, "model.h5")

#         # Check if the model file exists
#         if not os.path.exists(model_file_path):
#             raise FileNotFoundError(f"Model file 'model.h5' not found in {model_folder_path}.")

#         logging.info(f"Found best model at {model_file_path}")
#         logging.info("Exited the get_best_model_from_local method of Model Evaluation class")
        
#         return model_file_path

#     except Exception as e:
#         raise CustomException(e, sys) from e


#     # def get_best_model_from_gcloud(self) -> str:
#     #     """
#     #     :return: Fetch best model from gcloud storage and store inside best model directory path
#     #     """
#     #     try:
#     #         logging.info("Entered the get_best_model_from_gcloud method of Model Evaluation class")

#     #         os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok=True)

#     #         self.gcloud.sync_folder_from_gcloud(self.model_evaluation_config.BUCKET_NAME,
#     #                                             self.model_evaluation_config.MODEL_NAME,
#     #                                             self.model_evaluation_config.BEST_MODEL_DIR_PATH)

#     #         best_model_path = os.path.join(self.model_evaluation_config.BEST_MODEL_DIR_PATH,
#     #                                        self.model_evaluation_config.MODEL_NAME)
#     #         logging.info("Exited the get_best_model_from_gcloud method of Model Evaluation class")
#     #         return best_model_path
#     #     except Exception as e:
#     #         raise CustomException(e, sys) from e 
        

    
#     def evaluate(self):
#         """

#         :param model: Currently trained model or best model from gcloud storage
#         :param data_loader: Data loader for validation dataset
#         :return: loss
#         """
#         try:
#             logging.info("Entering into to the evaluate function of Model Evaluation class")
#             print(self.model_trainer_artifacts.x_test_path)

#             x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path,index_col=0)
#             print(x_test)
#             y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path,index_col=0)

#             with open('tokenizer.pickle', 'rb') as handle:
#                 tokenizer = pickle.load(handle)

#             load_model=keras.models.load_model(self.model_trainer_artifacts.trained_model_path)

#             x_test = x_test['tweet'].astype(str)

#             x_test = x_test.squeeze()
#             y_test = y_test.squeeze()

#             test_sequences = tokenizer.texts_to_sequences(x_test)
#             test_sequences_matrix = pad_sequences(test_sequences,maxlen=MAX_LEN)
#             print(f"----------{test_sequences_matrix}------------------")

#             print(f"-----------------{x_test.shape}--------------")
#             print(f"-----------------{y_test.shape}--------------")
#             accuracy = load_model.evaluate(test_sequences_matrix,y_test)
#             logging.info(f"the test accuracy is {accuracy}")

#             lstm_prediction = load_model.predict(test_sequences_matrix)
#             res = []
#             for prediction in lstm_prediction:
#                 if prediction[0] < 0.5:
#                     res.append(0)
#                 else:
#                     res.append(1)
#             print(confusion_matrix(y_test,res))
#             logging.info(f"the confusion_matrix is {confusion_matrix(y_test,res)} ")
#             return accuracy
#         except Exception as e:
#             raise CustomException(e, sys) from e
        

    
#     def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
#         """
#             Method Name :   initiate_model_evaluation
#             Description :   This function is used to initiate all steps of the model evaluation

#             Output      :   Returns model evaluation artifact
#             On Failure  :   Write an exception log and then raise an exception
#         """
#         logging.info("Initiate Model Evaluation")
#         try:

#             logging.info("Loading currently trained model")
#             trained_model=keras.models.load_model(self.model_trainer_artifacts.trained_model_path)
#             with open('tokenizer.pickle', 'rb') as handle:
#                 load_tokenizer = pickle.load(handle)

#             trained_model_accuracy = self.evaluate()

#             logging.info("Fetch best model from gcloud storage")
#             best_model_path = self.get_best_model_from_gcloud()

#             logging.info("Check is best model present in the gcloud storage or not ?")
#             if os.path.isfile(best_model_path) is False:
#                 is_model_accepted = True
#                 logging.info("glcoud storage model is false and currently trained model accepted is true")

#             else:
#                 logging.info("Load best model fetched from gcloud storage")
#                 best_model=keras.models.load_model(best_model_path)
#                 best_model_accuracy= self.evaluate()

#                 logging.info("Comparing loss between best_model_loss and trained_model_loss ? ")
#                 if best_model_accuracy > trained_model_accuracy:
#                     is_model_accepted = True
#                     logging.info("Trained model not accepted")
#                 else:
#                     is_model_accepted = False
#                     logging.info("Trained model accepted")

#             model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
#             logging.info("Returning the ModelEvaluationArtifacts")
#             return model_evaluation_artifacts

#         except Exception as e:
#             raise CustomException(e, sys) from e




import os
import sys
import keras
import pickle
import numpy as np
import pandas as pd
from hate.logger import logging
from hate.exception import CustomException
from keras.utils import pad_sequences
from hate.constants import *
from sklearn.metrics import confusion_matrix
from hate.entity.config_entity import ModelEvaluationConfig
from hate.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts,
                 data_transformation_artifacts: DataTransformationArtifacts):
        """
        :param model_evaluation_config: Configuration for model evaluation
        :param model_trainer_artifacts: Output reference of model trainer artifact stage
        :param data_transformation_artifacts: Data transformation artifact stage
        """
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        # self.gcloud = GCloudSync()

    def get_best_model_from_gcloud(self) -> str:
        try:
            logging.info("Entered the get_best_model_from_local method of Model Evaluation class")

            # Define the artifacts directory path
            artifacts_dir = "D:/ML Project/Hate-Speech-Classification/artifacts"

            # Check if the artifacts directory exists
            if not os.path.exists(artifacts_dir):
                raise FileNotFoundError(f"The artifacts directory was not found: {artifacts_dir}")

            # List all subdirectories (timestamped folders) inside the 'artifacts' directory
            subfolders = [f for f in os.listdir(artifacts_dir) if os.path.isdir(os.path.join(artifacts_dir, f))]

            if not subfolders:
                raise FileNotFoundError("No timestamped folders found in the artifacts directory.")

            # Sort the subfolders to find the latest one based on the timestamp in the folder name
            subfolders.sort(reverse=True)  # Sorting in descending order to get the latest folder first
            latest_folder = subfolders[0]

            # Construct the path to the 'ModelTrainerArtifacts' folder within the latest timestamped folder
            model_folder_path = os.path.join(artifacts_dir, latest_folder, "ModelTrainerArtifacts")

            # Check if the folder exists
            if not os.path.exists(model_folder_path):
                raise FileNotFoundError(f"ModelTrainerArtifacts folder not found in {model_folder_path}.")

            # Define the path to the model file (model.h5)
            model_file_path = os.path.join(model_folder_path, "model.h5")

            # Check if the model file exists
            if not os.path.exists(model_file_path):
                raise FileNotFoundError(f"Model file 'model.h5' not found in {model_folder_path}.")

            logging.info(f"Found best model at {model_file_path}")
            logging.info("Exited the get_best_model_from_local method of Model Evaluation class")
            
            return model_file_path

        except Exception as e:
            raise CustomException(e, sys) from e


    def evaluate(self):
        """
        :param model: Currently trained model or best model from gcloud storage
        :param data_loader: Data loader for validation dataset
        :return: loss
        """
        try:
            logging.info("Entering into the evaluate function of Model Evaluation class")
            print(self.model_trainer_artifacts.x_test_path)

            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path, index_col=0)
            print(x_test)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path, index_col=0)

            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)

            load_model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)

            x_test = x_test['tweet'].astype(str)

            x_test = x_test.squeeze()
            y_test = y_test.squeeze()

            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_matrix = pad_sequences(test_sequences, maxlen=MAX_LEN)
            print(f"Test sequences matrix: {test_sequences_matrix}")

            accuracy = load_model.evaluate(test_sequences_matrix, y_test)
            logging.info(f"The test accuracy is {accuracy}")

            lstm_prediction = load_model.predict(test_sequences_matrix)
            res = []
            for prediction in lstm_prediction:
                if prediction[0] < 0.5:
                    res.append(0)
                else:
                    res.append(1)

            cm = confusion_matrix(y_test, res)
            print(f"Confusion Matrix: \n{cm}")
            logging.info(f"Confusion Matrix: \n{cm}")
            return accuracy

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
            Method Name :   initiate_model_evaluation
            Description :   This function is used to initiate all steps of the model evaluation

            Output      :   Returns model evaluation artifact
            On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Initiate Model Evaluation")
        try:
            logging.info("Loading currently trained model")
            trained_model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)

            with open('tokenizer.pickle', 'rb') as handle:
                load_tokenizer = pickle.load(handle)

            trained_model_accuracy = self.evaluate()

            logging.info("Fetch best model from gcloud storage")
            best_model_path = self.get_best_model_from_gcloud()

            logging.info(f"Checking if the best model exists at {best_model_path}")
            if not os.path.isfile(best_model_path):
                is_model_accepted = True
                logging.info("Best model doesn't exist in GCloud storage, accepting the currently trained model.")
            else:
                logging.info("Load best model fetched from GCloud storage")
                best_model = keras.models.load_model(best_model_path)
                best_model_accuracy = self.evaluate()

                logging.info(f"Comparing model accuracies: Best Model Accuracy = {best_model_accuracy}, Trained Model Accuracy = {trained_model_accuracy}")
                if best_model_accuracy > trained_model_accuracy:
                    is_model_accepted = False
                    logging.info("Best model is better than the trained model. Trained model not accepted.")
                else:
                    is_model_accepted = True
                    logging.info("Trained model is better or equal to the best model. Trained model accepted.")

            model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
            logging.info("Returning the ModelEvaluationArtifacts")
            return model_evaluation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e



    # def get_best_model_from_gcloud(self) -> str:
    #     """
    #     :return: Fetch best model from gcloud storage and store inside best model directory path
    #     """
    #     try:
    #         logging.info("Entered the get_best_model_from_gcloud method of Model Evaluation class")

    #         os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok=True)

    #         self.gcloud.sync_folder_from_gcloud(self.model_evaluation_config.BUCKET_NAME,
    #                                             self.model_evaluation_config.MODEL_NAME,
    #                                             self.model_evaluation_config.BEST_MODEL_DIR_PATH)

    #         best_model_path = os.path.join(self.model_evaluation_config.BEST_MODEL_DIR_PATH,
    #                                        self.model_evaluation_config.MODEL_NAME)
    #         logging.info("Exited the get_best_model_from_gcloud method of Model Evaluation class")
    #         return best_model_path
    #     except Exception as e:
    #         raise CustomException(e, sys) from e