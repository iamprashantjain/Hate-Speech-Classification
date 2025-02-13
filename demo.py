from hate.logger import logging
from hate.exception import CustomException
import sys
from hate.configuration.gcloud_syncer import GCloudSync


obj = GCloudSync()
obj.sync_folder_from_gcloud("hate-speech-2024","dataset.zip", "dataset/dataset.zip")


