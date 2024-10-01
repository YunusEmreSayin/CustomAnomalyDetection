import itertools
import math
import subprocess
import sys

""" Defining the func that will downlaods needed libraries with pip """


def install_requirements():
    try:
        # installing needed libraries --> needed libraries stored in requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError as e:
        #Exception Handling
        print(f"Error occurred while installing requirements: {e}")
        sys.exit(1)


""" Installing libraries if they are not exists on current interpreter"""
if __name__ == "__main__":
    if (install_requirements()):
        print(f"[LOG] Requirements installed from requirements.txt")

""" Ignoring warnings for blocking cuda warning when cpu is on using"""
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

""" Importing Needed Libraries """
from lightning.pytorch.trainer import Trainer

from SerkonAnomalib.metrics import ManualThreshold
from SerkonAnomalib.data.utils.split import TestSplitMode
from SerkonAnomalib.engine import Engine
from matplotlib import pyplot as plt
import yaml
import cv2
from torchvision.transforms.functional import to_pil_image
from lightning.pytorch.callbacks import EarlyStopping
import time
from SerkonAnomalib.callbacks.metrics import _MetricsCallback
from SerkonAnomalib.callbacks.normalization.utils import get_normalization_callback
from SerkonAnomalib.callbacks.post_processor import _PostProcessorCallback
from SerkonAnomalib.callbacks.thresholding import _ThresholdCallback
from SerkonAnomalib.callbacks.timer import TimerCallback
from SerkonAnomalib.models.image.patchcore import Patchcore
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torchvision.transforms.v2 import Transform

from SerkonAnomalib.deploy.inferencers.torch_inferencer import TorchInferencer

from SerkonAnomalib.models.components.base.anomaly_module import AnomalyModule
from SerkonAnomalib.data.base import AnomalibDataModule
from SerkonAnomalib.deploy.export import InferenceModel

from SerkonAnomalib.data.image import Folder
from SerkonAnomalib.deploy import CompressionType, ExportType

from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from SerkonAnomalib.models.components.base.export_mixin import ExportMixin

if TYPE_CHECKING:
    pass
from lightning.pytorch.callbacks import Callback
import numpy as np
from SerkonAnomalib.data.utils.image import read_image

app_statement = 'predict'
""" app_statement options --> train | predict """

if __name__ == '__main__':

    print(f"[LOG] Application Started.. \nApplication Mode --> {app_statement}")

    ### Setting device for processes. If gpu/cuda is available set it , if its not set the cpu for our processes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """ Reading config file """
    config_path = Path("config.yaml")  # The path of config file on Google Drive
    print("Reading Config File....")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    """ Reading Config.yaml """
    print(config)  # Checking config content

    root_path = Path(config["dataset"]["path"])  # Defining root path from config

    """
    app_statement: train --> Training
    app_statement: predict --> Prediction
    """

    """ Training Process """
    if app_statement == 'train':
        """Defining Dataset and DataLoaders in here"""
        dataModule = Folder(
            name=config["dataset"]["name"],
            root=root_path,
            task="classification",
            test_split_mode=TestSplitMode.NONE,
            normal_dir=config["dataset"]["normal_dir"],
            abnormal_dir=config["dataset"]["abnormal_dir"],
            image_size=(224, 224),
            num_workers=config["dataset"]["num_workers"],
        )
        """
            Args:
              name--> The name for our dataset
              root--> The path of the root directory of dataset
              task--> Task type for our operations
                  ---Values:CLASSIFICATION,SEGMENTATION,DETECTION
              test_split_mode-->Seperating mode for test set
              normal_dir--> Path of the good images
              abnormal_dir--> Path of the defective images
              image_size--> Definition of the resolution of images we have
              num_workers--> Count of threads we will use

        """

        dataModule.setup()  # Defines the datamodule
        """Defining the dataloaders"""
        train_loader = dataModule.train_dataloader()

        """Setting up the Train,Test and Validation datasets from DataLoader"""
        # train
        i, data_train = next(enumerate(dataModule.train_dataloader()))
        print(data_train.keys(), data_train["image"].shape)
        img_train = to_pil_image(data_train["image"][0].clone())

        train_dataset = dataModule.train_data.samples

        """Log Commands"""
        print("TRAIN DATASET FEATURES")
        print(train_dataset.info())
        print("")
        print("IMAGE DISTRIBUTION BY CLASS")
        print("")
        desc_grouped = train_dataset[['label']].value_counts()
        print(desc_grouped)


        class UnassignedError(Exception):
            """Unassigned error."""


        """Defining the model"""
        model = Patchcore(
            backbone=config["model"]["backbone"],
            pre_trained=config["model"]["pre_trained"],
            coreset_sampling_ratio=config["model"]["coreset_sampling_ratio"],
            num_neighbors=config["model"]["num_neighbors"],
            layers=config["model"]["layers"],
        )

        if hasattr(model, 'to_torch'):
            print(f"[LOG] Model has attribute ---> 'create_expoort_root'")
        else:
            print(f"[LOG] Modle has not attribute !!")

        """
        Args:
          backbone--> The Neural Network for our model
          pre_trained--> Status of the model is pre_trained or not
          coreset_sampling_ratio--> This parameters controls how much data will be kept on memory
            This parameter effect the accuracy and speed of model
          num_neighbors-->this param controls the models closeness and kinship relationships

        """
        # model.state_dict()

        callbacks = [
            EarlyStopping(
                monitor="image_AUROC",
                mode="max",
                patience=10,
            ),
        ]

        # wandb üzerinden loglar görüntünlenebilir API key en üst kısımda text içinde var
        # wandb_logger = AnomalibWandbLogger(project="image_anomaly_detection",
        #             name="name_wandb_experiment")

        """Setting the optimal threshold value manually"""
        manual_threshold = ManualThreshold(default_value=16.5)  # 19.5
        print(f"Threshold Type: {type(manual_threshold)} Threshold Value: {manual_threshold.compute()}")

        """Setting up the engine"""
        engine = Engine(
            max_epochs=1,
            # callbacks=callbacks,
            pixel_metrics="AUPRO",
            accelerator=device.type,  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
            devices=1,
            # logger=wandb_logger,
            task='classification',
            threshold=manual_threshold,
            enable_model_summary=False,
            enable_progress_bar=False,
            enable_checkpointing=False)
        """
        Args:
          max_epochs--> The number of epochs we will train
          callbacks--> The callbacks we will use
          pixel_metrics-->
          accelerator--> Device selection for engine
          devices--> How many cpu's we will use
          logger--> The logging method we have
            wandb--> This is optional, wandb is a common logging service
          task--> Task type of the operation that engine we will run
          threshold--> The threshold value for inspection
        """

        """Training the model"""

        trainStart = time.time()
        # engine._cache.args['callbacks'] = []
        print("Fit...")
        #engine.fit(datamodule=dataModule, model=model)
        trainEnd = time.time()
        train_duration = (trainEnd - trainStart) * 1000
        print(f"Train duration: {train_duration:.2f} ms")  # train finished and we printed the train duration

        """Exporting weights that we will use for predictions"""
        print("Export weights...")
        path_export_weights = engine.export(export_type="torch",
                                            model=model,
                                            export_root="serkon",
                                            export_file_name="serkon_torch_weights"
                                            )
        print("path_export_weights: ", path_export_weights)

        print(engine.__dict__)
        # engine.log_train_params()
        #        print(engine.checkpoint_callback)
        print(engine.trainer.enable_model_summary)
        # print(engine._cache.args['callbacks'].clear())
        print(engine._cache.args['enable_model_summary'])
        print(engine._cache.args)

    #""" Predicting Process """
    elif app_statement == 'predict':

        print('Predicting....')
        # Defining the folder that will be scaned for predictions and defining threshold value

        defect_folder = root_path / config["dataset"]["abnormal_dir"]
        threshold = 0.7

        ###

        image_no = 1

        ###
        image_paths = itertools.chain(defect_folder.glob('*.png'), defect_folder.glob('*.bmp'))
        ### All predictions for test/defect folder
        for image_path in image_paths:

            ### Paths for prediction process
            image_path = f'dataset/fabric_tunus/test/defect/{image_no}.png'
            result_dir_path = Path("prediction_results/")
            result_path = result_dir_path / f'result{image_no}.png'

            ### Creating results folder if its not exist
            result_dir_path.mkdir(parents=True, exist_ok=True)
            path_export_weights = Path('serkon/serkon_torch.pt')
            print(f"Weights taking from ---> {path_export_weights}")

            ### defining inferencer
            inferencer = TorchInferencer(path_export_weights, device=device.type)

            result = inferencer.predict(image=image_path)
            image = result.image

            image_no += 1

            ### Segmentation process for predictions
            anomaly_mask = result.anomaly_map > threshold
            anomaly_locations = np.argwhere(anomaly_mask)

            print(f"[LOG] Anomaly locations-->{anomaly_locations}")

            ### Drawing a rectangle on the defective area
            if len(anomaly_locations) > 0:
                min_y, min_x = np.min(anomaly_locations, axis=0)
                max_y, max_x = np.max(anomaly_locations, axis=0)

                cv2.rectangle(image, (min_x, min_y), (max_x, max_y), color=(255, 0, 0), thickness=1)

            ### Visualization with matplotlib
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(result.image.astype(np.uint8))
            axs[0].set_title('Detected')
            # axs[0].axis('off')

            axs[1].imshow(result.anomaly_map, cmap='afmhot')
            axs[1].set_title('Anomaly Map')
            # axs[1].axis('off')

            axs[2].imshow(result.heat_map)
            axs[2].set_title("Heat Map")
            # axs[2].axis('off')

            plt.suptitle(
                f"Prediction Result | Anomaly Score : {round((result.pred_score * 100), 1)} | Threshold:{round((threshold * 100), 1)}")
            plt.savefig(result_path)
            print(f"[LOG] Prediction result saved --> {result_path} | Threshold --> {round((threshold * 100), 1)}")


    else:
        print(f"[LOG] Unassigned app_statement--> {app_statement}")
