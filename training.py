from models.unet import Unet
from models.internals.losses import DiceLoss, CategoricalFocalLoss
from models.internals.metrics import JaccardIndex
from utils import parse_config

# libraries
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryFocalCrossentropy
import numpy as np


import logging
import yaml
from os.path import join
from os import mkdir
from datetime import timedelta
from time import time

from data_generator import LoadFromFolder


@parse_config
def main(config: dict) -> None:
    """ Load data
        Build and compile network
        Optimize network
        Save models and training history
    """


    folder_path = config["base_path"]



    ## BUILDING NETWORK GRAPH ##
    unet = Unet()
    unet.build(**config['unet']['build'])

    mkdir(folder_path)
    unet.save_model_graph(filename=join(folder_path,"network_graph.json"))
    with open(config["base_path"] + '/config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


    # compile
    dice_loss = DiceLoss()
    #metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    unet.compile(optimizer = optimizer,
                 loss = dice_loss,
                 metrics=JaccardIndex(threshold=0.5))


    ## CALLBACKS ##
    checkpoint_path = join(folder_path, "weights/cp.ckpt")
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 save_best_only=True,
                                 save_weights_only=True)
    callbacks=[checkpoint]
    

    ## TRAINING ##
    dataset = tf.data.Dataset.from_generator(LoadFromFolder,
                                             args=['data/imgs', 'data/labels'],
                                             output_types=((tf.float32), (tf.float32)),
                                             output_shapes=((800,1000,1), (800,1000,1))).batch(config['unet']['fit'].pop('batch_size'))
    unet.train(dataset,
               callbacks=callbacks,
               validation_dataset=dataset,
               **config['unet']['fit'])

    ## SAVING TRAINING HISTORY
    unet.save_training_history(join(folder_path, 'history.csv'))


if __name__ == '__main__':
    main()
