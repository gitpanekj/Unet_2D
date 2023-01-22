from models.unet_core.unet_block import UnetBlock
from tensorflow.keras import Input, Model
from pandas import DataFrame
import json
import os

class Unet:
    """ Temporary unet model class that mediates comunication between
        internals, tensorflwo API and configuration scripts.
        
        Will be replaced for a subclass of tensorflow.models.Model in future.

        For more information about config. see doc. in unet_core/unet_block.py.
    """
    def __init__(self):
        self.model = None

    def build(self,
              input_shape_: tuple,
              n_depth: int,
              z_depth: int=0,
              n_filters: int=16,
              n_conv_per_depth: int=2,
              normalization: str=None,
              norm_kwargs: dict=None,
              use_attention=False,
              use_transconv: bool=False,
              output_probabilities: bool=True,
              output_channels: int=1,
              last_activation: str='sigmoid'):

        i = Input(input_shape_)
        x = UnetBlock(input_shape_=input_shape_,
                      n_depth=n_depth,
                      z_depth=z_depth,
                      n_filters=n_filters,
                      n_conv_per_depth=n_conv_per_depth,
                      normalization=normalization,
                      norm_kwargs=norm_kwargs,
                      use_attention=use_attention,
                      use_transconv=use_transconv,
                      output_probabilities=output_probabilities,
                      output_channels=output_channels,
                      last_activation=last_activation)(i)
        self.model = Model(i,x)
        self.model.summary()

    def compile(self, optimizer, loss, metrics=None, **kwargs):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def train(self,
              training_dataset,
              epochs=None,
              steps_per_epoch=None,
              validation_dataset=None,
              validation_steps=None,
              callbacks=None,
              **kwargs):
        assert not isinstance(epochs, type(None)), "Number of training epochs must be specified"
        assert not isinstance(steps_per_epoch, type(None)), "Number of steps per epoch must be specified when using an infinite dataset"
        self.model.fit(training_dataset,
                       steps_per_epoch=steps_per_epoch,
                       epochs=epochs,
                       validation_data=validation_dataset,
                       validation_steps=validation_steps,
                       callbacks=callbacks, **kwargs)

    def load_model(self, config_path, weights_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            if isinstance(config, str): config = json.loads(config)
        self.model = Model().from_config(config['config'])
        
        self.model.load_weights(weights_path).expect_partial()

    def save_model_graph(self, filename):
        assert not isinstance(self.model, type(None)), 'Model has not been built yet'
        cfg = json.loads(self.model.to_json())
        with open(filename, 'w') as f:
            json.dump(cfg, f, indent=2)
         
    def save_model_weights(self, filename):
        assert not isinstance(self.model, type(None)), 'Model has not been built yet'
        self.model.save_weights(filename)
    
    def save_training_history(self, path):
        assert not isinstance(self.model, type(None)), 'Model has not been built yet'
        DataFrame.from_dict(self.model.history.history).to_csv(path)
