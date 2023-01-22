import tensorflow as tf
# normalizations
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
# layers
from tensorflow.keras.layers import Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose


class BaseConvBlock:
    """ Base block for convolutional operations
        followed by normalization.

        Attributes
        ----------
        opp : tensorflow.keras.layers.Layer
            an operation transforming data in __call__
    """
    def __init__(self, normalization = None, norm_kwargs = {}, activation=None, **opp_kwargs):
        """
        Parameters
        ----------
        normalization : str
            normalization layer (batch, layer)
            if None, no normalization is done
        norm_kwargs : dict
            kwargs parsed to normalization layer
        activation : callable
            activation function used after normalization
            if None, default activation is tf.nn.relu
        opp_kwargs : dict
            kwargs parsed to an self.opp
        """
        self.opp_kwargs = opp_kwargs
        self.normalization = normalization
        self.activation = activation if not isinstance(activation, type(None)) else tf.nn.relu
        self.valid_norms = {"batch": BatchNormalization, "layer": LayerNormalization}


        if self.normalization:
            self.norm_kwargs = norm_kwargs
            self._normalization_setup()

    def __call__(self, input_tensor, training = False):
        x = self.opp(input_tensor)
        if self.normalization:
            x = self.norm_opp(x)
        return self.activation(x)

    def _normalization_setup(self):
        assert (self.normalization in self.valid_norms.keys()), f"{self.normalization} is not supported normalization method"
        self.norm_opp = self.valid_norms[self.normalization]
        self.norm_opp = self.norm_opp(**self.norm_kwargs)


class ConvBlock2D(BaseConvBlock):
    """ Subclass of BaseConvBlock with specified Conv2D convolutional operation """

    def __init__(self, normalization = None, norm_kwargs = {}, activation=None, **opp_kwargs):
        super(ConvBlock2D, self).__init__(normalization = normalization, norm_kwargs = norm_kwargs, activation=activation, **opp_kwargs)
        self.opp = Conv2D(**self.opp_kwargs)
    

class ConvBlock3D(BaseConvBlock):
    """ Subclass of BaseConvBlock with specified Conv3D convolutional operation """

    def __init__(self, normalization = None, norm_kwargs = {}, activation=None, **opp_kwargs):
        super(ConvBlock3D, self).__init__(normalization = normalization, norm_kwargs = norm_kwargs, activation=activation, **opp_kwargs)
        self.opp = Conv3D(**self.opp_kwargs)


class TransposedConvBlock2D(BaseConvBlock):
    """ Subclass of BaseConvBlock with specified Conv2DTranspose convolutional operation """

    def __init__(self, normalization = None, norm_kwargs = {}, activation=None, **opp_kwargs):
        super(TransposedConvBlock2D, self).__init__(normalization = normalization, norm_kwargs = norm_kwargs, activation=activation, **opp_kwargs)
        self.opp = Conv2DTranspose(**self.opp_kwargs)
        

class TransposedConvBlock3D(BaseConvBlock):
    """ Subclass of BaseConvBlock with specified Conv3DTranspose convolutional operation """

    def __init__(self, normalization = None, norm_kwargs = {}, activation=None, **opp_kwargs):
        super(TransposedConvBlock3D, self).__init__(normalization = normalization, norm_kwargs = norm_kwargs, activation=activation, **opp_kwargs)
        self.opp = Conv3DTranspose(**self.opp_kwargs)
