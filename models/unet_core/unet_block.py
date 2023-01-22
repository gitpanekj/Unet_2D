from tensorflow.keras.layers import MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Concatenate, Conv2D, Conv3D, Activation
import tensorflow as tf
from models.unet_core.conv_blocks import ConvBlock2D, ConvBlock3D, TransposedConvBlock2D, TransposedConvBlock3D
from models.unet_core.attention_gate_blocks import AttentionGate


class UnetBlock:
    """ Unet block class
    
        Configurable unet block constructor.
        To append this block to an existing tf.Graph call the
        initialized object of unet block with the previous node.

        example
        -------
        # create an object with given configuration
        unet = UnetBlock(input_shape=(52,192,192,1), n_depth=5, z_depth=2)
        i = Input((52,192,192,1))
        # append to the tf.Graph after the node Input
        u = unet(i)
    """
    def __init__(self, input_shape_: tuple,
                 n_depth: int, 
                 z_depth: int=None, 
                 n_filters: int=16,
                 n_conv_per_depth: int=2, 
                 normalization: str=None,
                 norm_kwargs: dict=None,
                 use_attention: list=False,
                 use_transconv: bool=False, 
                 output_probabilities: bool=True,
                 output_channels: int=1,
                 last_activation: str='sigmoid'):
        """
        Parameters
        ----------
            input_shape : tuple(int)
                Shape of data parsed to the unet block.
                Input shape must be compatible with transformations
                in the block. (For instance, you cannot perform
                pooling(2x2x2, stride 2) on data with shape (13,52,52,1))
            n_depth : int
                Number of level of the network including bottleneck
            z_depth : int | None
                The lowest level where a filter with shape 1x2x2 instead of
                2x2x2 is used in a downsampling and upsampling layer.
                Only valid for 3D processing -> Semi-3D processing.
                If None, 3D processing is performed in all levels.
            n_filters : int
                Base number of filters used in conv layers.
                The number doubles with every transition to a lower level.
            n_conv_per_depth : int
                Number of convolutional layers used in one conv block.
            normalization : str | None
                Name of normalization layer.
                supported: 'batch', 'layer'
                If None, no normalization is performed.
            norm_kwargs : dict | None
                Kwargs parsed to a normalization layer.
                For more info see doc. of given layer.
            use_attention : list[int] | False
                List of levels where attention gate should be used.
                Levels indexing starts with 0.
                Attention gate can be used only in levels 
                with index (0,...,self.n_depth - 1)
            use_transconv : bool
                If True, transposed convolution is used to 
                upsample data instead of UpSampling.
            output_probabilities : bool
                If True, another conv layer is used to create a map
                of probabilities after whole unet block.
                Else, return raw output from last conv layer.
            output_channels : int
                Determines number of channels (classes) 
                if output_probabilites is True.
                For binary segmentation -> 1
                For 3 class segmentation -> 3
            last_activation : str
                Activation used to map values to interval
                (0,1) when creating a probability map.
                For binary segmentation -> sigmoid
                For multi-class segmentation -> softmax
        """

        ndims = len(input_shape_)
        if ndims == 4:
            self.mode = '3D' # ZYXC
        elif ndims == 3:
            self.mode = '2D' # YXC
        else:
            raise Exception("Check input_shape_, \
                            only 2D and 3D data are available")

        self.input_shape_ = input_shape_
        self.n_depth = n_depth
        self.z_depth = z_depth if not isinstance(z_depth,type(None)) else 0

        # input shape validation with respect to network config
        assert self.input_shape_[0]%(2**(self.n_depth-1-self.z_depth)) == 0,\
            "Invalid shape along axis Z. Z must me divisible by 2**(self.n_depth-1-self.z_depth)."
        assert self.input_shape_[1]%(2**(self.n_depth-1)) == 0,\
            "Invalid shape along axis Y. Y must be divisible by 2**(self.n_depth - 1)"
        if self.mode == '3D':
            assert self.input_shape_[2]%(2**(self.n_depth-1)) == 0,\
             "Invalid shape along axis X. X must be divisible by 2**(self.n_depth - 1)"

        # Layers´ parameters
        self.n_filters = n_filters
        self.kernel_size = (3, )*2 if self.mode == "2D" else (3, )*3
        self.pool = (2,)*2 if self.mode == "2D" else (2,)*3
        self.norm = normalization
        self.norm_kwargs = norm_kwargs if isinstance(norm_kwargs, dict) else {}

        # Architecture
        self.n_conv_per_depth = n_conv_per_depth
        if isinstance(use_attention, bool):
            if use_attention == True:
                self.use_attention = [i for i in range(self.n_depth - 1)]
            else:
                self.use_attention = []
        elif isinstance(use_attention, list):
            assert all(list(map(lambda x: x < self.n_depth, use_attention))),\
                "Can not apply attention gate in undefined level of network"
            self.use_attention = use_attention 
        self.use_transconv = use_transconv

        # Output
        self.output_probabilities = output_probabilities
        self.output_channels = output_channels
        self.last_activation = last_activation

        # Defining layers
        self.conv_block = ConvBlock2D if self.mode == '2D'\
                     else ConvBlock3D

        self.pooling = MaxPooling2D if self.mode == '2D'\
                  else MaxPooling3D

        if self.use_transconv:
            self.upsampling = TransposedConvBlock2D if self.mode == '2D'\
                         else TransposedConvBlock3D
        else:
            self.upsampling = UpSampling2D if self.mode == '2D'\
                         else UpSampling3D

        self.attention_gate = AttentionGate

    def __call__(self, input_tensor):
        layer = input_tensor

        self.skip_layers = []

        # Encoder
        for i in range(0, self.n_depth-1):
            # pool filter size update
            if self.mode == "2D":
                updated_pool = self.pool
            else:
                if i < self.z_depth:
                    updated_pool = list(self.pool)
                    updated_pool[0] = 1
                    updated_pool = tuple(updated_pool)
                else:
                    updated_pool = self.pool

            # successive convolutional layers (1 block)
            for j in range(self.n_conv_per_depth):
                layer = self.conv_block(filters=self.n_filters*(2**i),
                                        kernel_size=self.kernel_size,
                                        activation=tf.nn.relu,
                                        padding="same",
                                        normalization=self.norm,
                                        norm_kwargs = self.norm_kwargs,
                                        name=f"Encoder_CONV_l.{i}-n.{j}")(layer)
            self.skip_layers.append(layer)
            layer = self.pooling(pool_size=updated_pool,
                                 strides=updated_pool,
                                 padding="valid",
                                 name=f"Encoder_POOLING_l.{i}")(layer)

        # Botleneck
        for b in range(self.n_conv_per_depth):
            layer = self.conv_block(filters=self.n_filters*(2**(self.n_depth-1)),
                                    kernel_size=self.kernel_size,
                                    activation=tf.nn.relu, padding="same",
                                    normalization=self.norm,
                                    norm_kwargs=self.norm_kwargs,
                                    name=f"Botleneck_CONV_n.{b}")(layer)
   

        # Decoder
        for i in reversed(range(self.n_depth-1)):
            # pool filter size update
            if self.mode == "2D":
                updated_pool = self.pool
            else:
                if i < self.z_depth:
                    updated_pool = list(self.pool)
                    updated_pool[0] = 1
                    updated_pool = tuple(updated_pool)
                else:
                    updated_pool = self.pool

            # Upsampling params setUp
            if self.use_transconv:
                self.upsampling_params = { 'filters': self.n_filters*(2**(i+1)),
                                            'kernel_size': updated_pool, # 2x2
                                            'strides': updated_pool,     # 2x2
                                            'activation': tf.nn.relu,       
                                            'normalization' : self.norm,
                                            'norm_kwargs' : self.norm_kwargs,
                                            'name': f"Decoder_TRANSPOSED_CONV_l.{i}"
                    }
            else:
                self.upsampling_params = { 'size': updated_pool,         # 2x2
                                           'name': f"Decoder_UPSAMPLING_l.{i}"
                    }
            
            # Upsampling and Skip connection application
            if i in self.use_attention:
                # Attention Gate
                ag_skip = self.attention_gate(mode=self.mode, level=i,
                                              normalization = 'layer',
                                              norm_kwargs = self.norm_kwargs)(layer, self.skip_layers[i])
                # Upsampling operation
                layer = self.upsampling(**self.upsampling_params)(layer)
                # Concatenation of skip*featuer_map and lower_block_features
                layer = Concatenate(name=f"SKIP_CONNECTION_l.{i}")([layer,ag_skip])
            else:
                # Upsampling operation
                layer = self.upsampling(**self.upsampling_params)(layer)
                # Concatention of skip adn lower_block_features
                layer = Concatenate(name=f"SKIP_CONNECTION_l.{i}")([layer, self.skip_layers[i]])


            # Successive convolutional layers (block)
            for j in range(self.n_conv_per_depth-1):
                layer = self.conv_block(filters=self.n_filters*(2**i),
                                        kernel_size=self.kernel_size,
                                        activation=tf.nn.relu,
                                        padding="same",
                                        normalization=self.norm,
                                        norm_kwargs=self.norm_kwargs,
                                        name=f"Decoder_CONV_l.{i}-n.{j}")(layer)
            layer = self.conv_block(filters=self.n_filters*(2**i),
                                    kernel_size=self.kernel_size,
                                    activation=tf.nn.relu,
                                    padding="same",
                                    normalization=self.norm,
                                    norm_kwargs=self.norm_kwargs,
                                    name=f"Decoder_CONV_l.{i}-n.{j+1}")(layer)

        # True when output is supposed to be in probability range
        if self.output_probabilities:
            if self.mode == '2D':
                output = Conv2D(filters=self.output_channels, kernel_size=1,
                                name="Probability")(layer)
                output = Activation(self.last_activation, name=self.last_activation)(output)
            else:
                output = Conv3D(filters=self.output_channels, kernel_size=1,
                                name="Probability")(layer)
                output = Activation(self.last_activation, name=self.last_activation)(output)
            return output

        return layer