from tensorflow.keras.layers import Layer, Conv2D, Conv3D,UpSampling2D, UpSampling3D, Add, Multiply, Activation, BatchNormalization, LayerNormalization
import numpy as np


class AttentionGate:
    """ Attention gate structure

        Highlight significant areas in skip connection feature maps.

        Based on the feature maps from a lower level and skip connection in
        a current level of network create a probability-map and multiply it with the
        feature maps from skip connection. Then concatenate skip connection and
        upsampled data from lower level.

        Attention gate graph
        (  Lower block  )──────»[conv(1x1x1xN)]──┐ (g)                                               Pixel-wise mull
                                                 │                                                          ▼
                            (pixel-wise add) ► [+]──»[ReLU]──»[conv(1x1x1x1)]──»[Sigmoid]──»[Upsampling]──[*]── ...
                                                 │                                                          │
        (skip connection)──┬───»[conv(1x1x1xN)]──┘ (f)                                                      │
                           └────────────────────────────────────────────────────────────────────────────────┘


        Executes lines in __call__ to add separate nodes to network graph
    """

    def __init__(self, mode: str, level: int = 0, normalization = None, norm_kwargs = {}):
        self.mode = mode
        self.level = level
        self.conv = Conv2D if mode == '2D' else Conv3D
        self.upsample = UpSampling2D if mode == '2D' else UpSampling3D
        self.normalization = normalization

        # validate normalization method
        if self.normalization != None:
            self.valid_norms = {"batch": BatchNormalization, "layer": LayerNormalization}
            assert self.normalization in self.valid_norms.keys(), f"{self.normalization} is not supported normlization method"
            self.norm_opp = self.valid_norms[self.normalization]
            self.norm_kwargs = norm_kwargs

    def __call__(self, lower_block, skip):
        # calculate parameters of conv layers based on lower_block and skip connection properties
        skip_filters = skip.shape[-1]
        self.kernel_1 = (1,)*2 if self.mode == '2D' else (1,)*3
        self.kernel_2 = (np.array(skip.shape[1:-1]) // np.array(lower_block.shape[1:-1])).astype(int)
        self.strides_2 = self.kernel_2


        # process a tensor from a lower net block to match number of channels of a skip connection
        g = self.conv(filters = skip_filters, kernel_size = self.kernel_1, name = f"AG_matching_channels-{self.level}")(lower_block)
        if self.normalization != None:
            g = self.norm_opp(**self.norm_kwargs)(g)
        
        # process a tensor from a skip connection to match ZYX size
        f = self.conv(filters = skip_filters, kernel_size = self.kernel_2, strides = self.strides_2, name = f"AG_matching_ZYX-{self.level}")(skip)
        if self.normalization != None:
            f = self.norm_opp(**self.norm_kwargs)(f)
        
        
        add = Add(name=f"AG_add-{self.level}")([f,g]) # pixel-wise addition on unified lower block tensor (g) and skip connection tensor (f)
        rel = Activation('relu', name=f"AG_relu-{self.level}")(add) # nonlinear activation
        f_map = self.conv(filters = 1, kernel_size = self.kernel_1, activation = 'sigmoid', name=f"AG_weight_map-{self.level}")(rel) # weight map creation
        upsample = self.upsample(self.strides_2, name=f"AG_upsampling-{self.level}")(f_map) # upsampling to original resolution of skip connection tensor

        return Multiply(name=f"AG_multiplication-{self.level}")([upsample, skip]) # multiplication of skip connection and weight map produced by Attention Gate

class AttentionGate_debug(Layer):
    """ Attention gate structure

        For better understanding see AttentionGate (+comments)
    
        AttentionGate_debug is considered to be a single layer
        so that it´s not possible to inspect individual defined layers
        as network graph nodes (whole layer is a node)
    """
    def __init__(self, mode, level = 0, normalization = None, norm_kwargs = {}):
        super(AttentionGate_debug, self).__init__(name = f"ATTENTION_GATE_l.{level}")

        self.mode = mode
        self.level = level
        self.conv = Conv2D if mode == '2D' else Conv3D
        self.upsample = UpSampling2D if mode == '2D' else UpSampling3D
        self.normalization = normalization

        if self.normalization != None:
            self.valid_norms = {"batch": BatchNormalization, "layer": LayerNormalization}
            assert self.normalization in self.valid_norms.keys(), f"{self.normalization} is not supported normlization method"
            self.norm_opp = self.valid_norms[self.normalization]
            self.norm_kwargs = norm_kwargs

    def call(self, lower_block, skip):
        skip_shape = skip.shape[-1]
        self.kernel_1 = (1,)*2 if self.mode == '2D' else (1,)*3
        self.kernel_2 = (np.array(skip.shape[1:-1]) // np.array(lower_block.shape[1:-1])).astype(int)
        self.strides_2 = self.kernel_2
        
        # lower block
        g = self.conv(filters = skip_shape, kernel_size = self.kernel_1)(lower_block)
        if self.normalization != None:
            g = self.norm_opp(**self.norm_kwargs)(g)
        # skip connection
        f = self.conv(filters = skip_shape, kernel_size = self.kernel_2, strides = self.strides_2)(skip)
        if self.normalization != None:
            f = self.norm_opp(**self.norm_kwargs)(f)

        # add
        add = Add()([f,g])
        rel = Activation('relu')(add)
        f_map = self.conv(filters = 1, kernel_size = self.kernel_1, activation = 'sigmoid')(rel)
        upsample = self.upsample(self.strides_2)(f_map)

        # mul
        return Multiply()([upsample, skip])