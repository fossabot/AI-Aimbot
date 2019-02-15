from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.engine import Layer
from keras.engine import InputSpec
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.utils import conv_utils, plot_model
from keras.utils.data_utils import get_file
import datetime

from agents import agent_callbacks
from utilz import utils_data
import config
config = config.Config()
import os
cwd = os.getcwd()

from agents import agent

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, img_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                img_shape[1] if img_shape[1] is not None else None
            width = self.upsampling[1] * \
                img_shape[2] if img_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (img_shape[0],
                height,
                width,
                img_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SIS(agent.Agent):

    def __init__(self, mode):
        super()
        """
            Model Constructor
            Args:
                img_cols:        The x
                img_rows:        The y
                encodedDim:  Input layer dimensions
                color:       Whether a RGB image is used                  
        """
        self.img_cols = config.capture_mode.get(mode)[0]
        self.img_rows = config.capture_mode.get(mode)[1]
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.classes = len(config.SIS_ENTITIES)

        self.pretrained_weights = None

        self.sis = None


        return


    def build_sis(self, input_tensor=None, OS=16):
        """ Instantiates the Deeplabv3+ architecture

        Optionally loads pretrained_weights pre-trained
        on PASCAL VOC. This model is available for TensorFlow only,
        and can only be used with inputs following the TensorFlow
        data format `(width, height, channels)`.
        # Arguments
            pretrained_weights: one of 'pascal_voc' (pre-trained on pascal voc)
                or None (random initialization)
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            img_shape: shape of input image. format HxWxC
                PASCAL VOC model was trained on (512,512,3) images
            classes: number of desired classes. If classes != 21,
                last layer is initialized randomly
            OS: determines img_shape/feature_extractor_output ratio. One of {8,16}.
                Used only for xception backbone.


        # Returns
            A Keras model instance.

        # Raises
            RuntimeError: If attempting to run this model with a
                backend that does not support separable convolutions.
            ValueError: in case of invalid argument for `pretrained_weights` or `backbone`

        """

        def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
                """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
                    Implements right "same" padding for even kernel sizes
                    Args:
                        x: input tensor
                        filters: num of filters in pointwise convolution
                        prefix: prefix before name
                        stride: stride at depthwise conv
                        kernel_size: kernel size for depthwise convolution
                        rate: atrous rate for depthwise convolution
                        depth_activation: flag to use activation between depthwise & poinwise convs
                        epsilon: epsilon to use in BN layer
                """

                if stride == 1:
                    depth_padding = 'same'
                else:
                    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
                    pad_total = kernel_size_effective - 1
                    pad_beg = pad_total // 2
                    pad_end = pad_total - pad_beg
                    x = ZeroPadding2D((pad_beg, pad_end))(x)
                    depth_padding = 'valid'

                if not depth_activation:
                    x = Activation('relu')(x)
                x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                                    padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
                x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
                if depth_activation:
                    x = Activation('relu')(x)
                x = Conv2D(filters, (1, 1), padding='same',
                        use_bias=False, name=prefix + '_pointwise')(x)
                x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
                if depth_activation:
                    x = Activation('relu')(x)

                return x


        def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
            """Implements right 'same' padding for even kernel sizes
                Without this there is a 1 pixel drift when stride = 2
                Args:
                    x: input tensor
                    filters: num of filters in pointwise convolution
                    prefix: prefix before name
                    stride: stride at depthwise conv
                    kernel_size: kernel size for depthwise convolution
                    rate: atrous rate for depthwise convolution
            """
            if stride == 1:
                return Conv2D(filters,
                            (kernel_size, kernel_size),
                            strides=(stride, stride),
                            padding='same', use_bias=False,
                            dilation_rate=(rate, rate),
                            name=prefix)(x)
            else:
                kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
                pad_total = kernel_size_effective - 1
                pad_beg = pad_total // 2
                pad_end = pad_total - pad_beg
                x = ZeroPadding2D((pad_beg, pad_end))(x)
                return Conv2D(filters,
                            (kernel_size, kernel_size),
                            strides=(stride, stride),
                            padding='valid', use_bias=False,
                            dilation_rate=(rate, rate),
                            name=prefix)(x)


        def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                            rate=1, depth_activation=False, return_skip=False):
            """ Basic building block of modified Xception network
                Args:
                    inputs: input tensor
                    depth_list: number of filters in each SepConv layer. len(depth_list) == 3
                    prefix: prefix before name
                    skip_connection_type: one of {'conv','sum','none'}
                    stride: stride at last depthwise conv
                    rate: atrous rate for depthwise convolution
                    depth_activation: flag to use activation between depthwise & pointwise convs
                    return_skip: flag to return additional tensor after 2 SepConvs for decoder
                    """
            residual = inputs
            for i in range(3):
                residual = SepConv_BN(residual,
                                    depth_list[i],
                                    prefix + '_separable_conv{}'.format(i + 1),
                                    stride=stride if i == 2 else 1,
                                    rate=rate,
                                    depth_activation=depth_activation)
                if i == 1:
                    skip = residual
            if skip_connection_type == 'conv':
                shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                        kernel_size=1,
                                        stride=stride)
                shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
                outputs = layers.add([residual, shortcut])
            elif skip_connection_type == 'sum':
                outputs = layers.add([residual, inputs])
            elif skip_connection_type == 'none':
                outputs = residual
            if return_skip:
                return outputs, skip
            else:
                return outputs


        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        if not (self.pretrained_weights in {'pascal_voc', None}):
            raise ValueError('The `pretrained_weights` argument should be either '
                            '`None` (random initialization) or `pascal_voc` '
                            '(pre-trained on PASCAL VOC)')

        if input_tensor is None:
            img_input = Input(shape=self.img_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=self.img_shape)
            else:
                img_input = input_tensor


        if OS == 8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)

        x = Conv2D(32, (3, 3), strides=(2, 2),
                name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
        x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
        x = Activation('relu')(x)

        x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
        x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
        x = Activation('relu')(x)

        x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                            skip_connection_type='conv', stride=2,
                            depth_activation=False)
        x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                                skip_connection_type='conv', stride=2,
                                depth_activation=False, return_skip=True)

        x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                            skip_connection_type='conv', stride=entry_block3_stride,
                            depth_activation=False)
        for i in range(16):
            x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                                skip_connection_type='sum', stride=1, rate=middle_block_rate,
                                depth_activation=False)

        x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                            skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                            depth_activation=False)
        x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                            skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                            depth_activation=True)


        # end of feature extractor

        # branching for Atrous Spatial Pyramid Pooling

        # Image Feature branch
        #out_shape = int(np.ceil(img_shape[0] / OS))
        b4 = AveragePooling2D(pool_size=(int(np.ceil(self.img_shape[0] / OS)), int(np.ceil(self.img_shape[1] / OS))))(x)
        b4 = Conv2D(256, (1, 1), padding='same',
                    use_bias=False, name='image_pooling')(b4)
        b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
        b4 = Activation('relu')(b4)
        b4 = BilinearUpsampling((int(np.ceil(self.img_shape[0] / OS)), int(np.ceil(self.img_shape[1] / OS))))(b4)

        # simple 1x1
        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
        b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
        b0 = Activation('relu', name='aspp0_activation')(b0)


        # rate = 6 (12)
        b1 = SepConv_BN(x, 256, 'aspp1',
                        rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        b2 = SepConv_BN(x, 256, 'aspp2',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        b3 = SepConv_BN(x, 256, 'aspp3',
                        rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # concatenate ASPP branches & project
        x = Concatenate()([b4, b0, b1, b2, b3])


        x = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='concat_projection')(x)
        x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)

        # DeepLab v.3+ decoder

        # Feature projection
        # x4 (x2) block
        x = BilinearUpsampling(output_size=(int(np.ceil(self.img_shape[0] / 4)),
                                            int(np.ceil(self.img_shape[1] / 4))))(x)
        dec_skip1 = Conv2D(48, (1, 1), padding='same',
                        use_bias=False, name='feature_projection0')(skip1)
        dec_skip1 = BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation('relu')(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, 'decoder_conv0',
                    depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, 'decoder_conv1',
                    depth_activation=True, epsilon=1e-5)

        # you can use it with arbitary number of classes
        if self.classes == 21:
            last_layer_name = 'logits_semantic'
        else:
            last_layer_name = 'custom_logits_semantic'

        x = Conv2D(self.classes, (1, 1), padding='same', name=last_layer_name)(x)
        x = BilinearUpsampling(output_size=(self.img_shape[0], self.img_shape[1]))(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input

        self.sis = Model(inputs, x, name='deeplabv3plus')
        self.sis.summary()
        plot_model(self.sis, to_file="sis_plot.png", show_layer_names=True, show_shapes=True)

        # load pretrained_weights

        if self.pretrained_weights != None:
            self.sis.load_weights(cwd+self.pretrained_weights, by_name=True)
        return


    def train_sis(self):
        self.build_sis()

        #load testing and training data
        sis_train_x, sis_train_y, sis_test_x, sis_test_y = utils_data.load_sis(config.PATH_AUTOENCODER_TRAIN,config.PATH_AUTOENCODER_TEST)  

        #tensorboard --logdir path_to_current_dir/Graph/ to see visual progress     
        tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True, update_freq='epoch')
        callback = agent_callbacks.Segmentation_Callbacks()

        self.sis.fit(sis_train_x,sis_train_y,
                epochs=30,
                batch_size=1, 
                shuffle=False,
                validation_data=(sis_test_x,sis_test_y), 
                callbacks=[tbCallBack,callback])

        self.sis.save(cwd+config.PATH_MODELS+'sis_{0}.h5'.format(datetime.datetime.now().strftime("%m-%d_%H")))

        return