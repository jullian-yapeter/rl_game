# Common default params
class COMMON_PARAMS:
    CHECKPOINT_DIR = "saved_models"


# Linear module default params
class LINEAR_PARAMS:
    LEARNING_RATE = 0.001
    CHKPT_FILE = "linear"


# Linear module default params
class U_LINEAR_PARAMS:
    LEARNING_RATE = 0.001
    OUTPUT_DIM = 64
    NUM_LAYERS = 3
    CHKPT_FILE = "u_linear"

    @staticmethod
    def generate_uniform_linear_layers(input_dim, output_dim, num_layers):
        linear_layers = []
        if num_layers < 1:
            return linear_layers
        linear_layers.append({"in_dim": input_dim, "out_dim": output_dim})
        for _ in range(1, num_layers):
            linear_layer = {}
            linear_layer["in_dim"] = output_dim
            linear_layer["out_dim"] = output_dim
            linear_layers.append(linear_layer)
        return linear_layers


# Convolutional encoder default params
class CONV_ENCODER_PARAMS:
    LEARNING_RATE = 0.001
    OUTPUT_DIM = 64
    CHKPT_FILE = "conv_encoder"


# Doubling convolutional encoder default params
class D_CONV_ENCODER_PARAMS:
    LEARNING_RATE = 0.001
    OUTPUT_DIM = 64
    CHKPT_FILE = "d_conv_encoder"

    @staticmethod
    def generate_doubling_conv_layers(channels, side_len):
        conv_layers = []
        c = channels
        sl = side_len
        while sl > 1:
            conv_layer = {}
            conv_layer["kernel_size"] = 3
            conv_layer["stride"] = 2
            conv_layer["padding"] = 1
            conv_layer["in_channels"] = c
            conv_layer["out_channels"] = c * 2
            conv_layers.append(conv_layer)
            c *= 2
            sl /= 2
        return conv_layers
