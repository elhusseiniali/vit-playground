patch_size = 4

num_hidden_layers = 4
num_attention_heads = 8

hidden_dropout_prob = 0.0
attention_probs_dropout_prob = 0.0
initializer_range = 0.02

qkv_bias = True


CIFAR10 = {
    "patch_size": patch_size,
    "hidden_size": 48,
    "num_hidden_layers": num_hidden_layers,
    "num_attention_heads": num_attention_heads,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": hidden_dropout_prob,
    "attention_probs_dropout_prob": attention_probs_dropout_prob,
    "initializer_range": initializer_range,
    "image_size": 32, # image_size of CIFAR10 
    "num_classes": 10, # num_classes of CIFAR10
    "num_channels": 3, # num_channels of CIFAR10 - color
    "qkv_bias": qkv_bias,
}

MNIST = {
    "patch_size": patch_size,
    "hidden_size": 48,
    "num_hidden_layers": num_hidden_layers,
    "num_attention_heads": num_attention_heads,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": hidden_dropout_prob,
    "attention_probs_dropout_prob": attention_probs_dropout_prob,
    "initializer_range": initializer_range,
    "image_size": 28, # image_size of MNIST
    "num_classes": 10, # num_classes of MNIST
    "num_channels": 1, # num_channels of MNIST - greyscale
    "qkv_bias": qkv_bias,
}

Places365 = {
    "patch_size": patch_size,
    "hidden_size": 48,
    "num_hidden_layers": num_hidden_layers,
    "num_attention_heads": num_attention_heads,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": hidden_dropout_prob,
    "attention_probs_dropout_prob": attention_probs_dropout_prob,
    "initializer_range": initializer_range,
    "image_size": 64, # image_size of Places365 - variable - convention is 256 but CUDA memory
    "num_classes": 365, # num_classes of Places365
    "num_channels": 3, # num_channels of Places365 - color
    "qkv_bias": qkv_bias,
}

ImageNet200 = {
    "patch_size": patch_size,
    "hidden_size": 48,
    "num_hidden_layers": num_hidden_layers,
    "num_attention_heads": num_attention_heads,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": hidden_dropout_prob,
    "attention_probs_dropout_prob": attention_probs_dropout_prob,
    "initializer_range": initializer_range,
    "image_size": 64, # image_size of ImageNet - variable - convention is 224 but CUDA memory
    "num_classes": 200, # num_classes of ImageNet
    "num_channels": 3, # num_channels of ImageNet - color
    "qkv_bias": qkv_bias,
}


data_config = {
    'CIFAR10': CIFAR10,
    'MNIST': MNIST,
    'Places365': Places365,
    'ImageNet200': ImageNet200
}
