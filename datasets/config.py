SUPPORTED_DATASETS = ['cifar10', 'mnist', 'places365', 'imagenet200']


def data_config(dataset):
    if not isinstance(dataset, str):
        raise TypeError('Name of dataset should be str.'
                        f'Got {type(dataset)} instead.')
    dataset = dataset.lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f'Unsupported dataset: {dataset}')
    
    if dataset == 'cifar10':
        return CIFAR10
    elif dataset == 'mnist':
        return MNIST
    elif dataset == 'places365':
        return Places365
    elif dataset == 'imagenet200':
        return ImageNet200


patch_size = 4
hidden_size = 192

num_hidden_layers = 4
num_attention_heads = 8

hidden_dropout_prob = 0.0
attention_probs_dropout_prob = 0.0
initializer_range = 0.02

qkv_bias = True


CIFAR10 = {
    "name": "CIFAR10",
    "patch_size": patch_size,
    "hidden_size": hidden_size,
    "num_hidden_layers": num_hidden_layers,
    "num_attention_heads": num_attention_heads,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": hidden_dropout_prob,
    "attention_probs_dropout_prob": attention_probs_dropout_prob,
    "initializer_range": initializer_range,
    "image_size": 32,
    "num_classes": 10,
    "num_channels": 3,
    "qkv_bias": qkv_bias,
}

MNIST = {
    "name": "MNIST",
    "patch_size": patch_size,
    "hidden_size": hidden_size,
    "num_hidden_layers": num_hidden_layers,
    "num_attention_heads": num_attention_heads,
    "intermediate_size": 4 * hidden_size,
    "hidden_dropout_prob": hidden_dropout_prob,
    "attention_probs_dropout_prob": attention_probs_dropout_prob,
    "initializer_range": initializer_range,
    "image_size": 28,
    "num_classes": 10,
    "num_channels": 1,
    "qkv_bias": qkv_bias,
}

Places365 = {
    "name": "Places365",
    "patch_size": patch_size,
    "hidden_size": hidden_size,
    "num_hidden_layers": num_hidden_layers,
    "num_attention_heads": num_attention_heads,
    "intermediate_size": 4 * hidden_size,
    "hidden_dropout_prob": hidden_dropout_prob,
    "attention_probs_dropout_prob": attention_probs_dropout_prob,
    "initializer_range": initializer_range,
    "image_size": 28,
    "num_classes": 365,
    "num_channels": 3,
    "qkv_bias": qkv_bias,
}

ImageNet200 = {
    "name": "ImageNet200",
    "patch_size": patch_size,
    "hidden_size": hidden_size,
    "num_hidden_layers": num_hidden_layers,
    "num_attention_heads": num_attention_heads,
    "intermediate_size": 4 * hidden_size,
    "hidden_dropout_prob": hidden_dropout_prob,
    "attention_probs_dropout_prob": attention_probs_dropout_prob,
    "initializer_range": initializer_range,
    "image_size": 28,
    "num_classes": 200,
    "num_channels": 3,
    "qkv_bias": qkv_bias,
}
