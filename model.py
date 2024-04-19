from torch import nn
import torch

from layers.embedding import Embeddings
from layers.encoder import Encoder


class ViTForClassfication(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, config, relu=False, m=16):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # Create the embedding module
        self.embedding = Embeddings(config)
        # Create the transformer encoder module
        self.relu = relu
        self.m = m
        self.encoder = Encoder(config, relu=self.relu, m=self.m)
        # Create a linear layer to project the encoder's output to 
        # the number of classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        # print(f'shape in model of x {x.shape}')
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # print(f'shape after embedding of x {x.shape}')
        # print(f'shape of embedding {embedding_output.shape}')
        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(
            embedding_output, output_attentions=output_attentions
        )
        # print(f'shape after encoder of x {x.shape}')
        # Calculate the logits, take the [CLS] token's output as 
        # features for classification
        logits = self.classifier(encoder_output[:, 0, :])
        # Return the logits and the attention probabilities (optional)
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config["initializer_range"]
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)
