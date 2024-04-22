from model import ViTForClassfication

import torch
from torch import nn, optim

import os

from datasets.config import data_config
from datasets import load_data
from trainer import Trainer

torch.cuda.empty_cache()

batch_size = 64
epochs = 35
lr = 1e-4
save_model_every = 0

exp_name = f'vit-with-{epochs}-epochs'


device = 'cuda' if torch.cuda.is_available() else 'cpu'

for data in ['MNIST', 'CIFAR10', 'ImageNet200']:  
    config = data_config(data)

    # These are not hard constraints, but are used to prevent misconfigurations
    assert config["hidden_size"] % config["num_attention_heads"] == 0
    assert config["intermediate_size"] == 4 * config["hidden_size"]
    assert config["image_size"] % config["patch_size"] == 0

    img_size = (config["image_size"], config["image_size"])
    batch_size = 64  # 256
    print('Preparing data loaders.')
    trainloader, testloader, _ = load_data(
        name=config["name"], img_size=img_size, batch_size=batch_size
    )
    print('Done preparing data loaders.')
    epochs = 40  # 100
    lr = 1e-4
    save_model_every = 0  # 10

    save_model_every_n_epochs = save_model_every

    loss_fn = nn.CrossEntropyLoss()

    for random_features in [False, True]:
        if random_features:
            attention_type = "Performer-Softmax"
            m_range = [8, 16, 32, 64, 128]

            for m in m_range:
                exp_name = (
                    data + "_" + attention_type + "_with" + "_" + str(m)
                    + "_" + "random features"
                )

                print(f"Experiment: {exp_name}")

                model = ViTForClassfication(
                    config,
                    random_features=random_features,
                    relu=False, m=m
                )

                optimizer = optim.AdamW(model.parameters(),
                                        lr=lr, weight_decay=1e-2)

                trainer = Trainer(
                    model=model,
                    config=config,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    exp_name=exp_name,
                    device=device,
                )

                trainer.train(
                    trainloader,
                    testloader,
                    epochs,
                    save_model_every_n_epochs=save_model_every_n_epochs,
                )

        else:
            m = 1  # dummy variable
            for relu in [False, True]:
                if relu:
                    attention_type = "Performer-ReLU"
                else:
                    attention_type = "Transformer"

                exp_name = data + "_" + attention_type
                print(f"Experiment: {exp_name}")

                model = ViTForClassfication(
                    config,
                    random_features=random_features,
                    relu=relu, m=m
                )

                optimizer = optim.AdamW(model.parameters(),
                                        lr=lr, weight_decay=1e-2)

                trainer = Trainer(
                    model=model,
                    config=config,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    exp_name=exp_name,
                    device=device,
                )

                trainer.train(
                    trainloader,
                    testloader,
                    epochs,
                    save_model_every_n_epochs=save_model_every_n_epochs,
                )
