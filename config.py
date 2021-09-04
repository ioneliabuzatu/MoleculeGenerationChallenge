import torch
import experiment_buddy

import torch.nn as nn

host = ""

hidden_size = 1024
embedding_dimension = 248
n_layers = 3
lr = 0.0005
optimizer = "adam"
momentum = 0.96
bidirectional = False

batch_size = 128
epochs = 300

criterion = nn.CrossEntropyLoss()

pretrained_filepath = ""
checkpoint_filepath = "checkpoint_generator_model.pt"

experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(
    host,
    sweep_yaml="",
    proc_num=1,
    wandb_kwargs={"entity": "ionelia"}
)
