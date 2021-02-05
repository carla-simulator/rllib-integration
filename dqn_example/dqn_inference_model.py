#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import torch
import torch.nn as nn

def get_activation_fn(name=None):
    if name in ["linear", None]:
        return None
    if name == "relu":
        return nn.ReLU
    elif name == "tanh":
        return nn.Tanh
    raise ValueError("Unknown activation ({})!".format(name))


class SlimConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, stride, padding,
                 initializer = "default", activation_fn = "default", bias_init = 0):

        super(SlimConv2d, self).__init__()
        layers = []

        # Padding layer.
        if padding:
            layers.append(nn.ZeroPad2d(padding))

        # Actual Conv2D layer (including correct initialization logic).
        conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
        if initializer:
            if initializer == "default":
                initializer = nn.init.xavier_uniform_
            initializer(conv.weight)
        nn.init.constant_(conv.bias, bias_init)
        layers.append(conv)

        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn)
        if activation_fn is not None:
            layers.append(activation_fn())

        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)


class SlimFC(nn.Module):

    def __init__(self, in_size, out_size, initializer=None, activation_fn=None, use_bias=True, bias_init=0.0):

        super(SlimFC, self).__init__()
        layers = []

        # Actual nn.Linear layer (including correct initialization logic).
        linear = nn.Linear(in_size, out_size, bias=use_bias)
        if initializer:
            initializer(linear.weight)
        if use_bias is True:
            nn.init.constant_(linear.bias, bias_init)
        layers.append(linear)

        # Activation function (if any; default=None (linear)).
        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn)
        if activation_fn is not None:
            layers.append(activation_fn())

        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)


class CustomDQNModel(nn.Module):

    def __init__(self, gpu_n=0):

        nn.Module.__init__(self)

        self._gpu_n = gpu_n

        # Convolutional layer
        convs = nn.Sequential()
        convs.add_module("{}".format(0), SlimConv2d(12, 16, kernel=[5, 5], stride=4, padding=(0, 1, 0, 1), activation_fn="relu"))
        convs.add_module("{}".format(1), SlimConv2d(16, 32, kernel=[5, 5], stride=2, padding=(2, 2, 2, 2), activation_fn="relu"))
        convs.add_module("{}".format(2), SlimConv2d(32, 32, kernel=[5, 5], stride=2, padding=(1, 2, 1, 2), activation_fn="relu"))
        convs.add_module("{}".format(3), SlimConv2d(32, 64, kernel=[5, 5], stride=1, padding=(2, 2, 2, 2), activation_fn="relu"))
        convs.add_module("{}".format(4), SlimConv2d(64, 64, kernel=[5, 5], stride=2, padding=(2, 2, 2, 2), activation_fn="relu"))
        convs.add_module("{}".format(5), SlimConv2d(64, 128, kernel=[5, 5], stride=2, padding=(1, 2, 1, 2), activation_fn="relu"))
        convs.add_module("{}".format(6), SlimConv2d(128, 256, kernel=[5, 5], stride=1, padding=0, activation_fn="relu"))
        self._convs = convs

        # Ray creates this layer but it is never used. Needed to avoid failures when loading the state dictionary
        value_branch = SlimFC(256, 1, activation_fn=None)
        self._value_branch = value_branch

        advantage_module = nn.Sequential()
        advantage_module.add_module("dueling_A_{}".format(0), SlimFC(256, 256, activation_fn='relu'))
        advantage_module.add_module("dueling_A_{}".format(1), SlimFC(256, 512, activation_fn='relu'))
        advantage_module.add_module("A", SlimFC(512, 29, activation_fn=None))
        self.advantage_module = advantage_module

        value_module = nn.Sequential()
        value_module.add_module("dueling_V_{}".format(0), SlimFC(256, 256, activation_fn='relu'))
        value_module.add_module("dueling_V_{}".format(1), SlimFC(256, 512, activation_fn='relu'))
        value_module.add_module("V", SlimFC(512, 1, activation_fn=None))
        self.value_module = value_module

    def forward(self, inputs):

        # Preprocess the input
        torch_input_cpu = torch.from_numpy(inputs)
        if self._gpu_n >= 0:
            torch_input = torch_input_cpu.cuda(self._gpu_n)
        torch_input = torch_input.unsqueeze(0)

        self._features = torch_input.float().permute(0, 3, 1, 2)
        conv_out = self._convs(self._features)
        logits = conv_out.squeeze(3)
        model_out = logits.squeeze(2)

        action_scores = self.advantage_module(model_out)
        state_score = self.value_module(model_out)

        # Get the values
        mask = torch.ne(action_scores, float("-inf"))
        x_zeroed = torch.where(mask, action_scores, torch.zeros_like(action_scores))
        advantages_mean = torch.sum(x_zeroed, 1) / torch.sum(mask.float(), 1)
        advantages_centered = action_scores - torch.unsqueeze(advantages_mean, 1)
        values = state_score + advantages_centered

        return int(torch.argmax(values))

