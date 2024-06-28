# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" Style transfer network."""

from mindspore import nn, ops, Tensor

class BasicConv2d(nn.Cell):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, activation_fn=nn.ReLU(),
                 normalizer_fn=None, pad_mode='same', **kwargs):
        super(BasicConv2d, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              pad_mode=pad_mode, **kwargs)
        self.bn = normalizer_fn
        self.activation_fn = activation_fn
        self.pad = ops.MirrorPad(mode='REFLECT')
        self.paddings = Tensor([[0, 0], [0, 0], [padding, padding], [padding, padding]])
    def construct(self, x):
        x, normalizer_fn, params, order = x
        x = self.conv(x)
        if normalizer_fn:
            x = normalizer_fn((x, params, order))
        if self.bn:
            x = self.bn(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return (x, normalizer_fn, params, order + 1)

class Residual(nn.Cell):
    def __init__(self, channels, kernel_size):
        super(Residual, self).__init__()
        self.conv1 = BasicConv2d(channels, channels, kernel_size=kernel_size, stride=1)
        self.conv2 = BasicConv2d(channels, channels, kernel_size=kernel_size, stride=1, activation_fn=None)
    def construct(self, x):
        h_1 = self.conv1(x)
        h_2 = self.conv2(h_1)
        out1, _, _, _ = x
        out2, normalizer_fn, params, order = h_2
        return (out1 + out2, normalizer_fn, params, order)

class Upsampling(nn.Cell):
    def __init__(self, stride, size, kernel_size, in_channels, out_channels, activation_fn=nn.ReLU()):
        super().__init__()
        self.stride = stride
        self.resize = ops.ResizeNearestNeighbor([i*stride for i in size])
        self.conv = BasicConv2d(in_channels, out_channels, kernel_size=kernel_size, activation_fn=activation_fn)
    def construct(self, input_):
        x, normalizer_fn, params, order = input_
        _, _, height, width = x.shape
        resize = ops.ResizeNearestNeighbor([height * self.stride, width * self.stride])
        x = resize(x)
        x = self.conv((x, normalizer_fn, params, order))
        return x


class Transform(nn.Cell):
    def __init__(self, in_channels=3, alpha=1.0):
        super(Transform, self).__init__()
        self.contract = nn.SequentialCell([
            BasicConv2d(in_channels, int(alpha * 32), kernel_size=9, stride=1,
                        normalizer_fn=nn.BatchNorm2d(int(alpha * 32), eps=0.001)),
            BasicConv2d(int(alpha * 32), int(alpha * 64), kernel_size=3, stride=2,
                        normalizer_fn=nn.BatchNorm2d(int(alpha * 64), eps=0.001)),
            BasicConv2d(int(alpha * 64), int(alpha * 128), kernel_size=3, stride=2,
                        normalizer_fn=nn.BatchNorm2d(int(alpha * 128), eps=0.001))
        ])
        self.residual = nn.SequentialCell([
            Residual(int(alpha * 128), 3),
            Residual(int(alpha * 128), 3),
            Residual(int(alpha * 128), 3),
            Residual(int(alpha * 128), 3),
            Residual(int(alpha * 128), 3)
            ])
        self.expand = nn.SequentialCell([
            Upsampling(2, (32, 32), 3, int(alpha * 128), int(alpha * 64)),
            Upsampling(2, (64, 64), 3, int(alpha * 64), int(alpha * 32)),
            Upsampling(1, (128, 128), 9, int(alpha * 32), 3, activation_fn=nn.Sigmoid())
        ])
    def construct(self, x):
        x, normalizer_fn, style_params = x
        out = self.contract((x, None, None, 0))
        x, _, _, _ = out
        x = self.residual((x, normalizer_fn, style_params, 0))
        out = self.expand(x)
        x, _, _, _ = out
        return x

class ConditionalStyleNorm(nn.Cell):
    def __init__(self, style_params=None, activation_fn=None):
        super(ConditionalStyleNorm, self).__init__()
        self.style_params = style_params
        self.moments = nn.Moments(axis=(2, 3), keep_dims=True)
        self.activation_fn = activation_fn
        self.rsqrt = ops.Rsqrt()
        self.cast = ops.Cast()
    def get_style_parameters(self, style_params):
        """Gets style normalization parameters."""
        var = []
        for i in style_params.keys():
            var.append(style_params[i].expand_dims(2).expand_dims(3))
        return var
    def norm(self, x, mean, variance, style_parameters, variance_epsilon, order):
        """ Normalization function with specific parameters. """
        inv = self.rsqrt(variance + variance_epsilon)
        gamma = style_parameters[order*2+1]
        beta = style_parameters[order*2]
        if gamma is not None:
            inv *= gamma
        data1 = self.cast(inv, x.dtype)
        data2 = x * data1
        data3 = mean * inv
        if gamma is not None:
            data4 = beta - data3
        else:
            data4 = -data3
        data5 = data2 + data4
        return data5
    def construct(self, input_):
        x, style_params, order = input_
        mean, variance = self.moments(x)
        style_parameters = self.get_style_parameters(style_params)
        output = self.norm(x, mean, variance, style_parameters, 1e-5, order)
        if self.activation_fn:
            output = self.activation_fn(output)
        return output
