import torch
import torch.nn as nn
from .style_predict_t import StylePrediction, style_normalization_activations
from .transform_t import Transform, ConditionalStyleNorm
# from .conditional_style_norm import ConditionalStyleNorm

# class Ais(nn.Module):
#     def __init__(self, style_prediction_bottleneck=100):
#         super(Ais, self).__init__()
#         activation_names, activation_depths = style_normalization_activations()
#         self.style_predict = StylePrediction(activation_names, activation_depths, style_prediction_bottleneck=style_prediction_bottleneck)
#         self.transform = Transform(3)
#         self.norm = ConditionalStyleNorm()

#     def forward(self, x):
#         content, style = x
#         style_params, _ = self.style_predict(style)
#         # print('transform 인풋: ',(content.shape, self.norm.shape, style_params.shape))
#         stylized_images = self.transform((content, self.norm, style_params)) # (Tensor, ConditionalStyleNorm, dict)가 들어갔음 -> 텐서여야함
#         return stylized_images

class Ais(nn.Module):
    def __init__(self, style_prediction_bottleneck=100):
        super(Ais, self).__init__()
        activation_names, activation_depths = style_normalization_activations()
        self.style_predict = StylePrediction(activation_names, activation_depths,
                                             style_prediction_bottleneck=style_prediction_bottleneck)
        self.transform = Transform(3)
        self.norm = ConditionalStyleNorm()

    def forward(self, content, style):
        content = torch.tensor(content)
        style_params, _ = self.style_predict(style)
        stylized_images = self.transform((content, self.norm, style_params))
        print("AiS에서: {}".format(stylized_images)) # 얘가 (8,3,256,256)이 되어야할텐데 지금 (2,8,3,256,256)
        return stylized_images
