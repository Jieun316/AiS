import torch
import torch.nn as nn
from torchvision import models
from .inception_v3_t import InceptionV3Encoder

"""
style predict 네트워크
인셉션 V3를 인코더로 씀
"""
def style_normalization_activations(pre_name='transformer', post_name='StyleNorm', alpha=1.0):
    scope_names = [
        'residual/residual1/conv1', 'residual/residual1/conv2',
        'residual/residual2/conv1', 'residual/residual2/conv2',
        'residual/residual3/conv1', 'residual/residual3/conv2',
        'residual/residual4/conv1', 'residual/residual4/conv2',
        'residual/residual5/conv1', 'residual/residual5/conv2',
        'expand/conv1/conv', 'expand/conv2/conv', 'expand/conv3/conv'
    ]
    scope_names = ['{}/{}/{}'.format(pre_name, name, post_name) for name in scope_names]
    depths = [int(alpha * 128)] * 10 + [int(alpha * 64), int(alpha * 32), 3]
    return scope_names, depths

class StylePrediction(nn.Module):
    def __init__(self, activation_names, activation_depths, style_prediction_bottleneck=100):
        super(StylePrediction, self).__init__()
        self.encoder = InceptionV3Encoder()
        self.bottleneck = nn.Conv2d(768, style_prediction_bottleneck, kernel_size=1)
        self.activation_depths = activation_depths
        self.activation_names = activation_names
        self.beta = nn.ModuleList()
        self.gamma = nn.ModuleList()
        # self.squeeze = torch.squeeze((2,3))
        for i in activation_depths:
            self.beta.append(nn.Conv2d(style_prediction_bottleneck, i, kernel_size=1))
            self.gamma.append(nn.Conv2d(style_prediction_bottleneck, i, kernel_size=1))

    def forward(self, x):
        x = self.encoder(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = self.bottleneck(x)
        style_params = {}
        for i in range(len(self.activation_depths)):
            beta = self.beta[i](x)
            # print(f"style prediction에서: {beta.shape}")
            beta = torch.squeeze(beta,(2,3))
            style_params[self.activation_names[i] + '/beta'] = beta
            gamma = self.gamma[i](x)
            # print(f"style prediction에서: {gamma.shape}")
            gamma = torch.squeeze(gamma,(2,3))
            style_params[self.activation_names[i] + '/gamma'] = gamma
        return style_params, x

# Example usage
if __name__ == "__main__":
    activation_names, activation_depths = style_normalization_activations()
    model = StylePrediction(activation_names, activation_depths)

    # Example input image tensor (batch size, channels, height, width)
    example_input = torch.randn(1, 3, 299, 299)
    
    # Get the style parameters
    style_params, bottleneck_output = model(example_input)
    
    print("Style Parameters:", style_params)
    print("Bottleneck Output Shape:", bottleneck_output.shape)
