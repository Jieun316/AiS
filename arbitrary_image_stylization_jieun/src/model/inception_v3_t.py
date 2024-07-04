import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights

"""
논문에 나온 구조대로 Mixed_6e 레이어까지 사용한 인셉션 네트워크
style_predict_t.py에서 style predict할 때 사용하는 인코더로 사용됨 -> style이 x로 들어오고, 레이어들을 거쳐서 인코딩됨
"""
class InceptionV3Encoder(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        # Load the pre-trained InceptionV3 model
        inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        
        # Extract layers up to Mixed_6e layer
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e
        )
        
        # Fully connected layers for style prediction
        # self.fc1 = nn.Linear(768, 100)
        # self.fc2 = nn.Linear(100, 768)
        
    def forward(self, x):
        # Extract features
        x = self.features(x)
        # print("Shape after Mixed_6e:", x.shape)  # Debugging statement
        
        # Compute mean across each activation channel of Mixed_6e layer
        # if len(x.shape) == 4:
        #     x = torch.mean(x, dim=(2, 3))
        # elif len(x.shape) == 2:
        #     x = x  # Already in the correct shape
        # else:
        #     raise ValueError(f"Unexpected tensor shape: {x.shape}")
        
        # print("Shape after mean:", x.shape)  # Debugging statement
        
        # Apply fully connected layers
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        return x

# Example usage
if __name__ == "__main__":
    encoder = InceptionV3Encoder()
    input_tensor = torch.randn(1, 3, 299, 299)  # Example input tensor
    output_tensor = encoder(input_tensor)
    print("Final output shape:", output_tensor.shape)  # Should be [1, 768]
