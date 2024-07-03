import torch
import torch.nn as nn
import torch.nn.functional as F

# from model.vgg_t import VGG
from torchvision import models
from .vgg_t import VGGencoder
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""
content loss, style loss 계산 후 weighted sum 해서 단일 Loss값을 출력해주는 클래스
train_t.py에서 불러옴
content_weights, style_weights는 train_t에서 젤 윗줄의 DEFAULT_WEIGHTS가 됨
bmm 연산은 바꾸면 달라져서 그냥 이미지 사이즈를 줄여주었음
"""
class TotalLoss(nn.Module):
    def __init__(self, in_channel, content_weights, style_weights):
        super(TotalLoss, self).__init__()
        self.encoder = VGGencoder(in_channel)
        self.content_weights = content_weights
        self.style_weights = style_weights

    def content_loss(self, content_end_points, stylized_end_points, content_weights):
        total_content_loss = 0
        content_loss_dict = {}
        # recude_mean = torch.std_mean()
        for name, weights in content_weights.items():
            name = str(name)
            # print('content loss에서: ', name, weights)
            print(f"content_end_points[name]: {content_end_points[name].shape}")

            print(f"styleized_end_points[name]: {stylized_end_points[name].shape}")
            loss = torch.mean((content_end_points[name] - stylized_end_points[name]) ** 2)
            weighted_loss = weights * loss
            print(f"c loss : {weighted_loss}")
            content_loss_dict['content_loss/' + name] = loss.item()
            content_loss_dict['weighted_content_loss/' + name] = weighted_loss.item()
            total_content_loss += weighted_loss
        content_loss_dict['total_content_loss'] = total_content_loss.item()

        return total_content_loss, content_loss_dict

    def style_loss(self, style_end_points, stylized_end_points, style_weights):
        total_style_loss = 0
        style_loss_dict = {}
        for name, weights in style_weights.items():
            G_stylized = self.get_matrix(stylized_end_points[name])
            G_style = self.get_matrix(style_end_points[name])
            # print("G_stylized, G_style: ", G_stylized, G_style)
            loss = torch.mean((G_stylized - G_style) ** 2)
            weighted_loss = weights * loss
            print(f"s loss : {weighted_loss}")
            style_loss_dict['style_loss/' + name] = loss.item()
            style_loss_dict['weighted_style_loss/' + name] = weighted_loss.item()
            total_style_loss += weighted_loss
        style_loss_dict['total_style_loss'] = total_style_loss.item()

        return total_style_loss, style_loss_dict

    def get_matrix(self, feature):
        feature = feature.detach().cpu()
        # print("gram matrix 인풋: ", feature.shape)
        batch_size, height, width = feature.size()
        # batch_size, height, width = batch_size.detach().cpu(), height.detach().cpu(), width.detach().cpu()
        denominator = float(height * width)
        # denominator_tensor = torch.tensor(denominator).to(feature.device)
        # denominator = torch.full((batch_size, batch_size), denominator, dtype=torch.float32)
        feature_map = feature.view((batch_size, height * width, 1)) #.cpu().numpy()
        # torch.cuda.empty_cache()
        # print(f"feature_map shape: {feature_map.shape}")
        matrix = torch.bmm(feature_map, feature_map.transpose(1, 2))
        # matrix = torch.matmul(feature_map.squeeze(-1), feature_map.squeeze(-1).transpose(0, 1))
        # print(f"matrix shape: {matrix.shape}")
        # print(f"denom이랑 matrix : {denominator.shape},{feature_map.shape}, {matrix.shape}")
        matrix /= denominator
        
        return matrix.to(device)

    def forward(self, content, style, stylized):
        print(len(content), len(style), len(stylized))
        content = content[0]
        style = style[0]
        stylized = stylized[0]
        # print(f"loss에서: {content.shape}, {style.shape}, {stylized.shape}")
        content_end_points = self.encoder(content)
        style_end_points = self.encoder(style)
        stylized_end_points = self.encoder(stylized)
        total_content_loss, _ = self.content_loss(content_end_points, stylized_end_points, self.content_weights)
        total_style_loss, _ = self.style_loss(style_end_points, stylized_end_points, self.style_weights)
        loss = total_content_loss + total_style_loss
        print(f"Total loss: {loss}")
        return loss
