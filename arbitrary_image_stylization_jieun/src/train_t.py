import argparse
import os
from tqdm import tqdm
import torch
from torch import nn, optim
from torchvision import models
from torch.utils.data import DataLoader

from dataset.ais_dataset_t import create_dataset  # Ensure this is your custom dataset module
from model.ais_t import Ais  # Ensure these are your model definitions
from model.loss_t import TotalLoss  # Adjust the import as per your project structure
from model.inception_v3_t import InceptionV3Encoder

DEFAULT_CONTENT_WEIGHTS = {'vgg_16/conv3': 1}
DEFAULT_STYLE_WEIGHTS = {
    'vgg_16/conv1': 0.5e-3,
    'vgg_16/conv2': 0.5e-3,
    'vgg_16/conv3': 0.5e-3,
    'vgg_16/conv4': 0.5e-3
}
torch.autograd.set_detect_anomaly(True)
class WithLossCell(nn.Module):
    def __init__(self, network, loss_fn):
        super(WithLossCell, self).__init__()
        self.network = network
        self.loss_fn = loss_fn

    def forward(self, content, style):
        # print(f"loss cell에서: {content.shape}, {style.shape}")
        stylized = self.network(content, style)
        print(f"stylized: {stylized.shape}")
        loss = self.loss_fn(content, style, stylized)
        return loss

def main(args):
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    
    dataloader = create_dataset(args)
    # print(f"train에서: {args.content_weights}, {args.style_weights}")
    network = Ais(style_prediction_bottleneck=args.style_prediction_bottleneck).to(device) # style prediction network
    loss_fn = TotalLoss(3, args.content_weights, args.style_weights).to(device)  # loss 계산
    # vgg16 = models.vgg16(pretrained=True)# Load pretrained models
    # inception3 = models.inception_v3(pretrained=True) # 이것도 이상해서 지워주었음. InceptionV3Encoder에서 이미 pretrained를 불러와 주는데 이걸 왜 이렇게 했지

    network.style_predict.encoder = InceptionV3Encoder()
    # network.style_predict.encoder.load_state_dict(inception3.state_dict(), strict=False) # 이걸 해줄 필요가 없나..? pre trained 모델을 어떻게 쓰는지 잘 모르겠음
    # loss.load_state_dict(vgg16.state_dict(), strict=False) # TotalLoss는 forward에서 content, style, stylized를 받는데 이렇게 하면 안될것 같음
    
    # network.style_predict.encoder.load_state_dict(models.inception_v3(pretrained=True).state_dict(), strict=False)
    
    net_with_loss = WithLossCell(network, loss_fn).to(device)
    optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    step = 0
    with tqdm(total=args.max_step + 1) as pbar:
        for step in range(args.max_step + 1):
            for content, style in dataloader:
                content, style = content.to(device), style.to(device)
                
                optimizer.zero_grad()
                loss = net_with_loss(content, style)
                loss.backward()
                optimizer.step()
                
                if step % args.save_checkpoint_step == 0 and step != 0:
                    torch.save(network.state_dict(), f'{args.output}/model-{int(step/args.save_checkpoint_step)}.pth')
                
                pbar.set_description(f'loss: {loss.item():.4f}')
                pbar.update(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Arbitrary image stylization train')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_step', type=int, default=3, help='Number of total steps.')
    parser.add_argument('--content_path', type=str, default='../../../imagenet_mini/imagenet-mini/train/', help='Path of content image.')
    parser.add_argument('--style_path', type=str, default='../../../dtd/images/', help='Path of style image.')
    parser.add_argument('--shuffle', type=int, default=1, help='1 means True and 0 mean False')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--size', type=int, default=128, help='Image size for both content and style.')
    parser.add_argument('--content_weights', type=dict, default=DEFAULT_CONTENT_WEIGHTS, help='Weights for content loss.')
    parser.add_argument('--style_weights', type=dict, default=DEFAULT_STYLE_WEIGHTS, help='Weights for style loss.')
    parser.add_argument('--style_prediction_bottleneck', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--output', type=str, default='./result/', help='Directory to save checkpoint.')
    parser.add_argument('--save_checkpoint_step', type=int, default=1, help='Interval step of saving checkpoint.')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--device_target', type=str, default='GPU', choices=['Ascend', 'GPU', 'CPU'])
    parser.add_argument('--parallel', type=int, default=0, help='0==training on single card, 1==parallel training.')
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
