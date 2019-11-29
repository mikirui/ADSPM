import os
from models import WarpGenerator512 as Generator
from torch.backends import cudnn
from torchvision import transforms as T
from PIL import Image
import torch
import argparse
import yaml
import numpy as np

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default='configs/model512_celeba_smile.yaml')
parser.add_argument('--ckpt',
                    type=str,
                    default='Stage3-200000-G.ckpt')
parser.add_argument('--input_dir',
                    type=str,
                    default='samples')
parser.add_argument('--output_dir',
                    type=str,
                    default='output')
parser.add_argument("--gpu",
                    type=str,
                    default='0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
opts = get_config(args.config)
model = Generator(opts['g_conv_dim'], opts['c_dim'], opts['g_repeat_num'], repeat_num2=opts['g_repeat_num2'], coff=opts['coff'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=lambda storage, loc: storage))

# transform
transform = []
transform.append(T.Resize(512))
transform.append(T.ToTensor())
transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
transform = T.Compose(transform)

target_label = np.zeros((opts['c_dim'], ))
target_label[0] = 1    # let it smile
target_label = torch.FloatTensor(target_label).unsqueeze(0).to(device)
for img_name in sorted(os.listdir(args.input_dir)):
    print("processing %s" % (img_name))
    img_path = os.path.join(args.input_dir, img_name)
    dst_img_path = os.path.join(args.output_dir, img_name)
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0).to(device)
    res, _, _, _, _, _, _, _ = model(img, target_label, stage=2)
    concat = torch.cat((img, res), dim=3)
    concat = denorm(concat)
    concat = concat.data[0].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(concat)
    im.save(dst_img_path)

