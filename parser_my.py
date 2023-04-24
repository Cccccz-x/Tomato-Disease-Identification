import argparse
import torch

parser = argparse.ArgumentParser()


# TODO 常改动参数
parser.add_argument('--gpu', default=0, type=int) # gpu 卡号
parser.add_argument('--epochs', default=100, type=int) # 训练轮数
parser.add_argument('--lr', default=0.001, type=float) #learning rate 学习率
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--useGPU', default=True, type=bool) #是否使用GPU
parser.add_argument('--mean', default=[0.4179, 0.4867, 0.4081], type=list)
parser.add_argument('--std', default=[0.0542, 0.0461, 0.1009], type=list)

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")
args.device = device