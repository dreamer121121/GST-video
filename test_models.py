import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from dataset import VideoDataSet
from models import TemporalModel
from transforms import *


# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--root_path', type = str, default = './',
                    help = 'root path to video dataset folders')
parser.add_argument('--dataset', type=str, choices=['kinectics', 'diving', 'somethingv2','somethingv1'])
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=8)#每段视频采集25帧，每帧通过边角剪裁和中心剪裁扩展成10帧，总共250帧最终取平均得到最终的分类。
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')
parser.add_argument('--type', type=str, default="GST",choices=['GST','R3D','S3D'],
                    help = 'type of temporal models, currently support GST,Res3D and S3D')
parser.add_argument('--alpha', type=int, default=4, help = 'spatial temporal split for output channels')
parser.add_argument('--beta', type=int, default=2, choices=[1,2], help = 'channel splits for input channels, 1 for GST-Large and 2 for GST')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size')


args = parser.parse_args()


if args.dataset == 'ucf101':
    num_class = 101
elif args.dataset == 'hmdb51':
    num_class = 51
elif args.dataset == 'kinetics':
    num_class = 400
elif args.dataset == 'somethingv1':
    num_class = 174
else:
    raise ValueError('Unknown dataset '+args.dataset)

net = TemporalModel(num_class, args.test_segments, model = args.type, backbone=args.arch,
						alpha = args.alpha, beta = args.beta,
						dropout = args.dropout)

import datasets_video
categories, train_list, val_list, root_path, prefix = datasets_video.return_dataset(args.dataset, args.root_path)
#
checkpoint = torch.load(args.checkpoint)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = checkpoint['state_dict']
net.load_state_dict(base_dict)

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])

elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

data_loader = torch.utils.data.DataLoader(
        VideoDataSet(root_path, val_list, num_segments=args.test_segments,
                   new_length=1,
                   image_tmpl=prefix,
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)



# if args.gpus is not None:
#     devices = [args.gpus[i] for i in range(args.workers)]
# else:
#     devices = list(range(args.workers))
#
#
net = net.cuda()
net.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
output = []

def eval_video(video_data):
    with torch.no_grad():
        i, data, label = video_data
        # num_crop = args.test_crops #test_crops = 1 ,此参数留着后面多crops时用
        input_var = torch.autograd.Variable(data).cuda() #volatile表示是否处于推理,此处若不加.cuda()则input_var会在cpu上
        rst = net(input_var)
        rst = rst.data.cpu().numpy().copy() #至关重要，将数据从GPU复制到CPU
        return i, rst, label[0]


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

for i, (data, label) in data_gen:
    #data.size() [1,150,224,224] 当设test_segments = 5
    if i >= max_num:
        break
    rst = eval_video((i, data, label)) #tuple #处理一个batch的视频，将batch_size设置为1，即处理一段视频
    output.append(rst[1:])
    cnt_time = time.time() - proc_start_time
    print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                    total_num,
                                                                    float(cnt_time) / (i+1)))
video_pred = torch.from_numpy(np.array([np.argmax(x[0]) for x in output]))

video_labels = torch.from_numpy(np.array([x[1].item() for x in output]))
print("video_pred",video_pred)
print("video_labels",video_labels)

cls_cnt = len(video_labels)
cls_hit = video_pred.eq(video_labels).sum()
print("cls_cnt:",cls_cnt)
print("cls_hit:",cls_hit)

# cf = confusion_matrix(np.array(video_labels), np.array(video_pred)).astype(float) #创建混淆矩阵。
# print("cf",cf)
#
# cls_cnt = cf.sum(axis=1) #总共的视频数量。
# print("cls_cnt:",cls_cnt)
# cls_hit = np.diag(cf) #正确预测的数量
# print("cls_hit:",cls_hit)

cls_acc = cls_hit / cls_cnt #准确率。

print(cls_acc)

print('Accuracy {:.02f}%'.format(cls_acc * 100))

# if args.save_scores is not None:
#
#     # reorder before saving
#     name_list = [x.strip().split()[0] for x in open(args.test_list)]
#
#     order_dict = {e:i for i, e in enumerate(sorted(name_list))}
#
#     reorder_output = [None] * len(output)
#     reorder_label = [None] * len(output)
#
#     for i in range(len(output)):
#         idx = order_dict[name_list[i]]
#         reorder_output[idx] = output[i]
#         reorder_label[idx] = video_labels[i]
#
#     np.savez(args.save_scores, scores=reorder_output, labels=reorder_label)
#
#
#
#
