import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import numpy as np

from timm.utils import accuracy

from util.transform import val_transforms
from util.trees import load_distances
from util.helper import guo_ECE,MCE
from util.misc import Summary, AverageMeter, row_softmax, get_metrics, get_mistakes, post_hoc_adjustment

from datasets import iNaturalist, TieredImagenetH
from network import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(f'Using {n_gpu} GPUs!')
    
parser = argparse.ArgumentParser('Inference Code', add_help=False)
parser.add_argument('--dataset', default='tieredimagenet', type=str, choices=['inaturalist', 'tieredimagenet'])
parser.add_argument('--input_size', default=224, type=int)
parser.add_argument('--crm', default=0, type=int)
parser.add_argument('--post_hoc', default=0, type=int)
parser.add_argument('--data_dir', default='./datasets/dataset_hierarchy', type=str)
# parser.add_argument('--model_path', default='/home/kanishk/hierarchical_classification/HAF/out/inat/cross-entropy/species/seed1/checkpoint.pth.tar', type=str)
# parser.add_argument('--aux_path', default='/home/kanishk/hierarchical_classification/HAF/out/inat/cross-entropy/genus/seed1/checkpoint.pth.tar', type=str)
parser.add_argument('--model_path', default='/media/newhd/kanishk/HAF_ckpts/tieredimagenet/cross-entropy/fine-grained/seed0/checkpoint.pth.tar', type=str)
parser.add_argument('--aux_path', default='/media/newhd/kanishk/HAF_ckpts/tieredimagenet/cross-entropy/coarse-grained/seed0/checkpoint.pth.tar', type=str)
args = parser.parse_args()

arch = 'resnet18'
feature_dim = 512

if args.dataset == "inaturalist":
    num_classes = 1010
    aux_num_classes = 72
    valdir = "/media/newhd/inaturalist_2019/"
    dataset_name = "inaturalist19-224"

    aux_model = load_checkpoint(arch, aux_num_classes, args.aux_path)
    resnet = load_checkpoint(arch, num_classes, args.model_path)

    val_dataset = iNaturalist(root=valdir, mode="val", transform=val_transforms('inaturalist19-224', normalize=True, resize=224), taxonomy="species")
    
elif args.dataset == "tieredimagenet":
    num_classes = 608
    aux_num_classes = 201
    valdir = "/media/newhd/Imagenet2012/Imagenet-orig/"
    dataset_name = "tiered-imagenet-224"

    aux_model = load_checkpoint(arch, aux_num_classes, args.aux_path)
    resnet = load_checkpoint(arch, num_classes, args.model_path)

    val_dataset = TieredImagenetH(root=valdir, mode="val", transform=val_transforms('tiered-imagenet-224', normalize=True, resize=224), is_parent=False)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
)

distances = load_distances(dataset_name, 'ilsvrc', args.data_dir)

classes = val_dataset.classes
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
top10 = AverageMeter('Acc@10', ':6.2f', Summary.AVERAGE)

test_output = []
test_target = []

# define parent class mapping
parent_child_mapping = val_dataset.parent_to_child

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        images = batch['img'].cuda(non_blocking=True)
        target = batch['target'].cuda(non_blocking=True)

        output = resnet(images)
        
        if args.post_hoc:
            logit = aux_model(images)

            output = post_hoc_adjustment(F.softmax(output, dim=-1), F.softmax(logit, dim=-1), parent_child_mapping) 
            
            output = F.normalize(output, p=1, dim=-1)

        test_output.extend(output.cpu().tolist())
        test_target.extend(target.tolist())
        
        acc1, acc5, acc10 = accuracy(output, target, topk=(1, 5, 10))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))
        top10.update(acc10, images.size(0))
            
test_output = np.array(test_output)
test_target = np.array(test_target)

print("===================================")
print(f"top-1: {top1.avg.item():.2f}, top-5: {top5.avg.item():.2f}, top-10: {top10.avg.item():.2f}")

if args.post_hoc == 1:
    softmax_output = test_output
else:
    softmax_output=row_softmax(test_output)

model_ece = guo_ECE(softmax_output,test_target, bins=15)
model_mce = MCE(softmax_output,test_target, bins=15)
print(f"ECE: {model_ece:.2f}")
print(f"MCE: {model_mce:.2f}")
result = get_metrics(args, softmax_output, test_target, distances, classes)

print(f"Top-1 Accuracy          : {result[0]:.2f}")
print(f"Mistake Severity        : {result[1]:.2f}")
print(f"Hierarchical Distance@1 : {result[2]:.2f}")
print(f"Hierarchical Distance@5 : {result[3]:.2f}")
print(f"Hierarchical Distance@20: {result[4]:.2f}")

mistakes, distance_freq, count = get_mistakes(num_classes, distances, classes, test_output, test_target)
print(">= Level2 mistakes: ", count, "\% of >= mistakes: ", count/sum(list(mistakes.values())), "total mistakes: ", sum(list(mistakes.values())))