from enum import Enum
from collections import defaultdict

import heapq
import numpy as np

import torch

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)
    
def softmax(x):
    '''Compute softmax values for a single vector.'''
    return np.exp(x) / np.sum(np.exp(x))

def row_softmax(output):
    '''Compute Row-Wise SoftMax given a matrix of logits'''
    new=np.array([softmax(i) for i in output])
    return new

def get_all_cost_sensitive(output,distances,classes):
    '''Re-Rank all predictions in the dataset using CRM'''
    
    num_classes=len(classes)
    C=[[0 for i in range(num_classes)] for j in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            C[i][j]=distances[(classes[i],classes[j])]

    final=np.dot(output,C)
    return -1*final

def get_topk(prediction,target,distances,classes,k=1):
    '''Computing hierarchical distance@k'''
    ind=heapq.nlargest(k, range(len(prediction)), prediction.take)
    # heapq.n
    scores=[]
    for i in ind:
        scores.append(distances[(classes[i],classes[target])])
    return scores

def get_metrics(args, output,target,distances,classes):

    # Apply CRM
    if args.crm==1:
        output=get_all_cost_sensitive(output,distances,classes)

    orig_top1=[]
    orig_mistake=[]
    orig_avg_1=[]
    orig_avg_5=[]
    orig_avg_20=[]

    all_conf = []
    cor_conf = []
    wrong_conf = []
    all_std = []
    cor_std = []
    wrong_std = []

    for i in range(len(output)):

        if output[i].argmax()==target[i]:
            cor_std.append(np.std(output[i]))
            orig_top1.append(1)
            if output[i].max() > 0.5:
                cor_conf.append(1)
                all_conf.append(1)
            else:
                cor_conf.append(0)
                all_conf.append(0)
        else:
            if output[i].max() > 0.5:
                wrong_conf.append(1)
                all_conf.append(1)
            else:
                wrong_conf.append(0)
                all_conf.append(0)

            orig_top1.append(0)
            wrong_std.append(np.std(output[i]))
            orig_mistake.append(distances[(classes[target[i]], classes[output[i].argmax()])])

        all_std.append(np.std(output[i]))

        orig_avg_1.extend(get_topk(output[i],target[i],distances,classes,1))

        orig_avg_5.append(get_topk(output[i],target[i],distances,classes,5))

        orig_avg_20.append(get_topk(output[i],target[i],distances,classes,20))
    
    all_conf = np.round(np.array(all_conf).mean() * 100, 2)
    cor_conf = np.round(np.array(cor_conf).mean() * 100, 2)
    wrong_conf = np.round(np.array(wrong_conf).mean() * 100, 2)
    all_std = np.round(np.array(all_std).mean() * 100, 2)
    cor_std = np.round(np.array(cor_std).mean() * 100, 2)
    wrong_std = np.round(np.array(wrong_std).mean() * 100, 2)
    
    print("-----------------------------------------")
    print(f"Avg. All Confident Predictions    : {all_conf:.2f}")
    print(f"Avg. Correct Confident Predictions: {cor_conf:.2f}")
    print(f"Avg. Wrong Confident Predictions  : {wrong_conf:.2f}")
    print(f"Avg. Std Predcitions              : {all_std:.2f}")
    print(f"Avg. Std Correct Predcitions      : {cor_std:.2f}")
    print(f"Avg. Std Wrong Predcitions        : {wrong_std:.2f}")
    print("-----------------------------------------")

    result = np.array([np.array(orig_top1).mean() * 100,np.array(orig_mistake).mean(),np.array(orig_avg_1).mean(),np.array(orig_avg_5).mean(),np.array(orig_avg_20).mean()])
    result = np.round(result, 2)
    
    return result

def get_mistakes(num_classes, distances, classes, test_output, test_target):
    mistakes = {k:0 for k in range(num_classes)}
    costs = []
    distance_freq = defaultdict(int)

    for i in range(test_output.shape[0]):
        prediction = test_output[i].argmax()
    
        if prediction != test_target[i]:
            cost = distances[(classes[prediction],classes[test_target[i]])]
            distance_freq[cost] += 1
            costs.append(cost)
            mistakes[test_target[i]] += 1

    count = 0
    for k in distance_freq:
        if k!=1:
            count += distance_freq[k]
    return mistakes,distance_freq,count

def post_hoc_adjustment(y, a, parent_child_mapping):
    bs = y.shape[0]
    for i in range(a.shape[1]):
        indices = torch.tensor(parent_child_mapping[i]).view(1, -1)
        y[torch.arange(bs)[:, None], indices] = y[torch.arange(bs)[:, None], indices] * a[torch.arange(bs), i][:, None]
    return y

def post_hoc_marginalize(y, a, parent_child_mapping):
    bs = y.shape[0]
    a_ = torch.zeros_like(a)
    for i in range(a.shape[1]):
        indices = torch.tensor(parent_child_mapping[i]).view(1, -1)
        a_[torch.arange(bs), i] += y[torch.arange(bs)[:, None], indices].mean(dim=-1) #.exp()
        y[torch.arange(bs)[:, None], indices] = y[torch.arange(bs)[:, None], indices] * a_[torch.arange(bs), i][:, None]
    return y