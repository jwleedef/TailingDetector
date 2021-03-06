import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.autograd import Variable
import argparse

from simpleCNN.simpleCNN import Net
from simpleUnionCNN.simpleUnionCNN import UnionNet
from efficientnet import EfficientNet

def torchLoader(path):
    ret = torch.load(path)
    return ret

def infer(model, dataPath, is_cuda=False):
    model.eval()
    
    test_dataset = datasets.DatasetFolder(root=dataPath, loader=torchLoader, extensions='.pt')

    dataloaders = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                            shuffle=False, num_workers=4)
    batch_num = len(dataloaders)

    cnt, err = 0, 0
    for inputs, labels in dataloaders:
        if is_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)

        output = model(inputs)
        # print(f'\n[OUTPUT] : {output}')

        _, preds = torch.max(output, 1)
        cnt += 1
        # print(f'[PREDICTION] : {preds}')
        # print(f'[LABEL] : {labels}\n')

        if labels != preds:
            print(f'ERR - Label : {labels}, Pred : {preds}')
            print(f'[OUTPUT] : {output}\n')
            err += 1

    print(f'[Every Test Case] {cnt} [Error case] {err}\n[Accuracy] { (cnt - err) / cnt}\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='Dataset/test', help='path to test data directory')
    parser.add_argument('-c', '--checkfile', type=str, default='model.pt', help='')
    args = parser.parse_args()

    dataPath = args.data_path
    checkfile = args.checkfile

    is_cuda=False
    if torch.cuda.is_available():
        is_cuda = True
    
    model = torch.load(checkfile)
    if is_cuda:
        model.cuda()
    infer(model, dataPath, is_cuda)
