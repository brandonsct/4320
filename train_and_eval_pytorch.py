import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy

import medmnist
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from medmnist import INFO, Evaluator
from models import ResNet18
from tensorboardX import SummaryWriter
from torchvision.models import resnet18
from tqdm import trange

def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, download, model_flag, resize, as_rgb, model_path, run):

    lr = 0.001
    gamma=0.1
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]

    info = INFO[data_flag]
    task = info['task']
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_ids[0])

    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu') 
    print("device is using!!!",device)
    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    print('==> Preparing data...')
   
    if resize:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    else:
        data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
     
    train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb)

    
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False)

    print('==> Building and training model...')
    
    
    if model_flag == 'resnet18':
        model =  resnet18(pretrained=False, num_classes=n_classes) if resize else ResNet18(in_channels=n_channels, num_classes=n_classes)
    else:
        raise NotImplementedError

    model = model.to(device)

    train_evaluator = medmnist.Evaluator(data_flag, 'train')
    val_evaluator = medmnist.Evaluator(data_flag, 'val')
    test_evaluator = medmnist.Evaluator(data_flag, 'test')
    
 

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = CE_loss

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)
        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run, output_root)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run, output_root)

        print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
              'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
              'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))

    if num_epochs == 0:
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    logs = ['loss', 'auc', 'acc', 'sen', 'spe', 'prec', 'rec']
    train_logs = ['train_'+log for log in logs]
    val_logs = ['val_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)
    
    writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

    best_auc = 0
    best_epoch = 0
    best_model = deepcopy(model)

    global iteration
    iteration = 0
    
    for epoch in trange(num_epochs):        
        train_loss = train(model, train_loader, task, criterion, optimizer, device, writer)
        
        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run)
        
        scheduler.step()
        
        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        for key, value in log_dict.items():
            writer.add_scalar(key, value, epoch)
            
        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)
            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)

    state = {
        'net': best_model.state_dict(),
    }

    path = os.path.join(output_root, 'best_model.pth')
    torch.save(state, path)

    train_metrics = test(best_model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root)
    val_metrics = test(best_model, val_evaluator, val_loader, task, criterion, device, run, output_root)
    test_metrics = test(best_model, test_evaluator, test_loader, task, criterion, device, run, output_root)

    train_log = 'train  auc: %.5f  acc: %.5f  sen: %.5f  spe: %.5f  prec: %.5f  rec: %.5f\n' % (train_metrics[1], train_metrics[2],train_metrics[3],train_metrics[4],train_metrics[5],train_metrics[6])
    val_log = 'val  auc: %.5f  acc: %.5f  sen: %.5f  spe: %.5f  prec: %.5f  rec: %.5f\n' % (val_metrics[1], val_metrics[2],val_metrics[3],val_metrics[4],val_metrics[5],val_metrics[6])
    test_log = 'test  auc: %.5f  acc: %.5f  sen: %.5f  spe: %.5f  prec: %.5f  rec: %.5f\n' % (test_metrics[1], test_metrics[2],test_metrics[3],test_metrics[4],test_metrics[5],test_metrics[6])

    log = '%s\n' % (data_flag) + train_log + val_log + test_log
    print(log)
            
    with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
        f.write(log)  
            
    writer.close()


def train(model, train_loader, task, criterion, optimizer, device, writer):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)
          
        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1

        loss.backward()
        optimizer.step()
    
    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss


def test(model, evaluator, data_loader, task, criterion, device, run, save_folder=None):

    model.eval()
    
    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))
            
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)
   

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)
  

        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, save_folder, run)
        
        # other Evaluation Metrics
        y_target = evaluator.labels
        y_prediction = np.argmax(y_score, 1)
        from sklearn.metrics import confusion_matrix  
        cm = confusion_matrix(y_target, y_prediction)
        
        # You need to write down the right codes here #
        print()
        print("This is cm!!!!!!!!!\n", cm)
        def confusion_matrix_for(cls, cm):
            TP = cm[cls, cls]
            FN = cm[cls].sum() - TP
            FP = cm[:, cls].sum() - TP
            TN = cm.sum() - TP - FN - FP
            return np.array([[TP, FN], [FP, TN]])
        TP, FN, FP, TN = [0,0,0,0]
        for cls in range(cm.shape[0]):
            print(f'[class {cls} vs others]')
            TP_this, FN_this, FP_this, TN_this = confusion_matrix_for(cls, cm).ravel()
            print(f'TP: {TP_this}, FN: {FN_this}, FP: {FP_this}, TN: {TN_this}')        
            TP += TP_this
            FN += FN_this
            FP += FP_this
            TN += TN_this
        print(f'\nConclude TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}')
        sen = TP/(TP+FN)
        spe = TN/(TN+FP)
        pre = TP/(TP+FP)
        rec = TP/(TP+FN)
        print(f'\nConclude sen: {sen}, spe: {spe}, pre: {pre}, rec: {rec}')
        ###############################################
        
        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc, sen, spe, pre, rec]

def CE_loss(outputs, targets):
    
    # You need to write down the right codes here #
    import torch.nn.functional as F
    def cross_entrophy_loss(target, predicted_Output):
        softmax_ouput = F.softmax(predicted_Output, dim=1)
        loss = -torch.log(softmax_ouput[torch.arange(predicted_Output.size(0)),target]).mean()
        return loss

    loss = cross_entrophy_loss(targets, outputs)
    
    # you cannot use this function directly, 
    # you can compare the correct loss with your results.
    # criterion = nn.CrossEntropyLoss() 
    # loss = criterion(outputs, targets)
    ###############################################
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST2D')

    parser.add_argument('--data_flag',
                        default='octmnist',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=10, # 100 ori->10
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--gpu_ids', 
                        default='-1',  # if gpu: 0  else cpu: -1
                        type=str)
    parser.add_argument('--batch_size',
                        default=8, # 128 ori ->8 
                        type=int)
    parser.add_argument('--download',
                        default=True)
    parser.add_argument('--resize',
                        help='resize images of size 28x28 to 224x224',
                        action="store_true")
    parser.add_argument('--as_rgb',
                        help='convert the grayscale image to RGB',
                        action="store_true")
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_flag',
                        default='resnet18',
                        help='choose backbone from resnet18, resnet50',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)


    args = parser.parse_args()
    data_flag = args.data_flag
    output_root = args.output_root
    num_epochs = args.num_epochs
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    download = args.download
    model_flag = args.model_flag
    resize = args.resize
    as_rgb = args.as_rgb
    model_path = args.model_path
    run = args.run
    
    main(data_flag, output_root, num_epochs, gpu_ids, batch_size, download, model_flag, resize, as_rgb, model_path, run)
