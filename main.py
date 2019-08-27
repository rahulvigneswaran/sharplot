
import argparse

import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from  models import lanet
import utils
import matplotlib.pyplot as plt



#endless loader
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

#logger
from datetime import datetime

#ANCHOR Logger
def logging(fname, text):
    #import os
    if not os.path.exists(fname):
        with open(fname, 'w'): pass
    f = open(fname,"a")
    

    out_text = "[%s]\t%s" %(str(datetime.now()),text)
    f.write(out_text+"\n")
    # print (out_text)
    f.close()

#ANCHOR Main
def main(args, ITE=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resample = True if args.prune_type=="resample" else False
    reinit = True if args.prune_type=="reinit" else False

#ANCHOR Data Loader

    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,transform=transform),
        batch_size=args.batch_size, shuffle=True, num_workers=0,drop_last=False)

    train_loader = cycle(train_loader)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transform),
        batch_size=args.batch_size, shuffle=False, num_workers=0,drop_last=True)

#ANCHOR Importing LeNet Model
    model = lanet.LeNet(mask=True).to(device)

#ANCHOR Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    torch.save({"state_dict": initial_state_dict}, os.getcwd()+"/saves/initial_state_dict_"+args.prune_type+".pth.tar")

#ANCHOR Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

#REVIEW 
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(f"saves/model_{args.prune_type}.pth.tar")
        start_iteration = checkpoint["iter"]
        model.load_state_dict(checkpoint["state_dict"])
        initial_state_dict = torch.load(f"{os.getcwd()}/saves/initial_state_dict_{args.prune_type}.pth.tar")["state_dict"]
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
#===

#ANCHOR Pruning

# NOTE First Pruning Iteration is of No Compression
    bestacc = 0.0
    best_accuracy = 0
    ITERATION = 25
    comp = np.zeros(ITERATION,float)
    bestacc = np.zeros(ITERATION,float)

    for _ite in range(start_iteration, ITERATION):
        if not _ite == 0:
            model.prune_by_percentile(resample=resample, reinit=reinit)
            if not reinit:
                utils.original_initialization(model, initial_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print(f"--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: {1.0*(0.8**(_ite))*100:.1f}---")

#ANCHOR Print the table of Nonzeros in each layer

        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1

        pbar = tqdm(range(args.end_iter))
        for iter_ in pbar:

#ANCHOR Frequency for Testing
            if iter_ % args.valid_freq == 0:
                accuracy = test(model, test_loader)

#ANCHOR Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    if _ite == 0:
                        #NOTE  Take a look at this incase if you are trying produce randomly intialized (--prune_type=reinit) pruned networks
                        torch.save({"state_dict": model.state_dict(), "iter": _ite}, f"saves/best_unpruned_model_{args.prune_type}.pth.tar")
                    else:
                        torch.save({"state_dict": model.state_dict(), "iter": _ite}, f"saves/best_pruned_{_ite}_model_{args.prune_type}.pth.tar")

                
#ANCHOR Accuracy Logging
                if reinit==True:
                    logging(f"{os.getcwd()}/log/{ITE}_reinit/train_{_ite}_{args.prune_type}.log", "[%06d]\tAccuracy = %04f " % (iter_, float(accuracy)))
                elif resample==True:
                    logging(f"{os.getcwd()}/log/{ITE}_resample/train_{_ite}_{args.prune_type}.log", "[%06d]\tAccuracy = %04f " % (iter_, float(accuracy)))
                else:
                    logging(f"{os.getcwd()}/log/{ITE}_normal/train_{_ite}_{args.prune_type}.log", "[%06d]\tAccuracy = %04f " % (iter_, float(accuracy)))

#ANCHOR Training
            loss = train(model, train_loader, optimizer)

#ANCHOR Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')
        
        
            
        bestacc[_ite]=best_accuracy
        best_accuracy = 0

    

#ANCHOR Dumping Values for Plotting

    comp.dump(f"dumps/{args.prune_type}_compression.dat")
    bestacc.dump(f"dumps/{args.prune_type}_bestaccuracy.dat")
    #mat2 = numpy.load("my_matrix.dat")

#ANCHOR Plotting
    a = np.arange(25)
    plt.plot(a, bestacc, c="blue", label="winning tickets") 
    plt.title("Test Accuracy vs Pruning Rate (mnist)") 
    plt.xlabel("pruning rate") 
    plt.ylabel("test accuracy") 
    plt.xticks(a, comp, rotation ="vertical") 
    plt.ylim(90,100)
    plt.legend() 
    plt.grid(color="gray") 

    #NOTE Adjust Image Quality Here
    plt.savefig(f"fig/{args.prune_type}_fig1.png", dpi=1200) 
    plt.close()

#ANCHOR Function for Training
def train(model, train_loader, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer.zero_grad()

    model.train()
    imgs, targets = next(train_loader)
    imgs, targets = imgs.to(device), targets.to(device)
    output = model(imgs)
    train_loss = F.nll_loss(output, targets)
    train_loss.backward()

#ANCHOR Freezing Pruned weights by making their gradients Zero
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        grad_tensor = p.grad.data.cpu().numpy()
        grad_tensor = np.where(tensor == 0, 0, grad_tensor)
        p.grad.data = torch.from_numpy(grad_tensor).to(device)
    optimizer.step()
    return train_loss.item()

#ANCHOR Function for Testing
def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


if __name__=="__main__":

#ANCHOR Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default= 1.2e-3, type=float)
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=20000, type=int)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--valid_freq", default=100, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="normal", type=str, help="normal | resample | reinit")
    parser.add_argument("--gpu", default="7", type=str)



    args = parser.parse_args()

    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

#ANCHOR Looping Entire process
    #for i in range(0, 5):
    main(args, ITE=1)


