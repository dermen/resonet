#import statements
import logging
import os
import random
import time
from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython import embed
from matplotlib import pyplot as plt #changed pylab to pyplot
from torch.utils.data import random_split
from isashumod.dataLoader import CustomImageDataset  # import dataloader file
from torchvision.models import resnet50, resnet34, resnet18, vit_b_16, vit_b_32, ResNet50_Weights
from scipy.stats import pearsonr
import numpy
import re
#2527, 2463

def setup_logger(filename):
    format = '%(asctime)s >> %(message)s'
    logger = logging.getLogger('main')

    if (logger.hasHandlers()):
        logger.handlers.clear()

    handler = logging.FileHandler(filename)
    logger.addHandler(handler)
    logger.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(format))
    logger.addHandler(console)
    logger.setLevel(20)

def getInfoFolder(config):
    rootFolderName = config["rootDir"]
    if not os.path.exists(rootFolderName):
        os.makedirs(rootFolderName)
    folderName = os.path.join(rootFolderName, str(time.strftime('run_%Y-%m-%d_%H_%M_%S')))
    os.mkdir(folderName)

    return folderName

#Used to calculate the dimensions of an image after it passed through a pool or convolution layer
def calcLengOfConv(length , kern_size, stride_size = -1):
    if stride_size == -1:
        stride_size = kern_size
    return floor((length - kern_size)/stride_size + 1)

#Make the data loaders
def make_dataset(root):
    iiFile = "imageNameAndImage.hdf5"
    isFile = "imageNameAndSpots.csv"
    hd_filename = os.path.join(root, iiFile)
    cs_filename = os.path.join(root, isFile)
    dataset = CustomImageDataset(annotations_file=cs_filename, path_to_hdf5=hd_filename, transform=None, useSqrt=True)

    return dataset

def make_trainValLoaders(trainset, batch_size):
    # Split the training set into 50% training and 50% validation data
    test_abs = int(len(trainset) * 0.95)
    #generator = torch.Generator().manual_seed(2)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]#, generator=generator
        #trainset, [144, 144], generator=generator
    )
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size,
                                              shuffle=True)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size,
                                            shuffle=False)
    return trainloader, valloader

def make_testloader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True)

#Define the Convolutional Neural Network
class Net(nn.Module):
    def __init__(self, mpk=1, fema=3):
        super().__init__()

        #Implement the arch key like you did the preprocessing conditions.

        #Images have already gone through a 5*5 MaxPool2d.
        self.pool = nn.MaxPool2d(kernel_size=mpk, stride=mpk)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=fema, kernel_size=(5, 5), stride=1) #1 input chanels, 3 convolution layers, 5 x 5 convolution
        #self.pool2 = nn.MaxPool2d(1, 1)
        #self.conv2 = nn.Conv2d(1, 16, 5)
        #self.avg = nn.AvgPool2d((501, 488))
        width_of_x = calcLengOfConv(length=(calcLengOfConv(length=1263, kern_size=mpk, stride_size=mpk)), kern_size=5, stride_size=1)
        width_of_y = calcLengOfConv(length=(calcLengOfConv(length=1231, kern_size=mpk, stride_size=mpk)), kern_size=5, stride_size=1)

        self.fc1 = nn.Linear(fema*int(width_of_x*width_of_y), 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        #x.shape = 505 * 492
        #x = self.pool(x)
        x = F.relu(self.conv1(x))

        #x = F.relu(self.conv2(x))
        #x = self.pool2(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return x

class Rn(nn.Module):
    def __init__(self, num=18, mpk=1, fema=3, two_fc_mode=False):
        super().__init__()

        self.two_fc_mode = two_fc_mode

        if num==18:
            self.res = resnet18()
        if num==34:
            self.res = resnet34()
        if num == 50:
            self.res = resnet50()

        self.res.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)#nn.Conv2d(1, self.res.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc_1000_to_100 = nn.Linear(1000, 100)
        self.fc_100_to_1 = nn.Linear(100, 1)
        self.fc_1000_to_1 = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.res(x)
        x = F.relu(x)
        if self.two_fc_mode:
            x = self.fc_1000_to_100(x)
            x = F.relu(x)
            x = self.fc_100_to_1(x)
        else:
            x = self.fc_1000_to_1(x)
        return x

class Tr(nn.Module):
    def __init__(self, mpk=1, fema=3, num=16):
        super().__init__()
        if num == 16:
            self.tra = vit_b_16(image_size=832)#, hidden_dim=1)  #patch_size = 16, and 16 *52 = 832
        if num == 32:
            self.tra = vit_b_32(image_size=832)
        self.tra.conv_proj = torch.nn.Conv2d(1, 768, kernel_size=(num, num), stride=(num, num))

        self.fc1 = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.tra(x)
        x = F.relu(x)
        x = self.fc1(x)
        return x
def make_net(device, config):
    # Make the network and have it utilize the gpu

    net = config['arc'] #Rn() Tr(),
    # decide what resources the model will use
    if (torch.cuda.device_count() > 1) and config['useMultipleGPUs']:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        net = nn.DataParallel(net)
    if config['useCuda']:
        net.to(device, dtype=torch.float32)
    return net

def descend(trainloader, useCuda, device, optimizer, net, criterion, train_losses, train_accuracies, pearsons, err_margin=0.1, etimes=[], track_times=False):
    logger = logging.getLogger('main')
    train_loss = 0
    train_num_samp = 0
    t= 0
    total = 0
    correct = 0
    all_labs = []
    all_outs = []
    eb = time.time()
    num_batch = len(trainloader)
    for i_batch, data in enumerate(trainloader, 0):
        t = time.time()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        etime = time.time() - t
        if track_times:
            print('took %.4f seconds to separate the tuple "data"' % etime)

        # zero the parameter gradients
        ti = time.time()
        optimizer.zero_grad()
        etime = time.time() - ti
        if track_times:
            print('took %.4f seconds to zero the gradients' % etime)

        ti = time.time()
        outputs = net(inputs)  # forward
        #print(f'outputs: {outputs}')
        etime = time.time() - ti
        if track_times:
            print('took %.4f seconds to make a prediction' % etime)

        ti = time.time()
        loss = criterion(outputs, labels)  # compute the loss
        train_loss += loss.item()
        train_num_samp += len(labels)

        etime = time.time() - ti
        if track_times:
            print('took %.4f seconds to compute the loss' % etime)

        ti = time.time()
        loss.backward()  # propogate the error backwards
        etime = time.time() - ti
        if track_times:
            print('took %.4f seconds to propagate the loss backwards' % etime)

        ti = time.time()
        optimizer.step()
        etime = time.time() - ti
        if track_times:
            print('took %.4f seconds to adjust the parameters' % etime)

        #take the accuracy
        total += labels.size(0)
        correct += (abs(outputs - labels)/labels < err_margin).sum().item()

        #for the pearson cc
        all_labs += [l.item() for l in labels]
        all_outs += [o.item() for o in outputs]
        
        print(f"Finished batch {i_batch+1}/{num_batch}. Seen {total} examples.")

    cc = pearsonr(all_labs, all_outs)[0]

    logger.info(f'TRAIN LOSS $$$: {train_loss}')
    logger.info(f'TRAIN ACCURACIES $$$: {correct/total}')
    logger.info(f'TRAIN PEARSON $$$: {cc}')

    train_losses.append(train_loss/train_num_samp)
    train_accuracies.append(correct/total)
    etimes.append(time.time() - eb)
    pearsons.append(cc)

    return inputs.cpu().numpy()


def validate(valloader, device, net, criterion, val_losses, val_accuracies, pearsons, prefix, err_margin=0.1, inputs2=None):
    logger = logging.getLogger('main')
    val_loss = 0
    train_num_samp = 0

    total = 0
    correct = 0
    all_labs = []
    all_outs = []
    # ---
    with torch.no_grad():
        for i, data in enumerate(valloader, 0):
            # if useCuda:
            #     inputs, labels = data[0].to(device), data[1].to(device)
            # else:
            #     inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(inputs)

            #for the pearson cc
            all_labs += [l.item() for l in labels]
            all_outs += [o.item() for o in outputs]

            #for the loss
            loss_as_ten = criterion(outputs, labels)
            val_loss += loss_as_ten.item()  # compute the loss
            train_num_samp += len(labels)

            # take the accuracy
            total += labels.size(0)
            correct += (abs(outputs - labels)/labels < err_margin).sum().item()

    cc = pearsonr(all_labs, all_outs)[0]
    #b = torch.tensor(inputs2).to(device)

    val_accuracies.append(correct/total)
    pearsons.append(cc)

    if numpy.allclose(all_outs, 0):
        print('You failed')

    logger.info(f'{prefix} LOSS $$$: {val_loss}')
    logger.info(f'{prefix} ACCURACIES $$$: {correct/total}')
    logger.info(f'{prefix} PEARSON $$$: {cc}')

    val_losses.append(val_loss/train_num_samp)

def save_epoch(net, epoch, folderName):
    pathToModel = os.path.join(folderName, 'modelE'+str(epoch)+'.pth')
    torch.save(net.state_dict(), pathToModel)

def train_up_model(device, trainloader, valloader, unseenloader, etimes, train_losses, val_losses, unseen_losses,
                   train_accuracies, val_accuracies, unseen_accuracies,
                   train_pearsons, val_pearsons, unseen_pearsons, folderName, config):
    #Make the network and have it utilize the gpu
    net = make_net( device=device, config=config)
    criterion = nn.MSELoss(reduction='mean')
    if config['optim'] == 0:
        optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=config['mom'], weight_decay=config['wd']) #lr=0.00000000001, momentum=0.3
    elif config['optim'] == 1:
        optimizer = optim.Adam(net.parameters(), lr=config['lr'])
    else:
        raise Exception('optimizer not selected in  connfig dictionary')

    for i_epoch, epoch in enumerate(range(config['epochs'])):  # loop over the dataset multiple times
        print(f"Beginning Epoch {i_epoch+1} / {config['epochs']}")
        net.train()
        a = descend(trainloader=trainloader, useCuda=config['useCuda'], device=device, optimizer=optimizer, net=net,
                    criterion=criterion, train_losses=train_losses, train_accuracies=train_accuracies,
                    pearsons=train_pearsons, err_margin=config['err_margin'], etimes=etimes)
        net.eval()
        #validate on familiar experiments
        validate(valloader=valloader, device=device, net=net, criterion=criterion, val_losses=val_losses,
                 val_accuracies=val_accuracies, prefix='VAL', err_margin=config['err_margin'], pearsons=val_pearsons)
        #validate on unfamiliar experiments
        validate(valloader=unseenloader, device=device, net=net, criterion=criterion, val_losses=unseen_losses,
                 val_accuracies=unseen_accuracies, prefix='UNSEEN', err_margin=config['err_margin'],
                 pearsons=unseen_pearsons)
        if epoch % 10 == 0:
            save_epoch(net=net, epoch=epoch, folderName=folderName)


    print('Finished Training')
    return net

def test(testloader, device, net, config):
    # Test on the whole dataset
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    net.eval()
    with torch.no_grad():
        for data in testloader:
            if config['useCuda']:
                inputs, labels = data[0].to(device), data[1].to(device)
            else:
                inputs, labels = data
            t = time.time()
            outputs = net(inputs)
            print('time to predict: %.6f' % (time.time()-t))
            print('arrived at predictions')
            total += labels.size(0)
            correct += (abs(outputs - labels) / labels < config['err_margin']).sum().item()
    return correct/total
def save_run(net, train_losses, train_accuracies, train_pearsons, etimes, val_losses,
             val_accuracies, val_pearsons, final_accuracies, rootForTrain, rootForTest,
             folderName, config, timetorun):
    #Log the statistics of the run and save the model
    log_filename = os.path.join(folderName, 'metrics.log')
    pathToModel = os.path.join(folderName, 'modelFinal.pth')
    torch.save(net.state_dict(), pathToModel)

    logging.basicConfig(filename=log_filename, encoding='utf-8', level=logging.INFO, filemode='a')
    logging.info(f'Network is: {net}')
    logging.info(f'Config is: {config}')
    logging.info(f'rootForTrain: {rootForTrain}')
    logging.info(f'rootForTest: {rootForTest}')
    logging.info(f'Epoch run times: {etimes}')
    logging.info(f'Training losses:{train_losses}')
    logging.info(f'Training accuracies:{train_accuracies}')
    logging.info(f'Training pearson ccs: {train_pearsons}')
    logging.info(f'Validation losses losses: {val_losses}')
    logging.info(f'Validation accuracies:{val_accuracies}')
    logging.info(f'Validation pearson ccs: {val_pearsons}')
    logging.info(f'Final accuracies: {final_accuracies}')
    logging.info(f'Total runtime: {timetorun}')


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, train_pearsons, val_pearsons):
    # show the costs over time
    # x = range(len(training_losses))
    # y = training_losses
    # f, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.plot(x, y)
    #
    # a = range(len(test_losses))
    # ax2.plot(a, test_losses)
    # ax2.plot(a, contr_losses)
    fig, axes = plt.subplots(2, 2)

    axes[0, 0].plot(train_losses, label='train losses')
    axes[0, 0].plot(val_losses, label='val losses')

    axes[0, 1].plot(train_accuracies, label='train accuracies')
    axes[0, 1].plot(val_accuracies, label='val accuracies')

    axes[1, 0].plot(train_pearsons, label='train pearsons')
    axes[1, 0].plot(val_pearsons, label='val pearsons')

    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[1, 0].legend()

    plt.show()

def loadModel(pathToModel, net, device):
    net.load_state_dict(torch.load(pathToModel, map_location=torch.device('cpu')))
    net.to(device, dtype=torch.float32)

    return net

# def verifyLoaders():
#     config = {
#         "err_margin": 0.1,
#         "batch_size": 10,
#         "epochs": 500,
#         "useMultipleGPUs": False,
#         "useCuda": True,
#         "lm": False,
#         "arc": Rn(),
#         "lr": 1e-11,
#         "mom": 0.1
#     }
#
#     rootForTrain = "/mnt/tmpdata/data/isashu/smallerLoaders/firstSmallerTrainLoader"
#
#     trainloader, valloader = make_trainValLoaders(make_dataset(root=rootForTrain), batch_size=config['batch_size'])
#
#     net = make_net(device=device, config=config)
#     criterion = nn.MSELoss(reduction='mean')
#     #optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=config['mom'])  # lr=0.00000000001, momentum=0.3
#
#     sum = 0
#     i = 0
#     for data in valloader:
#         if i == 0:
#             inputs, labels = data[0].to(device), data[1].to(device)
#             outputs = net(inputs)
#             print(outputs)
#             print(labels)
#         i+=1
#         loss = criterion(outputs, labels)
#     embed()
#     i = 0
#     for data in valloader:
#         if i == 0:
#             inputs, labels = data[0].to(device), data[1].to(device)
#             outputs = net(inputs, train=False)
#             print(outputs)
#             print(labels)
#         i+=1
#
#     print(f'Sum of labels {sum}')
#
#
#     #validate(valloader=valloader, device=device, net=net, criterion=criterion, val_losses=val_losses)



def main(rootDir, loaderRoot, loaderUnseenRoot, num=18, lr=1e-7, optim=0, mom=0.99, epochs=30, gpu_num =0,
        two_fc_mode=False, wd=0.1):

    torch.cuda.empty_cache()

    t = time.time()
    config = {
        "err_margin": 0.1,
        "batch_size": 64,
        "epochs": epochs,
        "useMultipleGPUs": False,
        "useCuda": True,
        "lm": False,
        "arc": Rn(num=num, two_fc_mode=two_fc_mode), #Tr(num=num)
        "lr":  lr,
        "mom": mom,
        "optim": optim,
        "rootDir": rootDir,
        "wd": wd
    }

    rootForTrain = os.path.join( loaderRoot, "trainLoader")
    rootForVal = os.path.join(loaderUnseenRoot, "valLoader")
    rootForTest = os.path.join(loaderRoot, "testLoaders")

    rootsForTest = ["1.25ALoader",
                    "3.15ALoader",
                    "5.45ALoader"]
    rootsForTest = [os.path.join(rootForTest, dname) for dname in rootsForTest]
    
    for dirname in [rootForTrain, rootForVal, rootForTest] + rootsForTest:
        if not os.path.exists(dirname):
            raise OSError("%s does not exist!" % dirname)

    # the main output folder:
    folderName = getInfoFolder(config)

    log_filename = os.path.join(folderName, 'training.log')
    setup_logger(log_filename) #main could be replaced with anything
    logger = logging.getLogger('main')
    logger.info(f'train files in {rootForTrain}')
    logger.info(f'validation files in {rootForVal}')
    logger.info(f'test files in {rootForTest}')
    logger.info(f'num is {num}')
    logger.info(f'lr is {lr}')
    logger.info(f'optim is {optim}')
    logger.info(f'momentum is {mom}')
    logger.info(f'two_fc_mode is {two_fc_mode}')

    # use the gpu
    device = torch.device(('cuda:' + str(gpu_num)) if torch.cuda.is_available() else 'cpu')

    # trainloader = make_testloader(make_dataset(root=rootForTrain), batch_size=config['batch_size'])
    # valloader = make_testloader(make_dataset(root=rootForVal), batch_size=config['batch_size'])
    trainloader, valloader = make_trainValLoaders(make_dataset(root=rootForTrain), batch_size=config['batch_size'])
    unseenloader = make_testloader(make_dataset(root=rootForVal), batch_size=config['batch_size'])

    testloaders = []
    for rootForTest in rootsForTest:
        testloaders.append(make_testloader(make_dataset(root=rootForTest), batch_size=config['batch_size']))

    print(f'Trainloader length: {len(trainloader)}')
    etimes = []
    train_losses = []
    val_losses = []
    unseen_losses = []
    train_accuracies = []
    val_accuracies = []
    unseen_accuracies =[]
    train_pearsons = []
    val_pearsons = []
    unseen_pearsons = []
    if not config['lm']:
        net = train_up_model(
            device=device,
            trainloader=trainloader,
            valloader=valloader,
            unseenloader=unseenloader,
            etimes=etimes,
            train_losses=train_losses,
            val_losses=val_losses,
            unseen_losses=unseen_losses,
            train_accuracies=train_accuracies,
            val_accuracies=val_accuracies,
            unseen_accuracies=unseen_accuracies,
            train_pearsons=train_pearsons,
            val_pearsons=val_pearsons,
            unseen_pearsons=unseen_pearsons,
            folderName=folderName,
            config=config
        )
    else:
        net = loadModel("/mnt/tmpdata/data/isashu/runFolders/run_2023-08-STUFF.pth", device=device) #Current network must match loaded network
        if (torch.cuda.device_count() > 1) and config['useMultipleGPUS']:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            net = nn.DataParallel(net)

    final_accuracies = []
    for testloader in testloaders:
        accuracy = test(
            testloader=testloader,
            device=device,
            net=net,
            config=config
        )
        final_accuracies.append(accuracy)

    timetorun=time.time()-t

    save_run(
        net=net,
        train_losses=train_losses,
        etimes=etimes,
        val_losses=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        final_accuracies=final_accuracies,
        train_pearsons=train_pearsons,
        val_pearsons=val_pearsons,
        rootForTrain=rootForTrain,
        rootForTest=rootsForTest,
        folderName=folderName,
        config=config,
        timetorun=timetorun
    )
    print(f'Time to run is {timetorun}')
    # plot_metrics(train_losses=train_losses, val_losses=val_losses,
    #              train_accuracies=train_accuracies, val_accuracies=val_accuracies,
    #              train_pearsons = train_pearsons, val_pearsons = val_pearsons)
    print(f'Accuracies are {final_accuracies}')

    torch.cuda.empty_cache()

def get_num(fname):
    sr = re.search("[0-9]{5}", fname)
    assert sr is not None
    return int(fname[sr.start(): sr.end()])


def plot_trend(loader, net, device, err_margin, pre):
    fs = 30
    # logger = logging.getLogger('main')
    net.eval()
    with torch.no_grad():
        vals =[]
        act = []
        exps = []

        for data in loader:
            image, act_count, filename = data[0].to(device), data[1], data[2]
            pre_count = net(image)
            #exp = int(filename[-17:-12])

            vals += [d.item() for d in pre_count]
            act += [d.item() for d in act_count]

            exps += [get_num(d) for d in filename]
    cc = pearsonr(vals, act)[0]
    acc = (abs(numpy.array(vals) - numpy.array(act))/numpy.array(act) < err_margin).sum()/len(vals)

    fig, ax = plt.subplots()

    ax.tick_params(labelsize=fs)
    ax.scatter(exps, act, c='blue', marker='<')
    ax.scatter(exps, vals, c='red')
    plt.title("CC=%.2f, Accuracy=%.2f" % (cc, acc), fontsize=fs)
    ax.set_xlabel("Experiment Number", fontsize=fs)
    ax.set_ylabel("Spot Counts", fontsize=fs)
    plt.subplots_adjust(bottom=0.14)
    plt.show()


def plotGuesses(rootDir, loaderRoot, num=18, lr=1e-7, optim=0, mom=0.99, epochs=30, gpu_num =0,
        two_fc_mode=False, wd=0):
    t = time.time()
    config = {
        "err_margin": 0.2,
        "batch_size": 20,
        "epochs": epochs,
        "useMultipleGPUs": False,
        "useCuda": True,
        "lm": True,
        "arc": Rn(num=num, two_fc_mode=two_fc_mode),  # Tr(num=num)
        "lr": lr,
        "mom": mom,
        "optim": optim,
        "rootDir": rootDir,
        "wd": wd
    }
    rootForVal = os.path.join(loaderRoot, "valLoader")
    rootForTest = os.path.join(loaderRoot, "testLoaders")

    rootsForTest = ["1.25ALoader",
                    "3.15ALoader",
                    "5.45ALoader"]
    rootsForTest = [os.path.join(rootForTest, dname) for dname in rootsForTest]

    testloaders = []
    for rootForTest in rootsForTest:
        testloaders.append(make_testloader(make_dataset(root=rootForTest), batch_size=config['batch_size']))

    unseenloader = make_testloader(make_dataset(root=rootForVal), batch_size=config['batch_size'])

    net = loadModel("/mnt/tmpdata/data/isashu/thousandYearRun/run_2023-08-13_13_07_40/modelFinal.pth", config['arc'])  # Current network must match loaded network

    #
    # net.eval()
    # with torch.no_grad():
    #     vals = []
    #     act = []
    #     exps = []
    #
    #     for data in unseenloader:
    #         image, act_count, filename = data[0].to(device), data[1], data[2]
    #         pre_count = net(image)
    #         # exp = int(filename[-17:-12])
    #
    #         vals += [d.item() for d in pre_count]
    #         act += [d.item() for d in act_count]
    #
    #         exps += [get_num(d) for d in filename]
    # from scipy.stats import pearsonr
    # cc = pearsonr(vals, act)[0]
    # acc = (abs(numpy.array(vals) - numpy.array(act)) / numpy.array(act) < config['err_margin']).sum() / len(vals)
    #
    # print('cc is %f' % cc)
    # print('accuracy is %f' % acc)

    # folderName = rootDir
    # if not os.path.exists(folderName):
    #     os.makedirs(folderName)
    # log_filename = os.path.join(folderName, 'outputs.log')
    # setup_logger(log_filename)  # main could be replaced with anything
    # logger = logging.getLogger('main')
    # logger.info(f'test files in {rootForTest}')

    i = 0
    for loader in testloaders:
        plot_trend(loader, net, device, pre=rootsForTest[i][:4], err_margin=config['err_margin'])
        i+=1
    
if __name__ == "__main__":
    print('Hi')

    rootDir='/mnt/tmpdata/data/isashu/garbage'
    loaderRoot = "/mnt/tmpdata/data/isashu/newLoaders/threeMax/bigLoaders/"
    plotGuesses(rootDir, loaderRoot, num=18, gpu_num=0)

    #main(rootDir, loaderRoot, num=18)
    #spotFinder.main(outdir, loaderRoot, gpu_num=gpu_num, epochs=num_ep, **params)

    #weight decay is still not implemented