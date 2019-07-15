import time
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ..models import *
from ..utils import AverageMeter, calculate_accuracy, Logger, MyDataset
from visdom import Visdom

DatasetsList = ['CIFAR10', 'CIFAR100']

ModelList = {'AlexNet': AlexNet, 'alexnet': alexnet,
             'DenseNet': DenseNet, 'densenet121': densenet121, 'densenet169': densenet169,
             'densenet201': densenet201, 'densenet161': densenet161,
             'Inception3': Inception3, 'inception_v3': inception_v3,
             'ResNet': ResNet, 'resnet18': resnet18, 'resnet34': resnet34,
             'resnet50': resnet50, 'resnet101': resnet101, 'resnet152': resnet152,
             'SqueezeNet': SqueezeNet, 'squeezenet1_0': squeezenet1_0, 'squeezenet1_1': squeezenet1_1,
             'VGG': VGG, 'vgg11': vgg11, 'vgg11_bn': vgg11_bn, 'vgg13': vgg13, 'vgg13_bn': vgg13_bn,
             'vgg16': vgg16, 'vgg16_bn': vgg16_bn, 'vgg19_bn': vgg19_bn, 'vgg19': vgg19,
             'se_resnet18': se_resnet18, 'se_resnet34': se_resnet34, 'se_resnet50': se_resnet50,
             'se_resnet101': se_resnet101, 'se_resnet152': se_resnet152,
             'hr18_net': hr18_net,
             'mobilenetv3': mobilenetv3,
             'shufflenetv2': shufflenetv2}


class TrainPipline(object):
    def __init__(self, opt):
        self.root_path = opt['path']['root_path']
        self.result_path = os.path.join(self.root_path, opt['path']['result_path'])
        self.datasets_path = os.path.join(self.root_path, opt['path']['datasest_path'])

        self.n_classes = opt['model']['n_classes']
        self.momentum = opt['model']['momentum']
        self.weight_decay = opt['model']['weight_decay']
        self.nesterov = opt['model']['nesterov']

        self.n_epochs = opt['train']['n_epochs']
        self.batch_size = opt['train']['batch_size']
        self.learning_rate = opt['train']['learning_rate']
        self.n_threads = opt['train']['n_threads']
        self.checkpoint = opt['train']['checkpoint']

        self.no_cuda = opt['cuda']['no_cuda']
        self.model_name = ''
        self.model_ft = ''

        self.visdom_log_file = os.path.join(self.result_path, 'log_files', 'visdom.log')
        self.vis = Visdom(port=8097,
                          log_to_filename=self.visdom_log_file,
                          env='myTest_1')

        self.vis_loss_opts = {'xlabel': 'epoch',
                              'ylabel': 'loss',
                              'title': 'losses',
                              'legend': ['train_loss', 'val_loss']}

        self.vis_tpr_opts = {'xlabel': 'epoch',
                             'ylabel': 'tpr',
                             'title': 'val_tpr',
                             'legend': ['tpr@fpr10-2', 'tpr@fpr10-3', 'tpr@fpr10-4']}

        self.vis_epochloss_opts = {'xlabel': 'epoch',
                                   'ylabel': 'loss',
                                   'title': 'epoch_losses',
                                   'legend': ['train_loss', 'val_loss']}

    def datasets(self, data_name=None):
        assert data_name in DatasetsList

        if data_name == 'CIFAR10':
            training_data = datasets.CIFAR10(root='./modelzoo/datasets/', train=True, download=False,
                                             transform=transforms.Compose([
                                                       # transforms.RandomResizedCrop(224),
                                                       transforms.Pad(96),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.1307,), (0.3081,))]))

            val_data = datasets.CIFAR10(root='./modelzoo/datasets/', train=False, download=False,
                                        transform=transforms.Compose([
                                                  # transforms.RandomResizedCrop(224),
                                                  transforms.Pad(96),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))]))

        elif data_name == 'CIFAR100':
            training_data = datasets.CIFAR100(root='./modelzoo/datasets/', train=True, download=True,
                                              transform=transforms.Compose([
                                                        # transforms.RandomResizedCrop(224),
                                                        transforms.Pad(96),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.1307,), (0.3081,))]))

            val_data = datasets.CIFAR100(root='./modelzoo/datasets/', train=False, download=True,
                                         transform=transforms.Compose([
                                                   # transforms.RandomResizedCrop(224),
                                                   transforms.Pad(96),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.1307,), (0.3081,))]))
        else:
            train_txt_path = os.path.join(self.datasets_path, 'train.txt')
            val_txt_path = os.path.join(self.datasets_path, 'val.txt')

            my_transform = transforms.Compose([transforms.Resize(224),
                                               transforms.ToTensor()])

            training_data = MyDataset(train_txt_path, transform=my_transform)
            val_data = MyDataset(val_txt_path, transform=my_transform)

        return training_data, val_data

    def model(self, model_name='resnet18', model_path=None):
        assert model_name in ModelList
        self.model_name = model_name
        # model_ft = resnet18(pretrained=True)
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, 10)

        self.model_ft = ModelList[self.model_name](num_classes=self.n_classes)
        if model_path is not None:
            self.model_ft.load_state_dict(model_path)
        else:
            self.model_ft.apply(weights_init)

        return self.model_ft

    def train(self, training_data, val_data, model):
        # data init
        train_loader = DataLoader(training_data,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  # num_workers=self.n_threads,
                                  pin_memory=True)
        # result writer
        train_logger = Logger(os.path.join(self.result_path, self.model_name + '_train.log'),
                              ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(os.path.join(self.result_path, self.model_name + '_train_batch.log'),
                                    ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
        val_logger = Logger(os.path.join(self.result_path, self.model_name + '_test.log'),
                            ['time', 'loss', 'acc'])

        # optimizer init
        optimizer = optim.SGD(model.parameters(),
                              lr=self.learning_rate,
                              momentum=self.momentum,
                              weight_decay=self.weight_decay,
                              nesterov=self.nesterov)

        # loss init
        criterion = nn.CrossEntropyLoss()

        print(model)
        if not self.no_cuda:
            model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
        # start train
        for i in range(0, self.n_epochs + 1):
            self.train_epoch(i, train_loader, model, criterion, optimizer,
                             train_logger, train_batch_logger)
            self.validation(val_data, model, criterion, val_logger)

    def train_epoch(self, epoch, data_loader, model, criterion, optimizer,
                    epoch_logger, batch_logger):
        print('train at epoch {}'.format(epoch))
        # set model to train mode
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            if not self.no_cuda:
                model = model.cuda()
                inputs = inputs.cuda()
                targets = targets.cuda()

            # inputs = Variable(inputs)
            # targets = Variable(targets)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.data, inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'acc': accuracies.val,
                'lr': optimizer.param_groups[0]['lr']
            })

            self.vislog_batch(i, losses.val)

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                   epoch,
                   i + 1,
                   len(data_loader),
                   batch_time=batch_time,
                   data_time=data_time,
                   loss=losses,
                   acc=accuracies))

        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': optimizer.param_groups[0]['lr']
        })

        if epoch % self.checkpoint == 0:
            save_file_path = os.path.join(self.result_path, self.model_name+'save_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)

    def validation(self, val_data, model, criterion, val_logger):
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False,
            # num_workers=self.n_threads,
            pin_memory=True)

        model.eval()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        for i, (inputs, targets) in enumerate(val_loader):

            if not self.no_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.data, inputs.size(0))
            accuracies.update(acc, inputs.size(0))

        test_time = time.time() - end_time

        val_logger.log({'time': test_time,
                        'loss': losses.avg,
                        'acc': accuracies.avg})

        print('TestTime {test_time:.3f}\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
               test_time=test_time,
               loss=losses,
               acc=accuracies))

    def vislog_batch(self, batch_idx,loss):
        x_value = batch_idx
        y_value = loss
        self.vis.line([y_value], [x_value],
                      name='train_loss',
                      win='losses',
                      update='append')
        self.vis.line([2], [x_value],
                      name='test_loss',
                      win='losses',
                      update='append')
        self.vis.update_window_opts(win='losses', opts=self.vis_loss_opts)


def weights_init(m):
    # weight initialization
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        # nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
