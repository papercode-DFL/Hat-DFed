import numpy as np
from collections import defaultdict
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from torch.autograd import Variable




server_with_device = {
    0: 30,
    1: 30,
    2: 30,
    3: 30,
    4: 30
}


computational_cost = {
    0: 1,
    1: 2,
    2: 2,
    3: 1,
    4: 1
}

comm_d_s = 0.4



class FashionDataset(Dataset):
    """User defined class to build a datset using Pytorch class Dataset."""

    def __init__(self, data, transform=None):
        """Method to initilaize variables."""
        self.fashion_MNIST = list(data.values)
        self.transform = transform

        label = []
        image = []

        for i in self.fashion_MNIST:
            # first column is of labels.
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        # Dimension of Images = 28 * 28 * 1. where height = width = 28 and color_channels = 1.
        self.images = np.asarray(image).reshape(-1, 28, 28, 1).astype('float32')

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)




def get_par_csv(par_idx, train_csv,test_csv, num_par):
    par_train_csv = [pd.DataFrame(train_csv.iloc[par_idx[i][0],:]) for i in range(num_par)]
    par_test_csv = [pd.DataFrame(test_csv) for i in range(num_par)]
    return par_train_csv, par_test_csv

def get_par_set(par_train_csv, par_test_csv,dataset = 'mnist'):
    if dataset == 'mnist':
        participants_train_set = [FashionDataset(csv, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])) for csv in
                            par_train_csv]
        participants_test_set = [FashionDataset(csv, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])) for csv in
                            par_test_csv]
    return participants_train_set, participants_test_set



def get_par_loader(participants_train_set, participants_test_set,batch_size):
    participants_train_loader = [DataLoader(pset, batch_size=batch_size,shuffle= True) for pset in participants_train_set]
    participants_test_loader = [DataLoader(pset,batch_size = len(pset),shuffle= True) for pset in participants_test_set]
    return participants_train_loader, participants_test_loader


def device_connect_simu(num_par = 5,num_comm = 200,p = 0.5,n_devic_map = server_with_device):
    comm_device_split_ratio = defaultdict(list)
    rlt = defaultdict(list)
    for i in range(num_par):
        n_device = n_devic_map[i]
        sample = np.random.binomial(n_device, p, num_comm).tolist()
        sample = [e/n_device for e in sample]
        comm_device_split_ratio[i] = sample
    for i in range(num_comm):
        temp = []
        for j in range(num_par):
            temp.append(comm_device_split_ratio[j][i])
        rlt[i] = temp
    return rlt



def dirichlet_split_noniid(train_labels, alpha, n_clients, seed = 121):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''

    # np.random.seed(seed)
    n_classes = train_labels.max()+1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha]*n_classes, n_clients)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    num_data = 800
    for i in range(n_clients):
        for j in range(n_classes):
            client_idcs[i].extend(np.random.choice(class_idcs[j],replace = False,size = int(num_data*label_distribution[i][j])).tolist())
    return [np.array(e) for e in client_idcs]


def get_fixed_computation(num_par):
    # np.random.seed(1)
    rlt = np.random.choice([40,60], replace=True, size=num_par).tolist()
    return {i: rlt[i] for i in range(len(rlt))}

def check_fre(d_r,test_csv, num_participants):
    count_d = []
    for i in range(num_participants):
        count = defaultdict(int)
        client_idx = d_r[i]
        label = test_csv.iloc[client_idx, 0].to_list()
        for e in label:
            count[e] += 1
        count = {k:v/len(label) for k,v in count.items()}
        count_d.append(count)
    return count_d

def data_varying_loader_mnist(train_file = "Data/Fashion-MNIST/fashion-mnist_train.csv", test_file = "Data/Fashion-MNIST/fashion-mnist_test.csv",batch_size = 20,num_participants = 5,d_p =0.3):
    train_csv = pd.read_csv(train_file)
    test_csv = pd.read_csv(test_file)
    train_label = np.array(train_csv['label'])
    participants_ids = dirichlet_split_noniid(train_label,d_p,num_participants)
    check_fre(participants_ids,train_csv,num_participants)
    train_test_min = min([e.shape for e in participants_ids])[0]
    test_idx = 0
    par_idx = {i:[participants_ids[i][:train_test_min].tolist(),test_idx] for i in range(num_participants)}

    # data shift
    par_train_csv, par_test_csv  = get_par_csv(par_idx,train_csv, test_csv, num_participants)


    # Transform data into Tensor that has a range from 0 to 1
    participants_train_set, participants_test_set = get_par_set(par_train_csv, par_test_csv)
    participants_train_loader, participants_test_loader = get_par_loader(participants_train_set, participants_test_set,batch_size)
    return participants_train_loader, participants_test_loader, len(participants_train_set[0])


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def varying_cost(num_par,num_comm,fixed = computational_cost):
    rlt = [fixed for _ in range(num_comm)]
    return rlt

def varying_comm_cost(num_par,num_comm):
    return np.ones((num_comm,num_par,num_par))

def init_participants(model,num_participants = 4, lr = 0.001):
    participants = [model().cuda() for _ in range(num_participants)]
    # global_model = model()
    opts = [torch.optim.Adam(p.parameters(), lr=lr) for p in participants]

    # global_model.train()
    for p in participants:
        p.train()
    return participants, opts

def communication_cost_graph(num_par,num_comm = 200):
    graph_rand = torch.randn([num_par,num_par])
    rlt = torch.where(graph_rand > 0,1,2)
    simu = varying_comm_cost(num_par,num_comm)
    return torch.multiply(torch.from_numpy(simu),rlt)

def model_train(error,images_tot,labels_tot,model,optimizer,batch_size,device):
    total = images_tot.shape[0]
    if total == 0:
        return -1
    round = int(total/batch_size)
    mod_r = int(total)%int(batch_size)
    for i in range(round):
        model.train()
        images, labels = images_tot[i*batch_size:(i+1)*batch_size].to(device), labels_tot[i*batch_size:(i+1)*batch_size].to(device)
        train = Variable(images.view(batch_size, 1, 28, 28))
        labels = Variable(labels)
        # Forward pass
        outputs = model(train)
        loss = error(outputs, labels.squeeze())

        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()

        # Propagating the error backward
        loss.backward()

        # Optimizing the parameters
        optimizer.step()
    if mod_r != 0:
        images, labels = images_tot[round*batch_size:].to(device), labels_tot[
            round*batch_size:].to(device)
        train = Variable(images.view(mod_r, 1, 28, 28))
        labels = Variable(labels)
        # Forward pass
        outputs = model(train)
        loss = error(outputs, labels.squeeze())

        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()

        # Propagating the error backward
        loss.backward()

        # Optimizing the parameters
        optimizer.step()

    return loss.sum()



