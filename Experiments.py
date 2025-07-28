from Utilis import device_connect_simu,data_varying_loader_mnist,init_participants,\
    varying_cost,communication_cost_graph,server_with_device,comm_d_s
import numpy as np
import torch
import random
from collections import defaultdict
import json
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
from Model import FashionCNN
import argparse
import os



def round_1(tensor):
    for i in range(tensor.shape[0]):
        if tensor[i] > 1:
            tensor[i] = 1
    return tensor

def get_larger_1_idx(p_list,k,num_c):
    if np.where(np.array(p_list) > 1) != []:
        idx_list_larger_1 = np.where(np.array(p_list) > 1)[0].tolist()
    else:
        idx_list_larger_1 = []
    assert len(idx_list_larger_1) < k*num_c, "to many larger 1 p"
    return idx_list_larger_1

def get_flag(p_list):
    flag = sum([1 for e in p_list if e > 0 and e <1 ])
    return flag

def get_need_round_idx(p_list):
    if np.where(np.array(p_list) > 0) != []:
        idx_list_larger_0 = np.where(np.array(p_list) > 0)[0].tolist()
    else:
        idx_list_larger_0 = []
    if np.where(np.array(p_list) < 1):
        idx_list_smaller_1 = np.where(np.array(p_list) < 1)[0].tolist()
    else:
        idx_list_smaller_1 = []
    return list(set(idx_list_larger_0).intersection(set(idx_list_smaller_1)))

def get_sample(a,b):
    sample = np.random.rand(1)
    thresh = b/(a+b)
    if sample < thresh.numpy().tolist():
        return True
    else:
        return False

def dependent_rounding(p_list = [0.2,0.2,0.2,0.2,0.2],k = 2,num_c=5):
    larger_1_idx = get_larger_1_idx(p_list,k,num_c)
    for idx in larger_1_idx:
        p_list[idx] = 1
    flag = get_flag(p_list)
    count = 0
    while flag > 0:
        count += 1
        if flag == 1:
            idx = get_need_round_idx(p_list)[0]
            if p_list[idx] < 0.5:
                p_list[idx] = 0
            else:
                p_list[idx] = 1
            flag = get_flag(p_list)
        else:
            idx = get_need_round_idx(p_list)
            pi_idx,pj_idx = idx[0],idx[1]
            pi_value,pj_value =  p_list[pi_idx],p_list[pj_idx]
            a = min(1-pi_value,pj_value)
            b = min(pi_value,1-pj_value)
            if get_sample(a, b):
                pi_update,pj_update = pi_value + a, pj_value-a
            else:
                pi_update,pj_update = pi_value - b, pj_value + b
            p_list[pi_idx] = pi_update
            p_list[pj_idx] = pj_update
            flag = get_flag(p_list)
    # print(count)
    return np.where(np.array(p_list) == 1)[0].tolist()

def generate_topo(weight_dict,weight_in_candidate_idx_dict, num_c = 5, k = 2):
    topo = []
    p_list = []
    weight_vec = torch.hstack([v for k,v in weight_dict.items()])
    weight_vec = torch.div(weight_vec, torch.sum(weight_vec)) * (num_c * k )
    weight_vec = round_1(weight_vec)
    selected_idx = dependent_rounding(weight_vec,k,num_c)
    rlt = defaultdict(list)
    for idx in selected_idx:
        client_idx = idx // (num_c-1)
        ord_idx = idx % (num_c-1)
        i_par_selected_in = weight_in_candidate_idx_dict[client_idx][ord_idx]
        rlt[client_idx].append(i_par_selected_in)
    for i in range(num_c):
        in_client_list = rlt[i]
        in_client_list.append(i)
        topo.append(in_client_list)
        p_list.append(torch.Tensor(weight_vec[(i)*(num_c-1):(i+1)*(num_c-1)]))
    return topo, p_list

def device_upload_data(data,labels,ratio):
    total_data_num = data.shape[0]
    comm_data_idx = [i for i in range(total_data_num)]
    random.shuffle(comm_data_idx)
    return data[comm_data_idx[:int(total_data_num*ratio)]],labels[comm_data_idx[:int(total_data_num*ratio)]],ratio


def sample_train_data(sample_train_par, data_tot,label_tot,train_data,train_label,n_par,num_sampled = 50):
    num_tot = data_tot.shape[0]
    num_train = train_data.shape[0]
    if num_train < 50:
        if num_train != 0:
            chosen_idx = np.random.randint(0, num_tot, 50-num_train).tolist()
            sample_train_par[n_par] = [torch.vstack((train_data,data_tot[chosen_idx])),torch.vstack((train_label,label_tot[chosen_idx])),num_train]
        else:
            chosen_idx = np.random.randint(0, num_tot, num_sampled).tolist()
            sample_train_par[n_par] = [data_tot[chosen_idx], label_tot[chosen_idx], 0]
    else:
        chosen_idx = np.random.randint(0, num_train, num_sampled).tolist()
        sample_train_par[n_par] = [train_data[chosen_idx],train_label[chosen_idx],num_train]
    return sample_train_par

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

def local_train(error,train_loader,model,optimizer, sample_train_par, comm,len_train =1600, local_round = 1,\
                batch_size = 50, par_idx = None,ratio = None,device = None):
    batch_loss = 0
    frq = int((len_train/(batch_size* local_round)))
    start = comm % frq * local_round

    end = start  + local_round
    count = 0
    comm_data_list = []
    comm_label_list = []
    for images, labels in train_loader:
        if count < start or count > end or count == end:
                count += 1
                continue
            # Transfering images and labels to GPU if available
        else:
            comm_label_list.append(labels)
            comm_data_list.append(images)
            count += 1
    comm_data = torch.vstack(comm_data_list)
    comm_label = torch.vstack([e.unsqueeze(1) for e in comm_label_list])
    data,label,ratio = device_upload_data(comm_data,comm_label,ratio)
    data_num = data.shape[0]
    # get sample data for loss
    sample_train_par = sample_train_data(sample_train_par, comm_data,comm_label,data,label,par_idx)
    loss = model_train(error,data,label,model,optimizer,batch_size,device)
    if loss == -1:
        return torch.tensor(0),0,sample_train_par
    batch_loss += loss
    return batch_loss/data_num,ratio,sample_train_par

def get_sample_loss(train_sample,train_label,model,device,error):
    with torch.no_grad():
        model.eval()
        train_sample = train_sample.to(device)
        train_label = train_label.to(device)
        pred = model(train_sample)
        loss = error(pred, train_label.squeeze())
    return loss

def get_in_out_node(topology,num_node,sample_train_par,participants,device,error,beta):
    rlt = defaultdict(dict)
    for i in range(num_node):
        node_in = topology[i]
        # if i in node_in:
        #     node_in.remove(i)
        par_train_sample_train, par_train_sample_label = sample_train_par[i][0], sample_train_par[i][1]
        rlt[i]['in'] = node_in
        p = torch.zeros(num_node)
        k_loss_list = []
        k_train_num_list = []
        k_idx = []
        for k in range(num_node):
            if k in node_in:
                if True:
                    k_loss = get_sample_loss(par_train_sample_train, par_train_sample_label, participants[k],device,error)
                    k_loss_list.append(-torch.sqrt(torch.square(k_loss)/50))
                    k_train_num = sample_train_par[i][2]
                    k_train_num_list.append(k_train_num)
                    k_idx.append(k)

                # p[k] = 1/len(node_in)
        k_loss_list = torch.softmax(torch.Tensor(k_loss_list),0)
        k_train_num_list = torch.softmax(torch.Tensor(k_train_num_list),0)
        for k in range(len(k_idx)):
            neigh_idx = k_idx[k]
            p[neigh_idx] = beta*k_loss_list[k]+(1-beta)*k_train_num_list[k]
        rlt[i]['p'] = p
        out = []
        for j in range(num_node):
            if i!= j:
                if i in topology[j]:
                    out.append(j)
        rlt[i]['out'] = out
    return rlt

def weight_aggregate(global_model, node_in_out_dict,par_idx,participants ):
    # todo: push sum
    num_participants = len(node_in_out_dict)
    global_dict = global_model.state_dict().copy()
    new_par_weight = 0
    in_p = node_in_out_dict[par_idx]['p']
    in_node_idx = node_in_out_dict[par_idx]['in']
    # new_par_weight = torch.sum(torch.mul(weight,in_p))
    for k in global_dict.keys():
        shape_k = participants[0].state_dict()[k].shape
        p_reshape_size = [-1]
        p_reshape_size += [1 for i in range(len(list(shape_k)))]
        in_p = in_p.cuda()
        k_update = torch.mul(torch.stack([participants[i].state_dict()[k].float() for i in range(num_participants)], 0),
                  in_p.reshape(tuple(p_reshape_size))).sum(dim=0)
        global_dict[k] = k_update
    return global_dict, new_par_weight

def adaptive_topology_col_learning(num_participants,topology_decision, sample_train_par,participants,device,error,beta):
    rlt_participants = []
    node_in_out_dict = get_in_out_node(topology_decision,num_participants,sample_train_par,participants,device,error,beta)
    new_weight_list  = []
    for par_idx in range(num_participants):
        # neighbor_idx = topology_decision[par_idx]
        par_model_new, par_weight = weight_aggregate(participants[par_idx],node_in_out_dict,par_idx,participants)
        rlt_participants.append(par_model_new)
        # new_weight_list.append(par_weight)
    # update self
    for i in range(num_participants):
        participants[i].load_state_dict(rlt_participants[i])
    # return torch.tensor(new_weight_list)
    return node_in_out_dict

def get_local_cost_idx(idx,ratio_list,computational_cost,server_with_device,comm_d_s):
    return computational_cost[idx] * ratio_list[idx] + server_with_device[idx]*comm_d_s*ratio_list[idx]

def get_comm_s2s_cost(idx,comm_cost_table):
    return comm_cost_table[idx]

def get_total_cost(ratio_list,comm_cost_graph,node_in_out_dict,computational_cost,server_with_device,comm_d_s):
    total = 0
    num_par = comm_cost_graph.shape[0]
    comm_cost = 0
    for i in range(num_par):
        in_idx_list = node_in_out_dict[i]['in']
        if i in in_idx_list:
            in_idx_list.remove(i)
        comm_cost_table = comm_cost_graph[i]
        for in_idx in in_idx_list:
            total += get_comm_s2s_cost(in_idx,comm_cost_table)
            comm_cost += get_comm_s2s_cost(in_idx,comm_cost_table)
    try:
        total = total.item()
    except:
        print("0")
    for i in range(num_par):
        total += get_local_cost_idx(i,ratio_list,computational_cost,server_with_device,comm_d_s)
    return total,comm_cost.item()



def get_cost(ratio_list,par_idx,node_in_out_dict, comm_cost_graph,computational_cost,server_with_device,comm_d_s):
    in_idx_list = node_in_out_dict[par_idx]['in']
    comp_cost = []
    comm_cost_s = []
    comm_cost_d = []
    comm_cost_table = comm_cost_graph[par_idx]
    idx_key_list = []
    if par_idx in in_idx_list:
        in_idx_list.remove(par_idx)
    for i in range(len(in_idx_list)):
        in_dx = in_idx_list[i]
        comp_cost.append(computational_cost[in_dx]*ratio_list[in_dx])
        comm_cost_s.append(comm_cost_table[in_dx])
        comm_cost_d.append(server_with_device[in_dx]*comm_d_s*ratio_list[in_dx])
        idx_key_list.append((par_idx,in_dx))
    in_neigh_cost_list = []
    for i in range(len(in_idx_list)):
        in_neigh_cost_list.append(comp_cost[i]+comm_cost_s[i]+comm_cost_d[i])
    return in_neigh_cost_list,idx_key_list

def unbiased_estimation(a,p,flag):
    if flag:
        return torch.tensor(1-(1/p)*(1-a))
    else:
        return torch.tensor(1)


def test_participant(test,model,device,predictions_list,labels):
    outputs = model(test)
    predictions = torch.max(outputs, 1)[1].to(device)
    predictions_list.append(predictions)
    correct = (predictions == labels).sum()

    total = len(labels)

    accuracy = correct * 100 / total
    return accuracy

def print_rlt(loss, acc, num_p,comm):
    acc = [e.cpu() for e in acc]
    print("Communication Round: {}, Mean_test_accuracy:{}".format(comm, np.mean(acc)))
    for i in range(num_p):
        print("Communication Round: {}, Participant_{}, Loss: {}, Accuracy: {}%".format(comm, i, loss[i].cpu().item(), acc[i].cpu().item()))
    print("*"*20)
    return np.mean(acc)


def fl_exp(para):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(20)
    batch_size = para['batch_size']
    torch.manual_seed(4)
    random.seed(118)
    num_participants = para['num_participants']
    local_round = para['local_round']
    lr = para['lr']
    comm_d_s = para['comm_d_s']
    server_with_device = para['server_with_device']
    ratio_k = para['ratio_k']
    alpha = para['alpha']
    beta = para['beta']

    num_communication = 200

    device_simu_ratio = device_connect_simu(num_par=num_participants, num_comm=num_communication, p = para['device_p'])
    participants_train_loader, participants_test_loader, len_train = data_varying_loader_mnist(batch_size=batch_size,\
                                                                                               d_p = para['dir_p'],\
                                                                                               num_participants=para['num_participants'])
    participants, opts = init_participants(FashionCNN,num_participants = num_participants)

    # cost
    comm_cost_simu = communication_cost_graph(num_participants, num_comm= num_communication)

    # varying cost
    computational_cost_simu = varying_cost(num_participants,num_communication)
    print("Length of train:{}".format(len_train))

    weight_dict = {}


    for i in range(num_participants):
        weight_dict[i] = torch.ones(num_participants - 1)

    # in_neighbor candidate idx
    weight_in_candidate_idx_dict = {}

    for i in range(num_participants):
        weight_in_candidate_idx_dict[i] = np.array([j for j in range(num_participants) if i != j])

    # Lists for visualization of loss and accuracy

    accuracy_list = []
    mean_acc_list = []

    # Lists for knowing classwise accuracy
    predictions_list = []

    total_cost = defaultdict(list)

    participants_test_rlt = defaultdict(list)
    participants_train_loss = defaultdict(list)
    for i in range(num_participants):
        participants_test_rlt[i].append(0)
    error = nn.CrossEntropyLoss()

    sample_train_par = defaultdict(list)
    for comm in range(num_communication):
        computational_cost = computational_cost_simu[comm]
        comm_cost = comm_cost_simu[comm]
        # print(weight_dict)
        topology_decision, p_list = generate_topo(weight_dict, weight_in_candidate_idx_dict,num_c = num_participants,k = (num_participants-1)*ratio_k)
        # print(topology_decision)
        local_train_loss = []
        device_sample_ratio_list = []
        for i in range(num_participants):
            p_loss, device_sample_ratio, sample_train_par = local_train(error,participants_train_loader[i], participants[i],\
                                                                        opts[i], sample_train_par, comm=comm, \
                                                      len_train=len_train, batch_size=batch_size,
                                                      local_round=local_round, par_idx=i,
                                                      ratio=device_simu_ratio[comm][i],device = device)
            device_sample_ratio_list.append(device_sample_ratio)
            participants_train_loss[i].append(p_loss.item())
            local_train_loss.append(p_loss)
        node_in_out_dict = adaptive_topology_col_learning(num_participants, topology_decision, sample_train_par,\
                                                          participants,device,error,beta)
        # Testing the model
        test_rlt = []
        cost_1, comm_cost_1 = get_total_cost(device_sample_ratio_list, comm_cost, node_in_out_dict,computational_cost,server_with_device,comm_d_s)
        total_cost[0].append(cost_1)
        total_cost['comm'].append(comm_cost_1)

        par_utility_list = []
        in_par_cost_list = []
        in_par_cost_key = []
        for i in range(num_participants):
            for images, labels in participants_test_loader[i]:
                images, labels = images.to(device), labels.to(device)
                test = Variable(images.view(len(labels), 1, 28, 28))
                test_acc = test_participant(test, participants[i],device,predictions_list,labels)
                test_rlt.append(test_acc)
            accuracy_list.append(test_rlt)
            participants_test_rlt[i].append(test_acc.item())
            # utility of test accuracy
            par_utility_list.append((participants_test_rlt[i][-1] - participants_test_rlt[i][-2]))
            l1,l2 = get_cost(device_sample_ratio_list, i, node_in_out_dict, comm_cost,computational_cost,\
                                   server_with_device,comm_d_s)
            in_par_cost_list.extend(l1)
            in_par_cost_key.extend(l2)
        par_utility_list = torch.softmax(torch.Tensor(par_utility_list), 0).numpy().tolist()
        in_par_cost_list = torch.softmax(torch.Tensor(in_par_cost_list), 0).numpy().tolist()
        in_par_cost_list = [1-e for e in in_par_cost_list]
        for i in range(num_participants):
            candidate_in_par_idx_list = weight_in_candidate_idx_dict[i]
            weight_col = node_in_out_dict[i]['p']
            for j in range(num_participants - 1):
                in_par_idx = candidate_in_par_idx_list[j]
                if in_par_idx in node_in_out_dict[i]['in']:
                    for idx in range(len(node_in_out_dict[i]['in'])):
                        if in_par_idx == node_in_out_dict[i]['in'][idx]:
                            in_par_cost_idx = idx
                    cost_key = (i,in_par_idx)
                    cost_idx = [_ for _ in range(len(in_par_cost_list)) if in_par_cost_key[_] == cost_key][0]
                    true_gain = torch.tensor(alpha*par_utility_list[i]*weight_col[in_par_idx]+(1-alpha)*in_par_cost_list[cost_idx])
                    estimated = unbiased_estimation(true_gain, p_list[i][j], True)
                else:
                    estimated = unbiased_estimation(0, 0, False)
                weight_dict[i][j] *= torch.exp(estimated * lr)

        mean_acc_list.append(print_rlt(local_train_loss, test_rlt, num_participants, comm))
    print(torch.max(torch.tensor(accuracy_list), dim=0))
    participants_test_rlt['best'] = torch.Tensor(torch.max(torch.tensor(accuracy_list), dim=0)[0]).numpy().tolist()
    total_cost[1] = sum(total_cost[0])
    total_cost['comm_sum'] = sum(total_cost['comm'])
    print(total_cost['comm_sum'])
    return total_cost['comm_sum'], participants_test_rlt['best'], max(participants_test_rlt['best']),float(max(mean_acc_list))




if __name__ == "__main__":
    with open("Config/Fashion-MNIST.json", "r") as f:
        para = eval(f.read())
    para['server_with_device'] = server_with_device
    para['comm_d_s'] = comm_d_s
    rlt = defaultdict(dict)
    key = "experiments"
    rlt[key]['cost'],rlt[key]['client_performance'],rlt[key]['best_acc'],rlt[key]['mean'] = fl_exp(para)
    print(rlt[key]['best_acc'])
    with open("Log/rlt.json", 'w', encoding='utf-8') as f:
        json.dump(rlt, f, ensure_ascii=False, indent=4)

