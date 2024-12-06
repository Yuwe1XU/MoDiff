from torch.utils.data import TensorDataset, DataLoader
from data.data_generators import load_dataset
from utils.graph_utils import init_features, init_features_Bit, init_features_Time, graphs_to_tensor, graphs_to_MultiD_tensor, graphs_to_MultiD_tensor_rotate, calculate_degree_distribution, map_degree2class
import torch
import random
import numpy as np
from scipy.sparse.linalg import eigsh

def top_k_eigen(adjs_tensor, k=20):
    data_size, n, _ = adjs_tensor.shape
    print("Begin TopK")
    # Initialize tensors to store top k eigenvalues and eigenvectors
    
    # Convert back to torch tensor if needed
    # eigenvalues, eigenvectors = eigsh(adjs_tensor[0].cpu().numpy(), k=k, which = 'LM')  # 'LM' = Largest Magnitude

    top_eigenvalues = torch.zeros((data_size, k), dtype=torch.float32, device=adjs_tensor.device)
    top_eigenvectors = torch.zeros((data_size, n, k), dtype=torch.complex64, device=adjs_tensor.device)
    for i in range(data_size):
        la, u = torch.linalg.eigh(adjs_tensor[i])
        
        # Select the top k eigenvalues and corresponding eigenvectors
        top_indices = torch.argsort(la, descending=True)[:k]
        top_eigenvalues[i] = la[top_indices]
        top_eigenvectors[i] = u[:, top_indices]
        if i%100==0: print(f"Compute Top K Eigen: {i}/{data_size}")

    return top_eigenvalues, top_eigenvectors

def power_iteration(A, num_simulations: int):
    b_k = torch.rand(A.shape[1], dtype=A.dtype, device=A.device)
    
    for _ in range(num_simulations):
        b_k1 = torch.mv(A, b_k)
        b_k1_norm = torch.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    eigenvalue = torch.dot(b_k, torch.mv(A, b_k))
    
    return eigenvalue, b_k

def compute_A_top_k(A, k, num_simulations: int):
    n = A.shape[0]
    eigenvalues = torch.zeros(k, dtype=A.dtype, device=A.device)
    eigenvectors = torch.zeros((n, k), dtype=A.dtype, device=A.device)
    
    A_current = A.clone()
    
    for i in range(k):
        eigenvalue, eigenvector = power_iteration(A_current, num_simulations)
        eigenvalues[i] = eigenvalue
        eigenvectors[:, i] = eigenvector

        A_current = A_current - eigenvalue * torch.outer(eigenvector, eigenvector.conj())
        
    return eigenvalues, eigenvectors

def compute_top_k_eigen(adjs_tensor, k):
    data_size, n, _ = adjs_tensor.shape
    num_simulations = 30

    top_eigenvalues = torch.zeros((data_size, k), dtype=adjs_tensor.dtype, device=adjs_tensor.device)
    top_eigenvectors = torch.zeros((data_size, n, k), dtype=adjs_tensor.dtype, device=adjs_tensor.device)

    for i in range(data_size):
        eigenvalues, eigenvectors = compute_A_top_k(adjs_tensor[i], k, num_simulations)
        top_eigenvalues[i] = eigenvalues
        top_eigenvectors[i] = eigenvectors
    return top_eigenvalues, top_eigenvectors

def graphs_to_dataloader(config, graph_list):
    adjs_tensor = graphs_to_tensor(graph_list, config.data.max_node_num)
    x_tensor = init_features(config.data.init, adjs_tensor, config.data.max_feat_num)

    # la, u = torch.symeig(adjs_tensor, eigenvectors=True)
    # la = torch.diag_embed(la)

    train_ds = TensorDataset(x_tensor, adjs_tensor)
    train_dl = DataLoader(train_ds, batch_size=config.data.batch_size, shuffle=True)
    return train_dl

def graphs_to_dataloader2(config, graph_list):

    adjs_tensor = graphs_to_tensor(graph_list, config.data.max_node_num)
    x_tensor = init_features(config.data.init, adjs_tensor, config.data.max_feat_num)

    # la, u = torch.symeig(adjs_tensor, eigenvectors=True)
    la, u = torch.linalg.eigh(adjs_tensor)
    # la = torch.diag_embed(la)

    train_ds = TensorDataset(x_tensor, adjs_tensor, u, la)
    train_dl = DataLoader(train_ds, batch_size=config.data.batch_size, shuffle=True)
    return train_dl

def graphs_to_dataloader_D(config, graph_list):
    adjs_tensor = graphs_to_MultiD_tensor(graph_list, config.data.max_node_num)
    x_tensor = init_features(config.data.init, adjs_tensor, config.data.max_feat_num)

    # la, u = torch.symeig(adjs_tensor, eigenvectors=True)
    # la = torch.diag_embed(la)

    train_ds = TensorDataset(x_tensor, adjs_tensor)
    train_dl = DataLoader(train_ds, batch_size=config.data.batch_size, shuffle=True)
    return train_dl

def graphs_to_dataloader_D2(config, graph_list):
    adjs_tensor = graphs_to_MultiD_tensor(graph_list, config.data.max_node_num)
    x_tensor = init_features(config.data.init, adjs_tensor, config.data.max_feat_num)

    # la, u = torch.symeig(adjs_tensor, eigenvectors=True)
    # la, u = torch.linalg.eigh(adjs_tensor)
    la, u = top_k_eigen(adjs_tensor, config.data.spec_dim)
    # la = torch.diag_embed(la)

    train_ds = TensorDataset(x_tensor, adjs_tensor, u, la)
    train_dl = DataLoader(train_ds, batch_size=config.data.batch_size, shuffle=True)
    return train_dl

def graphs_to_dataloader_D2_comp(config, graph_lists, degree_map=None, time_map=None):
    adjs_tensor = graphs_to_MultiD_tensor_rotate(graph_lists, config.data.max_node_num)
    x_tensor = init_features(config.data.init, adjs_tensor, config.data.max_feat_num)

    # la, u = torch.symeig(adjs_tensor, eigenvectors=True)
    # la, u = torch.linalg.eigh(adjs_tensor)
    la, u = top_k_eigen(adjs_tensor, config.data.spec_dim)
    # la = torch.diag_embed(la)

    train_ds = TensorDataset(x_tensor, adjs_tensor, u, la)
    train_dl = DataLoader(train_ds, batch_size=config.data.batch_size, shuffle=True)
    return train_dl

def graphs_to_dataloader_D2_comp_T01(config, graph_lists_T0, graph_lists_T1):
    adjs_tensor_T0 = graphs_to_MultiD_tensor_rotate(graph_lists_T0, config.data.max_node_num)
    adjs_tensor_T1 = graphs_to_MultiD_tensor_rotate(graph_lists_T1, config.data.max_node_num)
    x_tensor = init_features(config.data.init, adjs_tensor_T0, config.data.max_feat_num)

    # la, u = torch.symeig(adjs_tensor, eigenvectors=True)
    # la, u = torch.linalg.eigh(adjs_tensor)
    # la = torch.diag_embed(la)
    data_size, n, _ = adjs_tensor_T1.shape
 
    la0, u0 = top_k_eigen(adjs_tensor_T0, config.data.spec_dim)
    la2 = torch.zeros((data_size, config.data.spec_dim), dtype=la0.dtype, device=adjs_tensor_T0.device)
    
    # la0, u0 = torch.linalg.eigh(adjs_tensor_T0[0])
    # la2 = torch.zeros((data_size, n), dtype=la0.dtype, device=adjs_tensor_T0.device)

    # u2 = torch.zeros((data_size, n, n), dtype=la0.dtype, device=adjs_tensor_T0.device)
    # value_shift_dict = {}
    # for i in range(data_size):
    #     la0, u0 = torch.linalg.eigh(adjs_tensor_T0[i])
    #     la1, u1 = torch.linalg.eigh(adjs_tensor_T1[i])
        
    for i in range(data_size):
        top_eigenvalues =  np.diag(u0[i].T @ adjs_tensor_T1[i] @ u0[i])
        la2[i] = torch.from_numpy(top_eigenvalues)
        if i%200==0: print(f"Compute Top K Eigen: {i}/{data_size}")
    train_ds_T0 = TensorDataset(x_tensor, adjs_tensor_T0, u0, la0)
    train_dl_T0 = DataLoader(train_ds_T0, batch_size=config.data.batch_size, shuffle=False)
    train_ds_T1 = TensorDataset(x_tensor, adjs_tensor_T1, u0, la2)
    train_dl_T1 = DataLoader(train_ds_T1, batch_size=config.data.batch_size, shuffle=False)

    return train_dl_T0, train_dl_T1

def dataloader(config, get_graph_list=False):
    graph_list = load_dataset(data_dir=config.data.dir, file_name=config.data.data)
    # print("graph_list:",graph_list)

    # This line can be deleted to fix train and test set
    # random.shuffle(graph_list)

    test_size = int(config.data.test_split * len(graph_list))

    train_graph_list, test_graph_list = graph_list[test_size:], graph_list[:test_size]
    if get_graph_list:
        return train_graph_list, test_graph_list

    return graphs_to_dataloader(config, train_graph_list), graphs_to_dataloader(config, test_graph_list)


def dataloader2(config, get_graph_list=False):
    graph_list = load_dataset(data_dir=config.data.dir, file_name=config.data.data)
    test_size = int(config.data.test_split * len(graph_list))
    train_graph_list, test_graph_list = graph_list[test_size:], graph_list[:test_size]
    if get_graph_list:
        return train_graph_list, test_graph_list

    return graphs_to_dataloader2(config, train_graph_list), graphs_to_dataloader(config, test_graph_list)


def dataloader3(config, get_graph_list=False):
    graph_list_G0 = load_dataset(data_dir=config.data.dir, file_name= config.data.file1)
    graph_list_G1 = load_dataset(data_dir=config.data.dir, file_name= config.data.file2)

    test_size = int(config.data.test_split * len(graph_list_G0))

    train_graph_list = graph_list_G0[test_size:]
    test_graph_list = graph_list_G0[:test_size]
    if get_graph_list:
        return train_graph_list, test_graph_list

    return graphs_to_dataloader(config, train_graph_list), graphs_to_dataloader(config, test_graph_list)

def dataloader4(config, get_graph_list=False):
    graph_list_G0 = load_dataset(data_dir=config.data.dir, file_name= config.data.file1)
    graph_list_G1 = load_dataset(data_dir=config.data.dir, file_name= config.data.file2)
    test_size = int(config.data.test_split * len(graph_list_G0))
    train_graph_list_G0, test_graph_list_G0 = graph_list_G0[test_size:], graph_list_G0[:test_size]
    train_graph_list_G1, test_graph_list_G1 = graph_list_G1[test_size:], graph_list_G1[:test_size]
    if get_graph_list:
        print("EXCEPTION")
        return "ERROR"
    
    return graphs_to_dataloader2(config, train_graph_list_G0), graphs_to_dataloader2(config, train_graph_list_G1), \
          graphs_to_dataloader(config, test_graph_list_G0), graphs_to_dataloader(config, test_graph_list_G1)



def dataloader_TD_train(config, get_graph_list=False):
    graph_list_G0 = load_dataset(data_dir=config.data.dir, file_name=config.data.file1)
    graph_list_G1 = load_dataset(data_dir=config.data.dir, file_name=config.data.file2)
    test_size = int(config.data.test_split * len(graph_list_G0))
    train_graph_list_G0, test_graph_list_G0 = graph_list_G0[test_size:], graph_list_G0[:test_size]
    train_graph_list_G1, test_graph_list_G1 = graph_list_G1[test_size:], graph_list_G1[:test_size]
    if get_graph_list:
        print("EXCEPTION")
        return "ERROR"
    
    return graphs_to_dataloader_D2(config, train_graph_list_G0[:300]), graphs_to_dataloader_D2(config, train_graph_list_G1[:300]), \
        graphs_to_dataloader_D2(config, test_graph_list_G0[:50]), graphs_to_dataloader_D2(config, test_graph_list_G1[:50])

def dataloader_TD_train_comp(config, get_graph_list=False):
    if get_graph_list:
        print("EXCEPTION")
        return "ERROR"
    
    file_name_first = [str(config.data.file1), str(config.data.file2)]
    file_name_last = ['V','R','T']
    train_graph_G0_list, train_graph_G1_list, test_graph_G0_list, test_graph_G1_list = [], [], [], []
    for file_name in file_name_last:
        config.data.file1 = file_name_first[0] + file_name
        config.data.file2 = file_name_first[1] + file_name


        graph_list_G0 = load_dataset(data_dir=config.data.dir, file_name=config.data.file1)
        graph_list_G1 = load_dataset(data_dir=config.data.dir, file_name=config.data.file2)
        test_size = int(config.data.test_split * len(graph_list_G0))
        train_graph_G0, test_graph_G0 = graph_list_G0[test_size:], graph_list_G0[:test_size]
        train_graph_G1, test_graph_G1 = graph_list_G1[test_size:], graph_list_G1[:test_size]
        # train_graph_G0, test_graph_G0 = graph_list_G0[:4], graph_list_G0[4:]
        # train_graph_G1, test_graph_G1 = graph_list_G1[:4], graph_list_G1[4:]
        train_graph_G0_list.append(train_graph_G0[:2])
        train_graph_G1_list.append(train_graph_G1[:2])
        test_graph_G0_list.append(test_graph_G0[:1])
        test_graph_G1_list.append(test_graph_G1[:1])

    # train_loader_G0 , train_loader_G1 = graphs_to_dataloader_D2_comp_T01(config, train_graph_G0_list, train_graph_G1_list)
    # test_loader_G0 , test_loader_G1 = graphs_to_dataloader_D2_comp_T01(config, test_graph_G0_list, test_graph_G1_list)
    train_loader_G0 , train_loader_G1 = graphs_to_dataloader_D2_comp(config, train_graph_G0_list), graphs_to_dataloader_D2_comp(config, train_graph_G1_list)
    test_loader_G0 , test_loader_G1 = graphs_to_dataloader_D2_comp(config, test_graph_G0_list), graphs_to_dataloader_D2_comp(config, test_graph_G1_list)
    return train_loader_G0, train_loader_G1, test_loader_G0 , test_loader_G1 

def dataloader_TD_test(config, get_graph_list=False):
    graph_list_G0 = load_dataset(data_dir=config.data.dir, file_name=config.data.file1)
    # graph_list_G1 = load_dataset(data_dir=config.data.dir, file_name=config.data.file2)

    test_size = int(config.data.test_split * len(graph_list_G0))
    # test_size = int(0.5 * len(graph_list_G0))

    train_graph_list = graph_list_G0[test_size:]
    test_graph_list = graph_list_G0[:test_size]
    if get_graph_list:
        return train_graph_list, test_graph_list
    
    return graphs_to_dataloader_D(config, train_graph_list), graphs_to_dataloader_D(config, test_graph_list)
