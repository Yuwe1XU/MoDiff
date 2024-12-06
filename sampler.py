import os
import time
import pickle
import math
import torch

from utils.logger import Logger, set_log, start_log, train_log, sample_log, check_log
from utils.loader import load_ckpt, load_data, load_data_G1, load_data_TD_test, load_seed, load_device, load_model_from_ckpt, \
    load_ema_from_ckpt, load_sampling_fn, load_eval_settings, load_sampling_fn2, load_sampling_fn4Di, load_sampling_fn4Di_spec
from utils.graph_utils import adjs_to_graphs,adjsWnodes_to_graphs,  init_flags, quantize,quantize_MultiD, quantize_DegreeBound, init_flags2, init_flags2_G01,\
    init_flags2_wnodes, mask_adjs, init_flags3,  combine_graphs, count_motifs, init_flags2_wnodes_4Comp, init_flags2_wnodes_woMotif,\
    quantize_DegreeBound_4Comp, quantize_DegreeBound_woMotif
from utils.plot import save_graph_list, plot_graphs_list, plot_graphs_list_DiT
from evaluation.stats import eval_graph_list
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx
from moses.metrics.metrics import get_all_metrics
import numpy as np
import networkx as nx



class Sampler_G_DiT(object):
    def __init__(self, config):
        super(Sampler_G_DiT, self).__init__()

        self.config = config
        self.device = [1] #load_device()

    def evaluation_ByCompound(self, ckpt, file_name_first):
        # -------- Load checkpoint --------
        self.config.ckpt = ckpt
        self.ckpt_dict = load_ckpt(self.config, self.device)
        self.configt = self.ckpt_dict['config']
        if not "type" in self.configt:
            print("self.configt has not type setting")
            self.configt.type = self.config.type
        load_seed(self.configt.seed)

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.ckpt}-sample"
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')

        if not check_log(self.log_folder_name, self.log_name):
            logger.log(f'{self.log_name}')
            start_log(logger, self.configt)
            train_log(logger, self.configt)
        sample_log(logger, self.config)

        # -------- Load models --------
        self.model_x = load_model_from_ckpt(self.ckpt_dict['params_x'], self.ckpt_dict['x_state_dict'], self.device)
        self.model_adj = load_model_from_ckpt(self.ckpt_dict['params_adj'], self.ckpt_dict['adj_state_dict'], self.device)

        if self.config.sample.use_ema:
            self.ema_x = load_ema_from_ckpt(self.model_x, self.ckpt_dict['ema_x'], self.configt.train.ema)
            self.ema_adj = load_ema_from_ckpt(self.model_adj, self.ckpt_dict['ema_adj'], self.configt.train.ema)

            self.ema_x.copy_to(self.model_x.parameters())
            self.ema_adj.copy_to(self.model_adj.parameters())

        self.sampling_fn2 = load_sampling_fn4Di_spec(self.configt, self.config.sampler, self.config.sample, self.device)

        logger.log(f'GEN SEED: {self.config.sample.seed}')
        load_seed(self.config.sample.seed)
        
        train_graph_lists, test_graph_lists = [], []
        for file_name_last in ['V','R','T']:
            self.configt.data.file1 = file_name_first + file_name_last
            train_graph_list, test_graph_list = load_data_TD_test(self.configt, get_graph_list=True)
            train_graph_lists.append(train_graph_list)
            test_graph_lists.append(test_graph_list)

        num_sampling_rounds = math.ceil(len(test_graph_list) / self.configt.data.batch_size)
        gen_adj_list = []
        train_node_list = []
        # -------- Generate samples --------
        for r in range(num_sampling_rounds):
            t_start = time.time()
            self.init_flags, train_hermitian_tensor, train_nodes_array = init_flags2_wnodes_4Comp(train_graph_lists, self.configt, r)
            self.init_flags = self.init_flags.to(self.device[0])
            train_hermitian_tensor = train_hermitian_tensor.to(self.device[0])
            x, adj, _ = self.sampling_fn2(self.model_x, self.model_adj, self.init_flags, train_hermitian_tensor, self.configt.data.spec_dim) #

            adj = mask_adjs(adj, self.init_flags)
            logger.log(f"Round {r} : {time.time() - t_start:.2f}s")
            print(f"Round {r} : {time.time() - t_start:.2f}s")
            adj_np = adj.cpu().numpy()
            
            if r == num_sampling_rounds-1:
                adj_np = adj_np[num_sampling_rounds*self.configt.data.batch_size - len(test_graph_list) : ]
                train_nodes_array = train_nodes_array[num_sampling_rounds*self.configt.data.batch_size - len(test_graph_list) : ]

            gen_adj_list.extend([adj_np[i] for i in range(adj_np.shape[0])])
            train_node_list.extend([set(train_nodes_array[i]) for i in range(train_nodes_array.shape[0])])
            
        assert len(gen_adj_list)==len(train_node_list)
        
        lower_lim = 0
        upper_lim = 100
        thres1 = 0.34
        # thres2 = 0.14
        samples_int = quantize_DegreeBound_4Comp(gen_adj_list, train_node_list, thres1, thres1, lower_lim, upper_lim)
        gen_graph_list = adjsWnodes_to_graphs(samples_int, train_node_list, True)

        assert len(gen_graph_list)==len(test_graph_list)
        save_graph_list(self.log_folder_name , f"{self.config.ckpt[:10]}_{file_name_first[-10:-7]}_{str(thres1)[-2:]}", gen_graph_list)
        print("Evaluation By Compund Finish")

        # -------- Load checkpoint --------
        self.config.ckpt = ckpt
        self.ckpt_dict = load_ckpt(self.config, self.device)
        self.configt = self.ckpt_dict['config']
        if not "type" in self.configt:
            print("self.configt has not type setting")
            self.configt.type = self.config.type
        load_seed(self.configt.seed)

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.ckpt}-sample"
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')

        if not check_log(self.log_folder_name, self.log_name):
            logger.log(f'{self.log_name}')
            start_log(logger, self.configt)
            train_log(logger, self.configt)
        sample_log(logger, self.config)

        # -------- Load models --------
        self.model_x = load_model_from_ckpt(self.ckpt_dict['params_x'], self.ckpt_dict['x_state_dict'], self.device)
        self.model_adj = load_model_from_ckpt(self.ckpt_dict['params_adj'], self.ckpt_dict['adj_state_dict'], self.device)

        if self.config.sample.use_ema:
            self.ema_x = load_ema_from_ckpt(self.model_x, self.ckpt_dict['ema_x'], self.configt.train.ema)
            self.ema_adj = load_ema_from_ckpt(self.model_adj, self.ckpt_dict['ema_adj'], self.configt.train.ema)

            self.ema_x.copy_to(self.model_x.parameters())
            self.ema_adj.copy_to(self.model_adj.parameters())

        self.sampling_fn2 = load_sampling_fn4Di_spec(self.configt, self.config.sampler, self.config.sample, self.device)

        logger.log(f'GEN SEED: {self.config.sample.seed}')
        load_seed(self.config.sample.seed)
        
        train_graph_lists, test_graph_lists = [], []
        for file_name_last in ['V','R','T']:
            self.configt.data.file1 = file_name_first + file_name_last
            train_graph_list, test_graph_list = load_data_TD_test(self.configt, get_graph_list=True)
            train_graph_lists.append(train_graph_list)
            test_graph_lists.append(test_graph_list)

        num_sampling_rounds = math.ceil(len(test_graph_list) / self.configt.data.batch_size)
        gen_adj_list = []
        train_node_list = []
        # -------- Generate samples --------
        for r in range(num_sampling_rounds):
            t_start = time.time()
            self.init_flags, train_hermitian_tensor, train_nodes_array = init_flags2_wnodes_woMotif(train_graph_lists, self.configt, r)
            self.init_flags = self.init_flags.to(self.device[0])
            train_hermitian_tensor = train_hermitian_tensor.to(self.device[0])
            x, adj, _ = self.sampling_fn2(self.model_x, self.model_adj, self.init_flags, train_hermitian_tensor, self.configt.data.spec_dim) #

            adj = mask_adjs(adj, self.init_flags)
            logger.log(f"Round {r} : {time.time() - t_start:.2f}s")
            print(f"Round {r} : {time.time() - t_start:.2f}s")
            adj_np = adj.cpu().numpy()
            
            if r == num_sampling_rounds-1:
                adj_np = adj_np[num_sampling_rounds*self.configt.data.batch_size - len(test_graph_list) : ]
                train_nodes_array = train_nodes_array[num_sampling_rounds*self.configt.data.batch_size - len(test_graph_list) : ]

            gen_adj_list.extend([adj_np[i] for i in range(adj_np.shape[0])])
            train_node_list.extend([set(train_nodes_array[i]) for i in range(train_nodes_array.shape[0])])
            
        assert len(gen_adj_list)==len(train_node_list)
        
        lower_lim = 0
        upper_lim = 100
        thres1 = 0.345
        # thres2 = 0.14
        samples_int = quantize_DegreeBound_woMotif(gen_adj_list, train_node_list, thres1, thres1, lower_lim, upper_lim)
        gen_graph_list = adjsWnodes_to_graphs(samples_int, train_node_list, True)

        assert len(gen_graph_list)==len(test_graph_list)
        save_dir = save_graph_list(self.log_folder_name, f"{self.config.ckpt[:10]}_4Ablation_sample_{str(thres1)[-2:]}", gen_graph_list)
        print("Evaluation By Compund Finish")