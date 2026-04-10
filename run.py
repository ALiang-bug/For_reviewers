
import time
import datetime
import numpy as np 
import Constants
import torch
from torch.utils.data import DataLoader
from dataLoader import datasets, Read_data, Split_data

from tqdm import tqdm
from utils.parsers import parser
from utils.util import *
from utils.EarlyStopping import *
from utils.Metrics import Metrics
from utils.graphConstruct import ConRelationGraph, ConHypergraph

from model.model import SNEP

metric = Metrics()
opt = parser.parse_args() 

def init_seeds(seed=2026):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_performance(crit, pred, gold):

    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
    return loss, n_correct

def count_model_params(model):
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        param_num = param.numel()
        total_params += param_num
        if param.requires_grad:
            trainable_params += param_num
    total_params_str = f"{total_params / 1e6:.2f}M" if total_params >= 1e6 else f"{total_params / 1e3:.2f}K"
    trainable_params_str = f"{trainable_params / 1e6:.2f}M" if trainable_params >= 1e6 else f"{trainable_params / 1e3:.2f}K"
    return total_params, trainable_params, total_params_str, trainable_params_str

def model_training(model, train_loader, epoch):
    ''' model training '''

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0

    print('start training: ', datetime.datetime.now())
    model.train()
    for step, (cascade_item, label, cascade_time, label_time, cascade_len) in enumerate(train_loader):
        cascade_item = cascade_item.long()
        tar = label.long()
        
        n_words = tar.data.ne(Constants.PAD).sum().float().item()
        n_total_words += n_words

        model.zero_grad()
        output_online, online_pred, target_proj, couple_pred, couple_proj = model(cascade_item, tar)
        loss, n_correct = get_performance(model.loss_function, output_online, tar)
        loss_BYOL = model.criterion(online_pred, target_proj, couple_pred, couple_proj)
        loss = loss + opt.lambda0 * loss_BYOL
    
        loss.backward()
        model.optimizer.step()
        model.optimizer.update_learning_rate()

        model.update_target_ema()
        if torch.isinf(model.user_embedding.weight).any():
            print(0)
        
        if torch.isnan(model.user_embedding.weight).any():
            print(0)

        total_loss += loss.item()
        n_total_correct += n_correct

        torch.cuda.empty_cache()

    print('\tTotal Loss:\t%.3f' % total_loss)

    return total_loss, n_total_correct/n_total_words

def model_testing(model, test_loader, k_list=[10, 50, 100], valid=False):
    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0.0
    n_correct = 0.0
    total_loss = 0.0


    if valid:
        print('start valid predicting: ', datetime.datetime.now())
        model.eval()
    else:
        print("Loading best model for final testing...")
        model.load_state_dict(torch.load(opt.save_path))
        print('start test predicting: ', datetime.datetime.now())
        model.eval()

    epoch_test_time = 0.0

    with torch.no_grad():
        for step, (cascade_item, label, cascade_time, label_time, cascade_len) in enumerate(test_loader):

            cascade_item = cascade_item.long()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()

            y_pred, att_score, _, _ ,_, _ = model.model_prediction(cascade_item, cascade_time)

            epoch_test_time += time.time() - start

            y_pred = y_pred.detach().cpu()
            tar = label.view(-1).detach().cpu()

            pred = y_pred.max(1)[1]
            gold = tar.contiguous().view(-1)
            correct = pred.data.eq(gold.data)
            n_correct = correct.masked_select(gold.ne(Constants.PAD).data).sum().float()

            scores_batch, scores_len = metric.compute_metric(y_pred, tar, k_list)
            n_total_words += scores_len

            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

        for k in k_list:
            scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
            scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

        return scores, n_correct/n_total_words, epoch_test_time

def train_test(epoch, model, train_loader, val_loader, test_loader):

    total_loss, _ = model_training(model, train_loader, epoch)
    val_scores, val_accuracy, _ = model_testing(model, val_loader, valid=True)
    test_scores, test_accuracy, epoch_test_time = model_testing(model, test_loader, valid=True)  # 另以逻辑实现验证

    return total_loss, val_scores, test_scores, val_accuracy.item(), test_accuracy.item(), epoch_test_time

def main(data_path, seed=503):

    init_seeds(seed)
    if opt.preprocess:
        Split_data(data_path, train_rate=opt.train_rate, valid_rate=opt.valid_rate, load_dict=True)
    train, valid, test, user_size = Read_data(data_path)

    train_data = datasets(train, opt.max_lenth)
    val_data = datasets(valid, opt.max_lenth)
    test_data = datasets(test, opt.max_lenth)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=opt.batch_size,
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )

    opt.n_node = user_size
    relation_graph, social_pagerank, social_outdegree, social_indegree = ConRelationGraph(data_path)
    start = time.time()
    HG_Item, HG_User, Diffusion_adj, temporal_similarity = ConHypergraph(opt.data_name, opt.n_node, opt.window)
    print(f"Construct Hypergraph Time：{time.time() - start}")
    save_model_path = opt.save_path
    early_stopping = EarlyStopping(patience=opt.patience, verbose=True, path=save_model_path)
    model = trans_to_cuda(SNEP(social_graph = [relation_graph, social_pagerank, social_outdegree, social_indegree], hypergraphs=[HG_Item, HG_User, Diffusion_adj], temporal_similarity=temporal_similarity,  opt = opt, reverse = True, dropout=opt.dropout))

    top_K = [10, 50, 100]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    validation_history = 0.0
    print(opt)
    total_test_time = 0.0
    epochs = 0
    for epoch in range(opt.epoch):
        print('\n[ Epoch', epoch, ']')
        total_loss, val_scores, test_scores, val_accuracy, test_accuracy, epoch_test_time = train_test(epoch, model, train_loader, val_loader, test_loader)
        total_test_time += epoch_test_time
        epochs += 1
        print('  - ( Validation )) ')
        for metric in val_scores.keys():
            print(metric + ' ' + str(val_scores[metric]))

        if validation_history <= sum(val_scores.values()):
            validation_history = sum(val_scores.values())
            for K in top_K:
                test_scores['hits@' + str(K)] = test_scores['hits@' + str(K)] * 100
                test_scores['map@' + str(K)] = test_scores['map@' + str(K)] * 100

                best_results['metric%d' % K][0] = test_scores['hits@' + str(K)]
                best_results['epoch%d' % K][0] = epoch
                best_results['metric%d' % K][1] = test_scores['map@' + str(K)]
                best_results['epoch%d' % K][1] = epoch


        early_stopping(-sum(list(val_scores.values())), model)
        if early_stopping.early_stop:
            print("Early_Stopping")
            break
    print(" -(Finished!!) \n parameter settings: ")
    print("--------------------------------------------")    
    print(opt)

    print(" -(Finished!!) \n test scores: ")
    print("--------------------------------------------")
    for K in top_K:
        print('Recall@%d: %.4f\tMAP@%d: %.4f' %
              (K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1]))
    total_params, trainable_params, total_params_str, trainable_params_str = count_model_params(model)
    print(f"\n[ Model Parameter Info ]")
    print(f"  - Total Parameters: {total_params} ({total_params_str})")
    print(f"  - Trainable Parameters: {trainable_params} ({trainable_params_str})")
    print(f"Average Infer Time：{total_test_time / epochs}")

if __name__ == "__main__": 
    main(opt.data_name, opt.seed)



