from __future__ import division
from __future__ import print_function
import copy
import argparse
import random
from timeit import default_timer as timer
from models import GCN, GAT, SAGE
from query import *
from dataset import load_data
import matplotlib.pyplot as plt
import pandas as pd
import os


if not os.path.exists('report'):
    os.makedirs('report')
    os.makedirs('report/avg')
        

def run(args, PERCENT_TO_DELETE_EDGE, result_to_report, KIND_DECAY):

    print('args.dataset:', args.dataset)
    print("PERCENT_TO_DELETE_EDGE:", PERCENT_TO_DELETE_EDGE)
    
    time_start_load_data = time.time()
    # Load dataset
    data_load_once, num_edges_deleted, list_edges_deleted, num_edges = load_data(name=args.dataset,
                    seed=args.seed,
                    PERCENT_TO_DELETE_EDGE = PERCENT_TO_DELETE_EDGE,
                     read=False, 
                     save=False)
    data_load_once = data_load_once.to(args.device)
    time_end_load_data = time.time()
    time_load_data = time_end_load_data - time_start_load_data
    print('args.device:', args.device)
    print('time_load_data:', time_load_data)
    print('num_edges_deleted:', num_edges_deleted)
    print('len(list_edges_deleted):', len(list_edges_deleted))
    

    for gnn in args.model:
        for baseline in args.baselines:
            
            for budget in args.budget:
                data = copy.deepcopy(data_load_once)
                

                budget = int(budget)
                seed = int(args.seed)

                # Set seeds
                if args.verbose == 1:
                    print('Seed {:03d}:'.format(seed))
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

                model_args = {
                    "in_channels": data.num_features,
                    "out_channels": data.num_classes,
                    "hidden_channels": args.hidden,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "activation": args.activation,
                    "batchnorm": args.batchnorm
                }
                

                
                if gnn == "gat":
                    model_args["num_heads"] = args.num_heads
                    model_args["hidden_channels"] = int(args.hidden / args.num_heads)
                    model = GAT(args.dataset)
                elif gnn == "gcn":
                    model = GCN(args.dataset)
                elif gnn == "sage":
                    model = SAGE(args.dataset)
                else:
                    raise NotImplemented

                model = model.to(args.device)
                
                if baseline == "random":
                    agent = Random(data, model, seed, args)
                elif baseline == "density":
                    agent = Density(data, model, seed, args)
                elif baseline == "uncertainty":
                    agent = Uncertainty(data, model, seed, args)
                elif baseline == "coreset":
                    agent = CoreSetGreedy(data, model, seed, args)

                elif baseline == "degree":
                    agent = Degree(data, model, seed, args)
                elif baseline == "pagerank":
                    agent = PageRank(data, model, seed, args)
                elif baseline == "age":
                    agent = AGE(data, model, seed, args)
                elif baseline == "featprop":
                    agent = ClusterBased(data, model, seed, args,
                                         representation='aggregation',
                                         encoder='gcn')

                elif baseline == "graphpart":
                    agent = PartitionBased(data, model, seed, args,
                                           representation='aggregation',
                                           encoder='gcn',
                                           compensation=0)
                elif baseline == "graphpartfar":
                    agent = PartitionBased(data, model, seed, args,
                                           representation='aggregation',
                                           encoder='gcn',
                                           compensation=1)
                # Our Methods
                elif baseline == "ALIN":
                    agent = ALIN(data, model, seed, args,
                                           representation='aggregation',
                                           encoder='gcn',
                                           compensation=0)
                    
                elif baseline == "ALINFar":
                    agent = ALIN(data, model, seed, args,
                                           representation='aggregation',
                                           encoder='gcn',
                                           compensation=1)
                    
                    
                    
                agent.init_clf()

                # Initialization
                budget_report = budget
                training_mask = np.zeros(data.num_nodes, dtype=bool)
                initial_mask = np.arange(data.num_nodes)
                np.random.shuffle(initial_mask)
                
                training_mask = torch.tensor(training_mask)
                agent.update(training_mask)
                
                num_budget = args.rounds

                list_budget = [int(budget / num_budget) for i in range(num_budget - 1)]
                list_budget.append(budget - (num_budget - 1) * int(budget / num_budget))
                
                budget_first = list_budget[0]
                list_budget = list_budget[1:]
                
                # Query 1
                if baseline in ['ALIN', 'ALINFar']:
                    indices = agent.query_first_time(budget_first)
                elif baseline in ['density', 'uncertainty', 'coreset', 'age']:
                    indices = initial_mask[:budget_first]
                else:
                    indices = agent.query(budget_first)
                
                count_i = 1
                    
                # Update
                training_mask[indices] = True
                print('training_mask query first:', sum(training_mask))
                agent.update(training_mask)
                agent.update_edges(training_mask)
                loss_report = agent.train(baseline, count_i, KIND_DECAY)
                
                for budget_i in list_budget:
                    count_i += 1
                    indices = agent.query(budget_i)
                    training_mask[indices] = True
                    agent.update(training_mask)
                    agent.update_edges(training_mask)
                    loss_report = agent.train(baseline, count_i, KIND_DECAY)
                
                # Evaluate
                f1, acc = agent.evaluate()
                labelled = len(np.where(agent.data.train_mask != 0)[0])

                if args.verbose > 0:
                        print('Labelled nodes: {:d}, Prediction macro-f1 score {:.4f}'
                              .format(labelled, f1))
                else:
                    print("{}, {}, {}, {}, Labelled: {}, F1: {}"
                              .format(gnn, baseline,
                                      args.dataset, seed,
                                      labelled, f1))
                    
                if baseline not in result_to_report.keys():
                    result_to_report[baseline] = {}
                if budget_report not in result_to_report[baseline].keys():
                    result_to_report[baseline][budget_report] = []
                result_to_report[baseline][budget_report].append(f1)
                
                if baseline not in loss_to_report.keys():
                    loss_to_report[baseline] = {}
                if budget_report not in loss_to_report[baseline].keys():
                    loss_to_report[baseline][budget_report] = []
                loss_to_report[baseline][budget_report].append(loss_report)
                
            print()
        
    return result_to_report, loss_to_report

if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha_combine_los", type=int, default=0.05, help="Node Edge Combine Case")
    parser.add_argument(
        "--gamma_combine", type=int, default=0.5, help="Node Edge Combine Case")
    parser.add_argument(
        "--remove_percent", type=int, default=0.3, help="remove edge percent")
    parser.add_argument(
        "--kind_decay", type=str, default='cosine_annealing', help="kind function decay weight")
    parser.add_argument(
        "--budget", type=list, default=[200, 230, 260], help="budget")
    
    parser.add_argument(
        "--rounds", type=int, default=8, help="Number of rounds to run the agent.")
    
    
    parser.add_argument(
        "--verbose", type=int, default=0, help="Verbose: 0, 1 or 2")
    parser.add_argument(
        "--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # General configs
    parser.add_argument(
        "--baselines", type=list, default=['ALINFar'])
    parser.add_argument(
        "--model", type=list, default=['gcn'])
    parser.add_argument(
        "--dataset", type=list, default=['cora'])
    parser.add_argument(
        "--partition", type=str, default='greedy')

    # Active Learning parameters
    parser.add_argument(
        "--retrain", type=bool, default=True)
    parser.add_argument(
        "--num_centers", type=int, default=1)
    parser.add_argument(
        "--representation", type=str, default='features')
    parser.add_argument(
        "--compensation", type=float, default=1.0)
    parser.add_argument(
        "--init", type=float, default=0, help="Number of initially labelled nodes.")

    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of epochs to train.")
    # parser.add_argument(
    #     "--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument(
        "--steps", type=int, default=4, help="Number of steps of random walk.")

    # GNN parameters
    parser.add_argument(
        "--seed", type=int, default=0, help="Number of random seeds.")
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4,
        help="Weight decay (L2 loss on parameters).")
    parser.add_argument(
        "--hidden", type=int, default=16, help="Number of hidden units.")
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of layers.")
    parser.add_argument(
        "--dropout", type=float, default=0,
        help="Dropout rate (1 - keep probability).")
    parser.add_argument(
        "--batchnorm", type=bool, default=False,
        help="Perform batch normalization")
    parser.add_argument(
        "--activation", default="relu")

    # GAT hyper-parameters
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of heads.")

    args, _ = parser.parse_known_args()
    
    datasets = args.dataset
    baselines = args.baselines
    KIND_GNN = args.model
    PERCENT_TO_DELETE_EDGE = args.remove_percent
    KIND_DECAY = args.kind_decay
    budget = args.budget
    alpha_combine_los = args.alpha_combine_los
    gamma_combine = args.gamma_combine

    for dataset in datasets:
        args.dataset = dataset
        result_to_report = {
        }
        loss_to_report = {}
        
        for seed in range(3):
            args.seed = seed
            result_to_report, loss_to_report = run(args, PERCENT_TO_DELETE_EDGE, result_to_report, KIND_DECAY)
            print()
            
        
        print("result_to_report")
        print(result_to_report)
        
        print()
        
        print("loss_to_report")
        print(loss_to_report)
        
        
        for baseline in baselines:
            print()
            print(baseline)
            try:
                for key, value in result_to_report[baseline].items():
                    print(key)
                    value = [round(val * 100, 1) for val in value]
                    avg = round((max(value) + min(value)) / 2, 1)
                    print(str(avg) + ' +- ' + str(round(avg - min(value), 1)))
                    print()
            except:
                print("No " + baseline)
            
        result_to_report = pd.DataFrame(result_to_report)
        swapped_df = result_to_report.transpose()
        swapped_df_percent = swapped_df.applymap(lambda x: [round(val * 100, 1) for val in x])
        swapped_df_percent.to_csv('report/result_{}_{}_{}_{}_{}_{}.csv'.format(dataset, PERCENT_TO_DELETE_EDGE, KIND_GNN, alpha_combine_los, gamma_combine, KIND_DECAY))
        
                
   