# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import RecWithContrastiveLearningDataset
from trainers import CoSeRecTrainer, ASTARTrainer
from models import SASRecModel, OfflineItemSimilarity, OnlineItemSimilarity
from ASTAR import Augmenter
from utils import EarlyStopping, get_user_seqs, get_item2attribute_json, check_path, set_seed


def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        val = getattr(args, arg)
        try:
            print(f"{arg:<30} : {val:>35}")
        except TypeError:
            print(f"{arg:<30} : {str(val):>35}")


def main():
    parser = argparse.ArgumentParser()

    # system args
    parser.add_argument('--data_dir',    default='data/', type=str)
    parser.add_argument('--output_dir',  default='output/',   type=str)
    parser.add_argument('--data_name',   default='Beauty',    type=str)
    parser.add_argument('--do_eval',     action='store_true')
    parser.add_argument('--model_idx',   default=3,           type=int)
    parser.add_argument('--gpu_id',      default='0',         type=str)

    # model selector
    parser.add_argument('--model_name',  default='ASTAR',   type=str,
                        help='CoSeRec or ASTAR')

    # data augmentation args
    parser.add_argument('--noise_ratio',             default=0.0,          type=float)
    parser.add_argument('--training_data_ratio',     default=1.0,          type=float)
    parser.add_argument('--augment_threshold',       default=5,            type=int)
    parser.add_argument('--similarity_model_name',   default='ItemCF_IUF', type=str)
    parser.add_argument('--augmentation_warm_up_epoches', default=400,     type=float)
    parser.add_argument('--base_augment_type',       default='mask',       type=str)
    parser.add_argument('--augment_type_for_short',  default='SIM',        type=str)
    parser.add_argument('--tao',                     default=0.2,          type=float)
    parser.add_argument('--gamma',                   default=0.7,          type=float)
    parser.add_argument('--beta',                    default=0.2,          type=float)
    parser.add_argument('--substitute_rate',         default=0.1,          type=float)
    parser.add_argument('--insert_rate',             default=0.4,          type=float)
    parser.add_argument('--max_insert_num_per_pos',  default=1,            type=int)

    # contrastive learning args
    parser.add_argument('--temperature', default=1.0,  type=float)
    parser.add_argument('--n_views',     default=2,    type=int)

    # model args
    parser.add_argument('--hidden_size',                  default=64,     type=int)
    parser.add_argument('--num_hidden_layers',            default=2,      type=int)
    parser.add_argument('--num_attention_heads',          default=2,      type=int)
    parser.add_argument('--hidden_act',                   default='gelu', type=str)
    parser.add_argument('--attention_probs_dropout_prob', default=0.5,    type=float)
    parser.add_argument('--hidden_dropout_prob',          default=0.5,    type=float)
    parser.add_argument('--initializer_range',            default=0.02,   type=float)
    parser.add_argument('--max_seq_length',               default=50,     type=int)

    # train args
    parser.add_argument('--lr',           default=0.001, type=float)
    parser.add_argument('--batch_size',   default=256,   type=int)
    parser.add_argument('--epochs',       default=400,   type=int)
    parser.add_argument('--no_cuda',      action='store_true')
    parser.add_argument('--log_freq',     default=1,     type=int)
    parser.add_argument('--seed',         default=1,     type=int)
    parser.add_argument('--cf_weight',    default=0.25,   type=float)
    parser.add_argument('--rec_weight',   default=1.0,   type=float)
    parser.add_argument('--weight_decay', default=0.0,   type=float)
    parser.add_argument('--adam_beta1',   default=0.9,   type=float)
    parser.add_argument('--adam_beta2',   default=0.999, type=float)

    # ASTAR-specific args
    parser.add_argument('--reclp_weight',  default=0.4,  type=float,
                        help='weight of last-position rec loss in recommender phase')
    parser.add_argument('--alpha',         default=0.1,  type=float,
                        help='augmenter adversarial weight')
    parser.add_argument('--astar_beta',    default=1.0,  type=float,
                        help='augmenter preservation weight')
    parser.add_argument('--warmup_epochs', default=0,   type=int,
                        help='epochs before adversarial game starts')
    parser.add_argument('--max_grad_norm', default=5.0,  type=float)
    parser.add_argument('--aug_tau',       default=2.0,  type=float,
                        help='initial augmenter temperature')
    parser.add_argument('--aug_tau_decay', default=0.99, type=float)
    parser.add_argument('--aug_min_tau',   default=0.5,  type=float)
    parser.add_argument('--N_rand',        default=20,   type=int,
                        help='number of random substitution candidates in pool')
    parser.add_argument('--N_sim',         default=0,   type=int,
                        help='number of similarity-based candidates in pool (0=disabled)')
    parser.add_argument('--N_hist',        default=0,   type=int,
                        help='number of user history candidates in pool (0=disabled)')
    parser.add_argument('--use_item_similarity', action='store_true',
                        help='use precomputed item similarity for pool')

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + '.txt'

    user_seq, max_item, valid_rating_matrix, test_rating_matrix = \
        get_user_seqs(args.data_file)

    args.item_size = max_item + 2
    args.mask_id   = 0

    # save model args
    args_str           = f'{args.model_name}-{args.data_name}-{args.model_idx}'
    args.log_file      = os.path.join(args.output_dir, args_str + '.txt')
    args.train_matrix  = valid_rating_matrix
    checkpoint         = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    show_args_info(args)
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # ── Item similarity precomputation ────────────────────────────────────────
    args.similarity_model_path = os.path.join(
        args.data_dir,
        args.data_name + '_' + args.similarity_model_name + '_similarity.pkl'
    )
    offline_similarity_model = OfflineItemSimilarity(
        data_file=args.data_file,
        similarity_path=args.similarity_model_path,
        model_name=args.similarity_model_name,
        dataset_name=args.data_name
    )
    args.offline_similarity_model = offline_similarity_model

    online_similarity_model = OnlineItemSimilarity(item_size=args.item_size)
    args.online_similarity_model = online_similarity_model

    # ── ASTAR: precompute item similarity tensor for pool ─────────────────────
    args.item_similarity = None
    if args.model_name == 'ASTAR' and args.use_item_similarity and args.N_sim > 0:
        print("Precomputing item similarity tensor for ASTAR pool...")
        top_k      = args.N_sim
        sim_matrix = []
        for item_id in range(args.item_size):
            top_k_items = offline_similarity_model.most_similar(item_id, top_k=top_k)
            sim_matrix.append(top_k_items)
        args.item_similarity = torch.tensor(
            sim_matrix, dtype=torch.long
        )
        if args.cuda_condition:
            args.item_similarity = args.item_similarity.cuda()
        print(f"Item similarity tensor: {args.item_similarity.shape}")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    train_dataset = RecWithContrastiveLearningDataset(
        args,
        user_seq[:int(len(user_seq) * args.training_data_ratio)],
        data_type='train'
    )
    train_sampler    = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size
    )

    eval_dataset     = RecWithContrastiveLearningDataset(args, user_seq, data_type='valid')
    eval_sampler     = SequentialSampler(eval_dataset)
    eval_dataloader  = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
    )

    test_dataset     = RecWithContrastiveLearningDataset(args, user_seq, data_type='test')
    test_sampler     = SequentialSampler(test_dataset)
    test_dataloader  = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=args.batch_size
    )

    # ── Model + Trainer ───────────────────────────────────────────────────────
    model = SASRecModel(args=args)

    if args.model_name == 'ASTAR':
        # Pass ASTAR-specific args to augmenter
        args.tau       = args.aug_tau
        args.tau_decay = args.aug_tau_decay
        args.min_tau   = args.aug_min_tau
        args.beta      = args.astar_beta   # augmenter beta (not CoSeRec beta)

        adv_model = Augmenter(args, N_rand=args.N_rand, N_sim=args.N_sim, N_hist=args.N_hist)

        trainer = ASTARTrainer(
            model, adv_model,
            train_dataloader, eval_dataloader, test_dataloader,
            args
        )
        print(f'Train ASTAR')

    else:
        # CoSeRec: adv_model is a dummy placeholder to satisfy Trainer signature
        adv_model = SASRecModel(args=args)

        trainer = CoSeRecTrainer(
            model, adv_model,
            train_dataloader, eval_dataloader, test_dataloader,
            args
        )
        print(f'Train CoSeRec')

    # ── Training / Evaluation ─────────────────────────────────────────────────
    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.test(0, full_sort=True)

    else:
        early_stopping = EarlyStopping(args.checkpoint_path, patience=40, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        trainer.args.train_matrix = test_rating_matrix
        print('---------------Change to test_rating_matrix!-------------------')
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

    print(args_str)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
    
    trainer.analyzer.plot(
        save_dir=os.path.join(args.output_dir, 'plots'),
        dataset_name=args.data_name
    )
    trainer.analyzer.print_summary(dataset_name=args.data_name)

main()

"""
1. Added --model_name ASTAR selector
   CoSeRec: original behaviour unchanged
   ASTAR:   builds Augmenter, uses ASTARTrainer

2. Added all ASTAR-specific args:
   --reclp_weight, --alpha, --astar_beta, --warmup_epochs
   --aug_tau, --aug_tau_decay, --aug_min_tau
   --N_rand, --N_sim, --use_item_similarity

3. Item similarity tensor precomputed once before training
   Only when --model_name ASTAR --use_item_similarity --N_sim > 0
   Stored in args.item_similarity → passed to augmenter forward

4. CoSeRec: adv_model is a dummy SASRecModel
   Satisfies Trainer signature without changing CoSeRec logic

5. Run ASTAR:
   python main.py --model_name ASTAR --data_name Beauty

   Run CoSeRec (unchanged):
   python main.py --model_name CoSeRec --data_name Beauty
"""