import os
import numpy as np
import random
import torch
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets import RecWithContrastiveLearningDataset
from trainers import CoSeRecTrainer, AdvAugmentTrainer, ASTARv2Trainer
from recommender import SASRecModel
from augmenter import Augmenter
from ASTAR import ASTARv2Augmenter
from utils import EarlyStopping, get_user_seqs, get_item2attribute_json, check_path, set_seed
from diagnose import GANDiagnostics


def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        value = getattr(args, arg)
        value_str = str(value) if value is not None else "None"
        print(f"{arg:<30} : {value_str:>35}")

def initialize_parser():
    parser = argparse.ArgumentParser()

    # system args
    parser.add_argument('--data_dir', default='/home/chenyixu/CoSeRec/data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--tensorboard_dir', default=None, type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--model_idx', default=0, type=int)
    parser.add_argument("--gpu_id", type=str, default="0")

    # data augmentation args
    parser.add_argument('--noise_ratio', default=0.0, type=float)
    parser.add_argument('--training_data_ratio', default=1.0, type=float)
    parser.add_argument('--augment_threshold', default=4, type=int)
    parser.add_argument('--similarity_model_name', default='ItemCF_IUF', type=str)
    parser.add_argument("--augmentation_warm_up_epoches", type=float, default=400)
    parser.add_argument('--base_augment_type', default='reorder', type=str)
    parser.add_argument('--augment_type_for_short', default='SIM', type=str)
    parser.add_argument("--tao", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--substitute_rate", type=float, default=0.1)
    parser.add_argument("--insert_rate", type=float, default=0.4)
    parser.add_argument("--max_insert_num_per_pos", type=int, default=1)

    # contrastive learning args
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--n_views', default=2, type=int, metavar='N')

    # model args
    parser.add_argument("--model_name", default='CoSeRec', type=str)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--cf_weight", type=float, default=0.2)
    parser.add_argument("--check_weight", type=float, default=0.2)
    parser.add_argument("--rec_weight", type=float, default=1.0)

    # learning related
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)

    # augmenter args
    parser.add_argument("--mask_tau", type=float, default=10.0)
    parser.add_argument("--penalty_weight", type=float, default=0.1)
    parser.add_argument("--reg_weight", type=float, default=0.2)
    parser.add_argument("--asym_weight", type=float, default=0.2)
    parser.add_argument("--target_rate", type=float, default=0.5)
    parser.add_argument("--ratio", type=float, default=0.7)
    parser.add_argument('--run_id', default=None, type=str)
    # Warmup: number of epochs before adversarial augmenter updates begin
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Epochs of recommender-only warmup before adversarial "
                             "augmenter updates start (ASTAR and CoSeRec paths).")
    # ASTARv2 ablation-specific args
    parser.add_argument("--v2_cl_weight", type=float, default=0.2,
                        help="Weight for view-to-view contrastive loss in ASTARv2.")
    parser.add_argument("--transport_reg_weight", type=float, default=0.1,
                        help="Weight for transport entropy+balance regularisation "
                             "in ASTARv2 augmenter update.")
    parser.add_argument("--transport_K", type=int, default=4,
                        help="Number of inter-sequence samples K for transport pool "
                             "in ASTARv2 (pool size = (1+K)*max_seq_length).")

    args = parser.parse_args()
    return args


def main():
    args = initialize_parser()

    set_seed(args.seed)
    check_path(args.output_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + '.txt'

    user_seq, max_item, valid_rating_matrix, test_rating_matrix = \
        get_user_seqs(args.data_file)

    args.item_size = max_item + 2
    args.mask_id = 0
    args.model_idx = 111
    args_str = f'{args.model_name}-{args.data_name}-{args.model_idx}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')

    show_args_info(args)

    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    args.train_matrix = valid_rating_matrix
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # -----------   datasets  --------- #
    train_dataset = RecWithContrastiveLearningDataset(
        args, user_seq[:int(len(user_seq) * args.training_data_ratio)], data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    # -----------   models  --------- #
    model = SASRecModel(args=args)

    if args.model_name == 'ASTARv2':
        # ── ASTARv2: dual-view transport ablation ────────────────────────────
        adv_model = ASTARv2Augmenter(
            num_items=args.item_size,
            hidden_size=args.hidden_size,
            max_seq_len=args.max_seq_length,
            num_heads=args.num_attention_heads,
            num_layers=args.num_hidden_layers,
            inner_size=args.hidden_size * 4,
            dropout=args.hidden_dropout_prob,
            K=args.transport_K,
            tau_init=args.mask_tau,
            tau_floor=1.0,
            tau_decay=0.99,
        )

        # -----------   trainer  --------- #
        trainer = ASTARv2Trainer(
            model, adv_model,
            train_dataloader, eval_dataloader, test_dataloader,
            args,
        )

    elif args.model_name == 'ASTAR':
        # ── ASTAR: adversarial mask-based augmentation (bugfixed) ────────────
        adv_model = Augmenter(args)

        diag = GANDiagnostics(window_size=50)
        probe_batch = next(iter(train_dataloader))
        _, probe_ids, _, _, _ = probe_batch[0]
        diag.register_probe(probe_ids)

        trainer = AdvAugmentTrainer(
            model, adv_model,
            train_dataloader, eval_dataloader, test_dataloader,
            args,
            diagnostics=diag,
        )

    else:
        # ── CoSeRec (default) ────────────────────────────────────────────────
        adv_model = Augmenter(args)

        diag = GANDiagnostics(window_size=50)
        probe_batch = next(iter(train_dataloader))
        _, probe_ids, _, _, _ = probe_batch[0]
        diag.register_probe(probe_ids)

        trainer = CoSeRecTrainer(
            model, adv_model,
            train_dataloader, eval_dataloader, test_dataloader,
            args,
            diagnostics=diag,
        )

    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.test(0, full_sort=True)

    else:
        print(f'Train {args.model_name}')
        early_stopping = EarlyStopping(args.checkpoint_path, patience=40, verbose=True)

        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Plot diagnostics (ASTAR / CoSeRec paths only)
        if args.model_name != 'ASTARv2':
            diag.plot('diagnostics.png')

        trainer.args.train_matrix = test_rating_matrix
        print('---------------Change to test_rating_matrix!-------------------')
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

    print(args_str)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')

main()