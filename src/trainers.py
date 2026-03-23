"""
trainers.py — ASTAR adversarial training on top of CoSeRec pipeline.

Inherits CoSeRecTrainer structure exactly.
Only change: augmentation is replaced by ASTAR's learned T matrix.

Key differences from CoSeRecTrainer:
    1. Two models: recommender (model) + augmenter (adv_model)
    2. Two-phase training per batch:
           Phase 1: recommender update (augmenter frozen, hard T)
           Phase 2: augmenter update  (recommender frozen, soft T)
    3. λ-adjusted contrastive loss (harder augmentation → higher weight)
    4. Warmup: augmenter starts after warmup_epochs (default 20)
               before warmup: recommender trains on original sequence only
               after warmup:  full adversarial game

Loss Functions:
    Recommender (Phase 1):
        L_B = rec_weight   * L_rec_orig
            + reclp_weight * L_rec_org
            + cf_weight    * λ_mean * L_contrast(hard_aug, org_seq)

        where:
            L_rec_orig:    cross_entropy on original sequence [B, L] supervision
            L_rec_org:     cross_entropy on original sequence last position
            L_contrast:    InfoNCE between hard augmented and original views
            λ_mean:        mean λ across batch (adjusts contrast loss weight)

    Augmenter (Phase 2):
        L_A = beta  * L_rec_aug(soft_mixed)
            - alpha * L_contrast(soft_aug, org_seq)

        where:
            L_rec_aug:     cross_entropy on soft λ-blended view (last position)
            L_contrast:    InfoNCE between soft and original views
            beta, alpha:   hyperparameters controlling preservation vs diversity

Contrastive Learning:
    Original sequence vs augmented view.
    Uses full sequence flatten [B, L*D] → NCELoss.
    Consistent with CoSeRec's _one_pair_contrastive_learning.

Warmup Strategy:
    epoch < warmup_epochs:
        Only recommender trains on original sequence
        No augmentation, no adversarial game

    epoch >= warmup_epochs:
        Both phases run every batch
        Full adversarial training

Temperature:
    augmenter.decay_tau() called once per epoch after warmup only.

Evaluation:
    Identical to CoSeRecTrainer.
    Recommender evaluated on original sequences only.
    No augmentation at inference.

Hyperparameters (all from args):
    rec_weight:    weight of original rec loss            (default 1.0)
    cf_weight:     weight of contrastive loss             (default 0.1)
    reclp_weight:  weight of last position rec loss       (default 0.1)
    alpha:         augmenter adversarial weight           (default 1.0)
    beta:          augmenter preservation weight          (default 0.5)
    warmup_epochs: epochs before adversarial starts       (default 20)
"""

import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import DataLoader, RandomSampler
from datasets import RecWithContrastiveLearningDataset
from modules import NCELoss, NTXent
from analyze_T import TMatrixAnalyzer
from utils import recall_at_k, ndcg_k, get_metric, get_user_seqs, nCr


class Trainer:
    def __init__(self, model, adv_model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):

        self.args           = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device         = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model                   = model
        self.adv_model               = adv_model
        self.online_similarity_model = args.online_similarity_model
        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)

        self.analyzer = TMatrixAnalyzer(args, N_rand=args.N_rand, N_sim=args.N_sim)

        self.projection = nn.Sequential(
            nn.Linear(self.args.max_seq_length * self.args.hidden_size,
                      512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.args.hidden_size, bias=True)
        )

        if self.cuda_condition:
            self.model.cuda()
            self.projection.cuda()

        self.train_dataloader = train_dataloader
        self.eval_dataloader  = eval_dataloader
        self.test_dataloader  = test_dataloader

        betas          = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim     = Adam(self.model.parameters(), lr=self.args.lr,
                              betas=betas, weight_decay=self.args.weight_decay)
        self.optim_adv = Adam(self.adv_model.parameters(), lr=self.args.lr,
                              betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:",
              sum([p.nelement() for p in self.model.parameters()]))

        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        print("self.cf_criterion:", self.cf_criterion.__class__.__name__)

    def __refresh_training_dataset(self, item_embeddings):
        user_seq, _, _, _ = get_user_seqs(self.args.data_file)
        self.args.online_similarity_model.update_embedding_matrix(item_embeddings)
        train_dataset    = RecWithContrastiveLearningDataset(
            self.args, user_seq,
            data_type='train', similarity_model_type='hybrid'
        )
        train_sampler    = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=self.args.batch_size
        )
        return train_dataloader

    def train(self, epoch):
        if epoch > self.args.augmentation_warm_up_epoches:
            print("refresh dataset with updated item embedding")
            self.train_dataloader = self.__refresh_training_dataset(
                self.model.item_embeddings
            )
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader,
                              full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader,
                              full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1":  '{:.4f}'.format(HIT_1),  "NDCG@1":  '{:.4f}'.format(NDCG_1),
            "HIT@5":  '{:.4f}'.format(HIT_5),  "NDCG@5":  '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5":  '{:.4f}'.format(recall[0]), "NDCG@5":  '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3]),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        pos_emb    = self.model.item_embeddings(pos_ids)
        neg_emb    = self.model.item_embeddings(neg_ids)
        pos        = pos_emb.view(-1, pos_emb.size(2))
        neg        = neg_emb.view(-1, neg_emb.size(2))
        seq_emb    = seq_out.view(-1, self.args.hidden_size)
        pos_logits = torch.sum(pos * seq_emb, -1)
        neg_logits = torch.sum(neg * seq_emb, -1)
        # infer from actual input size instead of hardcoding max_seq_length
        istarget   = (pos_ids > 0).view(-1).float()
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)
        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        test_logits   = torch.bmm(test_item_emb,
                                  seq_out.unsqueeze(-1)).squeeze(-1)
        return test_logits

    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred   = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def _contrastive_from_embeds(self, mixed, input_ids_orig):
        """
        Contrastive loss: original sequence vs augmented view.
        Uses full sequence flatten [B, L*D] — consistent with CoSeRec.

        mixed:          [B, L, D]  augmented embeddings (hard or soft)
        input_ids_orig: [B, L]     for attention mask
        """
        out_orig  = self.model.transformer_encoder(input_ids_orig)     # [B, L, D]
        out_aug   = self.model.transformer_encoder_from_embeds(
            mixed, input_ids_orig
        )                                                               # [B, L, D]
        flat_orig = out_orig.view(out_orig.shape[0], -1)               # [B, L*D]
        flat_aug  = out_aug.view(out_aug.shape[0],  -1)                # [B, L*D]
        return self.cf_criterion(flat_orig, flat_aug)


class ASTARTrainer(Trainer):

    def __init__(self, model, adv_model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):
        super(ASTARTrainer, self).__init__(
            model, adv_model,
            train_dataloader, eval_dataloader, test_dataloader,
            args
        )

        self.alpha           = getattr(args, 'alpha',         1.0)
        self.beta            = getattr(args, 'beta',          0.5)
        self.warmup_epochs   = getattr(args, 'warmup_epochs', 20)
        self.max_grad_norm   = getattr(args, 'max_grad_norm', 5.0)
        self.item_similarity = getattr(args, 'item_similarity', None)

        if self.cuda_condition:
            self.adv_model.cuda()

        self.analyzer = TMatrixAnalyzer(args, N_rand=args.N_rand, N_sim=args.N_sim)

    # ── Phase 1: Recommender update ──────────────────────────────────────────

    def _phase1_recommender(self, input_ids, target_pos, target_neg, epoch):
        """
        Update recommender. Augmenter frozen. Hard T.

        L_B = rec_weight   * L_rec_orig
            + reclp_weight * L_rec_org
            + cf_weight    * lam_mean * L_contrast(orig, hard_aug)
        """
        self.model.train()
        self.adv_model.eval()
        self.optim.zero_grad()

        # ── Original sequence ─────────────────────────────────────────────────
        seq_out_orig = self.model.transformer_encoder(input_ids)        # [B, L, D]
        L_rec_orig   = self.cross_entropy(seq_out_orig, target_pos, target_neg)
        L_rec_org    = self.cross_entropy(
            seq_out_orig[:, -1:, :],
            target_pos[:, -1:],
            target_neg[:, -1:]
        )

        # ── Hard augmented view (no grad to augmenter) ────────────────────────
        with torch.no_grad():
            soft_mixed, hard_mixed, lam, T, own_mask = self.adv_model(
                input_ids,
                self.model.item_embeddings,
                item_similarity=self.item_similarity,
            )
        lam_mean   = lam.mean().detach()
        if epoch % 5 == 0:
            self.analyzer.record(T, lam, own_mask, epoch)

        # ── Contrastive: orig vs hard aug ─────────────────────────────────────
        L_contrast = self._contrastive_from_embeds(soft_mixed, input_ids)

        # ── Total ─────────────────────────────────────────────────────────────
        L_B = (self.args.rec_weight    * L_rec_orig
             + self.args.reclp_weight  * L_rec_org
             + self.args.cf_weight     * lam_mean * L_contrast)

        L_B.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optim.step()

        return {
            'L_rec_orig': L_rec_orig.item(),
            'L_rec_org':  L_rec_org.item(),
            'L_contrast': L_contrast.item(),
            'L_B':        L_B.item(),
            'lam_mean':   lam_mean.item(),
        }

    # ── Phase 2: Augmenter update ─────────────────────────────────────────────

    def _phase2_augmenter(self, input_ids, target_pos, target_neg):
        """
        Update augmenter. Recommender parameters frozen. Soft T.

        L_A = beta  * L_rec_aug(soft_mixed)
            - alpha * L_contrast(orig, soft_aug)

        Recommender parameters frozen via requires_grad_(False).
        Gradient flows: loss → seq_out_aug → soft_mixed → T → augmenter.
        """
        self.model.eval()
        self.adv_model.train()
        self.optim_adv.zero_grad()

        # ── Freeze recommender parameters ─────────────────────────────────────
        for param in self.model.parameters():
            param.requires_grad_(False)

        # ── Frozen original ───────────────────────────────────────────────────
        seq_out_orig = self.model.transformer_encoder(input_ids)        # [B, L, D]

        # ── Soft augmented view ───────────────────────────────────────────────
        soft_mixed, hard_mixed, lam, T, own_mask = self.adv_model(
            input_ids,
            self.model.item_embeddings,
            item_similarity=self.item_similarity,
        )

        # Gradient flows through soft_mixed → T → augmenter
        # Recommender parameters frozen so they don't update
        seq_out_aug = self.model.transformer_encoder_from_embeds(
            soft_mixed, input_ids
        )                                                               # [B, L, D]

        # ── Rec loss: last position ───────────────────────────────────────────
        L_rec_aug = self.cross_entropy(
            seq_out_aug[:, -1:, :],
            target_pos[:, -1:],
            target_neg[:, -1:]
        )

        # ── Contrastive: augmenter maximizes distance from original ───────────
        flat_orig = seq_out_orig.view(seq_out_orig.shape[0], -1)        # [B, L*D]
        flat_aug  = seq_out_aug.view(seq_out_aug.shape[0],  -1)         # [B, L*D]
        L_contrast = self.cf_criterion(flat_orig, flat_aug)

        # ── Total augmenter loss ──────────────────────────────────────────────
        L_A = self.beta * L_rec_aug - self.alpha * L_contrast

        L_A.backward()
        torch.nn.utils.clip_grad_norm_(self.adv_model.parameters(), self.max_grad_norm)
        self.optim_adv.step()

        # ── Unfreeze recommender for Phase 1 next batch ───────────────────────
        for param in self.model.parameters():
            param.requires_grad_(True)

        return {
            'L_rec_aug_A':  L_rec_aug.item(),
            'L_contrast_A': L_contrast.item(),
            'L_A':          L_A.item(),
        }

    # ── Main iteration ────────────────────────────────────────────────────────

    def iteration(self, epoch, dataloader, full_sort=True, train=True):
        str_code = "train" if train else "test"

        if train:
            self.model.train()
            self.adv_model.train()

            metrics = {
                'L_rec_orig':   0.0,
                'L_rec_org':    0.0,
                'L_contrast':   0.0,
                'L_B':          0.0,
                'lam_mean':     0.0,
                'L_rec_aug_A':  0.0,
                'L_contrast_A': 0.0,
                'L_A':          0.0,
            }

            is_warmup        = epoch < self.warmup_epochs
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches) in rec_cf_data_iter:
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                if is_warmup:
                    # Warmup: recommender trains on original sequence only
                    self.model.train()
                    self.optim.zero_grad()
                    seq_out = self.model.transformer_encoder(input_ids)
                    L_rec   = self.cross_entropy(seq_out, target_pos, target_neg)
                    L_rec.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optim.step()
                    metrics['L_rec_orig'] += L_rec.item()
                    metrics['L_B']        += L_rec.item()

                else:
                    # Full adversarial training
                    stats1 = self._phase1_recommender(input_ids, target_pos, target_neg, epoch)
                    stats2 = self._phase2_augmenter(input_ids, target_pos, target_neg)
                    for k, v in {**stats1, **stats2}.items():
                        if k in metrics:
                            metrics[k] += v

                if i % 10 == 0:
                    rec_cf_data_iter.set_description(
                        f"Epoch {epoch} "
                        f"{'[warmup]' if is_warmup else '[adv]'}: "
                        f"L_B={metrics['L_B']/(i+1):.4f} "
                        f"L_A={metrics['L_A']/(i+1):.4f} "
                        f"λ={metrics['lam_mean']/(i+1):.3f}"
                    )

            # Decay tau after warmup only
            if not is_warmup:
                self.adv_model.decay_tau()

            num_batches = len(dataloader)
            for k in metrics:
                metrics[k] /= num_batches

            post_fix = {
                "epoch":        epoch,
                "warmup":       is_warmup,
                "L_rec_orig":   f"{metrics['L_rec_orig']:.4f}",
                "L_rec_org":    f"{metrics['L_rec_org']:.4f}",
                "L_contrast":   f"{metrics['L_contrast']:.4f}",
                "L_B":          f"{metrics['L_B']:.4f}",
                "lam_mean":     f"{metrics['lam_mean']:.4f}",
                "L_rec_aug_A":  f"{metrics['L_rec_aug_A']:.4f}",
                "L_contrast_A": f"{metrics['L_contrast_A']:.4f}",
                "L_A":          f"{metrics['L_A']:.4f}",
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))
            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            rec_data_iter = tqdm(
                enumerate(dataloader),
                desc="Recommendation EP_%s:%d" % (str_code, epoch),
                total=len(dataloader),
                bar_format="{l_bar}{r_bar}"
            )
            self.model.eval()
            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.transformer_encoder(input_ids)
                    recommend_output = recommend_output[:, -1, :]
                    rating_pred      = self.predict_full(recommend_output)
                    rating_pred      = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    ind             = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind         = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
                    if i == 0:
                        pred_list   = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list   = np.append(pred_list,   batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items   = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]
                    test_logits      = self.predict_sample(recommend_output, test_neg_items)
                    test_logits      = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                return self.get_sample_scores(epoch, pred_list)


class CoSeRecTrainer(Trainer):

    def __init__(self, model, adv_model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):
        super(CoSeRecTrainer, self).__init__(
            model, adv_model,
            train_dataloader, eval_dataloader, test_dataloader,
            args
        )

    def _one_pair_contrastive_learning(self, inputs):
        cl_batch            = torch.cat(inputs, dim=0).to(self.device)
        cl_sequence_output  = self.model.transformer_encoder(cl_batch)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        batch_size          = cl_batch.shape[0] // 2
        cl_output_slice     = torch.split(cl_sequence_flatten, batch_size)
        return self.cf_criterion(cl_output_slice[0], cl_output_slice[1])

    def iteration(self, epoch, dataloader, full_sort=True, train=True):
        str_code = "train" if train else "test"

        if train:
            self.model.train()
            rec_avg_loss             = 0.0
            cl_individual_avg_losses = [0.0 for _ in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss          = 0.0
            joint_avg_loss           = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches) in rec_cf_data_iter:
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                sequence_output = self.model.transformer_encoder(input_ids)
                rec_loss        = self.cross_entropy(sequence_output, target_pos, target_neg)

                cl_losses = []
                for cl_batch in cl_batches:
                    cl_loss = self._one_pair_contrastive_learning(cl_batch)
                    cl_losses.append(cl_loss)

                joint_loss = self.args.rec_weight * rec_loss
                for cl_loss in cl_losses:
                    joint_loss += self.args.cf_weight * cl_loss

                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()
                for j, cl_loss in enumerate(cl_losses):
                    cl_individual_avg_losses[j] += cl_loss.item()
                    cl_sum_avg_loss             += cl_loss.item()
                joint_avg_loss += joint_loss.item()

            post_fix = {
                "epoch":          epoch,
                "rec_avg_loss":   '{:.4f}'.format(rec_avg_loss / len(rec_cf_data_iter)),
                "joint_avg_loss": '{:.4f}'.format(joint_avg_loss / len(rec_cf_data_iter)),
                "cl_avg_loss":    '{:.4f}'.format(
                    cl_sum_avg_loss / (len(rec_cf_data_iter) * self.total_augmentaion_pairs)
                ),
            }
            for j, cl_individual_avg_loss in enumerate(cl_individual_avg_losses):
                post_fix[f'cl_pair_{j}_loss'] = '{:.4f}'.format(
                    cl_individual_avg_loss / len(rec_cf_data_iter)
                )

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))
            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            rec_data_iter = tqdm(
                enumerate(dataloader),
                desc="Recommendation EP_%s:%d" % (str_code, epoch),
                total=len(dataloader),
                bar_format="{l_bar}{r_bar}"
            )
            self.model.eval()
            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.transformer_encoder(input_ids)
                    recommend_output = recommend_output[:, -1, :]
                    rating_pred      = self.predict_full(recommend_output)
                    rating_pred      = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    ind             = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind         = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
                    if i == 0:
                        pred_list   = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list   = np.append(pred_list,   batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items   = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]
                    test_logits      = self.predict_sample(recommend_output, test_neg_items)
                    test_logits      = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                return self.get_sample_scores(epoch, pred_list)