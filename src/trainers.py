import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from datasets import RecWithContrastiveLearningDataset
from modules import NCELoss, NTXent, MultiPositiveInfoNCE
from utils import recall_at_k, ndcg_k, get_metric, get_user_seqs, nCr
import torch.nn.functional as F
from data_augmentation import Reeorder
from diagnose import GANDiagnostics


class Trainer:
    def __init__(self, model, adv_model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args,
                 diagnostics=None):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        self.adv_model = adv_model
        self.diag = diagnostics

        self.projection = nn.Sequential(
            nn.Linear(self.args.max_seq_length * self.args.hidden_size, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.args.hidden_size, bias=True)
        )

        if self.cuda_condition:
            self.model.cuda()
            self.adv_model.cuda()
            self.projection.cuda()

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim_model = Adam(self.model.parameters(), lr=self.args.lr,
                                betas=betas, weight_decay=self.args.weight_decay)
        self.optim_adv = Adam(self.adv_model.parameters(), lr=self.args.lr,
                              betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        print("self.cf_criterion:", self.cf_criterion.__class__.__name__)

        self.target_rate = args.target_rate

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
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
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
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
        if seq_out.dim() == 2:
            seq_emb = seq_out
            pos_ids_flat = pos_ids.view(-1)
            neg_ids_flat = neg_ids.view(-1)
        else:
            seq_emb = seq_out.view(-1, self.args.hidden_size)
            pos_ids_flat = pos_ids.view(-1)
            neg_ids_flat = neg_ids.view(-1)

        pos_emb = self.model.item_embeddings(pos_ids_flat)
        neg_emb = self.model.item_embeddings(neg_ids_flat)

        pos_logits = torch.sum(pos_emb * seq_emb, -1)
        neg_logits = torch.sum(neg_emb * seq_emb, -1)
        istarget = (pos_ids_flat > 0).float()

        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)
        return test_logits

    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class AdvAugmentTrainer(Trainer):
    """
    Adversarial Training

    Loss Functions:
        - Recommender: L_B = L_rec_orig + gamma * L_rec_aug + lambda * L_contrast
        - Augmenter:   L_A = beta * L_rec_aug - alpha * L_contrast
                           + reg_weight * rate_penalty
                           + asym_weight * asymmetry_penalty
    """
    def __init__(self, model, adv_model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args,
                 diagnostics=None):
        super(AdvAugmentTrainer, self).__init__(
            model, adv_model,
            train_dataloader, eval_dataloader, test_dataloader,
            args, diagnostics
        )

        self.alpha        = 0.1
        self.beta_w       = 1.0
        self.gamma        = 0.2
        self.lambda_      = 0.2
        self.eta          = 0.1
        self.max_grad_norm = 1.0

    def _one_pair_contrastive_learning(self, inputs):
        cl_batch = torch.cat(inputs, dim=0).to(self.device)
        cl_sequence_output = self.model.transformer_encoder(cl_batch)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        batch_size = cl_batch.shape[0] // 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        return self.cf_criterion(cl_output_slice[0], cl_output_slice[1])

    def iteration(self, epoch, dataloader, full_sort=True, train=True):
        str_code = "train" if train else "test"

        if train:
            self.model.train()
            self.adv_model.train()

            metrics = {
                'B_rec_loss': 0.0, 'B_aug_loss': 0.0,
                'B_contrast_loss': 0.0, 'B_total_loss': 0.0,
                'A_rec_loss': 0.0, 'A_contrast_loss': 0.0,
                'A_entropy': 0.0, 'A_kl': 0.0, 'A_total_loss': 0.0,
                'avg_mask_rate1': 0.0, 'avg_mask_rate2': 0.0
            }

            dataloader_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches) in dataloader_iter:
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                seq_lengths = (input_ids > 0).float().sum(dim=1)
                current_tau = self.adv_model.tau

                # ── Step 1: Generate augmented sequences (no grad) ──────────
                with torch.no_grad():
                    aug_seq1, aug_seq2, probs1, probs2, masks1, masks2, pad_mask = \
                        self.adv_model(input_ids, tau=current_tau)

                mask_rate1 = (masks1 * pad_mask).sum(dim=1) / seq_lengths
                mask_rate2 = (masks2 * pad_mask).sum(dim=1) / seq_lengths

                # ── Step 2: Update recommender (B) ──────────────────────────
                seq_out_orig = self.model.transformer_encoder(input_ids)
                rec_loss_orig = self.cross_entropy(seq_out_orig, target_pos, target_neg)

                seq_out_aug1 = self.model.transformer_encoder(aug_seq1)
                seq_out_aug2 = self.model.transformer_encoder(aug_seq2)

                rec_loss_aug1 = self.cross_entropy(seq_out_aug1[:, -1, :], target_pos[:, -1:], target_neg[:, -1:])
                rec_loss_aug2 = self.cross_entropy(seq_out_aug2[:, -1, :], target_pos[:, -1:], target_neg[:, -1:])
                rec_loss_aug  = rec_loss_aug1 + rec_loss_aug2

                contrast_loss_B = self._one_pair_contrastive_learning([aug_seq1, aug_seq2])

                loss_B = rec_loss_orig + self.gamma * rec_loss_aug + self.lambda_ * contrast_loss_B

                self.optim_model.zero_grad()
                loss_B.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optim_model.step()

                # ── Step 3: Update augmenter (A) ────────────────────────────
                aug_seq1, aug_seq2, probs1, probs2, masks1, masks2, pad_mask = \
                    self.adv_model(input_ids, tau=current_tau)

                with torch.no_grad():
                    seq_out_aug1 = self.model.transformer_encoder(aug_seq1)
                    seq_out_aug2 = self.model.transformer_encoder(aug_seq2)

                rec_loss_aug1  = self.cross_entropy(seq_out_aug1[:, -1, :], target_pos[:, -1:], target_neg[:, -1:])
                rec_loss_aug2  = self.cross_entropy(seq_out_aug2[:, -1, :], target_pos[:, -1:], target_neg[:, -1:])
                rec_loss_aug_A = (rec_loss_aug1 + rec_loss_aug2) / 2

                contrast_loss_A = self._one_pair_contrastive_learning([aug_seq1, aug_seq2])
                entropy         = self.adv_model.compute_entropy(probs1, probs2, pad_mask)

                # Recompute rate from fresh probs (WITH grad) so rate_penalty
                # carries gradient back through probs → logits → augmenter.
                min_rate = 0.3
                max_rate = 0.7
                aug_rate1 = (probs1 * pad_mask).sum(dim=1) / seq_lengths.clamp(min=1)
                aug_rate2 = (probs2 * pad_mask).sum(dim=1) / seq_lengths.clamp(min=1)
                rate_penalty = (
                    torch.clamp(min_rate - aug_rate1, min=0) +
                    torch.clamp(aug_rate1 - max_rate, min=0) +
                    torch.clamp(min_rate - aug_rate2, min=0) +
                    torch.clamp(aug_rate2 - max_rate, min=0)
                ).mean()

                # Per-sequence asymmetry penalty (also differentiable via probs)
                asymmetry_penalty = torch.clamp(
                    torch.abs(aug_rate1 - aug_rate2) - 0.3, min=0
                ).mean()

                if epoch >= self.args.warmup_epochs:
                    loss_A = (self.beta_w  * rec_loss_aug_A
                            - self.alpha   * contrast_loss_A
                            - self.eta     * entropy
                            + self.args.reg_weight  * rate_penalty
                            + self.args.asym_weight * asymmetry_penalty)

                    self.optim_adv.zero_grad()
                    loss_A.backward()
                    torch.nn.utils.clip_grad_norm_(self.adv_model.parameters(), self.max_grad_norm)
                    self.optim_adv.step()
                else:
                    loss_A = torch.tensor(0.0)

                # ── Debug print ─────────────────────────────────────────────
                if i == 2 and epoch > 1:
                    print(probs1)
                    print(probs2)
                    print(aug_seq1)
                    print(aug_seq2)

                # ── Diagnostics (last batch, every 5 epochs) ────────────────
                if self.diag is not None and epoch % 5 == 0 and i == len(dataloader) - 1:
                    self.diag.run(
                        epoch=epoch,
                        masks1=masks1,
                        masks2=masks2,
                        pad_mask=pad_mask,
                        seq_lengths=seq_lengths,
                        loss_A=loss_A.item(),
                        loss_B=loss_B.item(),
                        augmenter=self.adv_model,
                        recommender=self.model,
                        aug_seq1=aug_seq1,
                        aug_seq2=aug_seq2
                    )

                # ── Metrics ─────────────────────────────────────────────────
                metrics['B_rec_loss']      += rec_loss_orig.item()
                metrics['B_aug_loss']      += rec_loss_aug.item()
                metrics['B_contrast_loss'] += contrast_loss_B.item()
                metrics['B_total_loss']    += loss_B.item()
                metrics['A_rec_loss']      += rec_loss_aug_A.item()
                metrics['A_contrast_loss'] += contrast_loss_A.item()
                metrics['A_entropy']       += entropy.item()
                metrics['A_total_loss']    += loss_A.item()
                metrics['avg_mask_rate1']  += mask_rate1.mean().item()
                metrics['avg_mask_rate2']  += mask_rate2.mean().item()

                if i % 10 == 0:
                    dataloader_iter.set_description(
                        f"Epoch {epoch}: B_Loss={loss_B.item():.4f}, "
                        f"A_Loss={loss_A.item():.4f}, "
                        f"MaskRate={mask_rate1.mean().item():.3f}"
                    )

            num_batches = len(dataloader)
            for key in metrics:
                metrics[key] /= num_batches

            self.adv_model.decay_tau()

            post_fix = {
                "epoch":      epoch,
                "tau":        f"{current_tau:.4f}",
                "B_rec":      f"{metrics['B_rec_loss']:.4f}",
                "B_aug":      f"{metrics['B_aug_loss']:.4f}",
                "B_contrast": f"{metrics['B_contrast_loss']:.4f}",
                "B_total":    f"{metrics['B_total_loss']:.4f}",
                "A_rec":      f"{metrics['A_rec_loss']:.4f}",
                "A_contrast": f"{metrics['A_contrast_loss']:.4f}",
                "A_entropy":  f"{metrics['A_entropy']:.4f}",
                "A_kl":       f"{metrics['A_kl']:.4f}",
                "A_total":    f"{metrics['A_total_loss']:.4f}",
                "mask_rate1": f"{metrics['avg_mask_rate1']:.4f}",
                "mask_rate2": f"{metrics['avg_mask_rate2']:.4f}",
            }

            print(f"Training Metrics: {post_fix}")
            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

            return metrics

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
                    rating_pred = self.predict_full(recommend_output)
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
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
                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
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
                 args,
                 diagnostics=None):
        super(CoSeRecTrainer, self).__init__(
            model, adv_model,
            train_dataloader, eval_dataloader, test_dataloader,
            args, diagnostics
        )

    def _one_pair_contrastive_learning(self, inputs):
        cl_batch = torch.cat(inputs, dim=0).to(self.device)
        cl_sequence_output  = self.model.transformer_encoder(cl_batch)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        batch_size = cl_batch.shape[0] // 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        return self.cf_criterion(cl_output_slice[0], cl_output_slice[1])

    def iteration(self, epoch, dataloader, full_sort=True, train=True):
        str_code = "train" if train else "test"

        if train:
            self.model.train()
            self.adv_model.train()

            adv_avg_loss = 0.0
            adv_avg_loss_rec_loss = 0.0
            adv_avg_loss_similarity_penalty = 0.0
            adv_avg_loss_sparsity_bonus = 0.0
            rec_avg_loss = 0.0
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0
            total_avg_prob1 = 0.0
            total_avg_prob2 = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches) in rec_cf_data_iter:
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                seq_lengths = (input_ids > 0).float().sum(dim=1)
                tau = max([1, self.args.mask_tau * (0.99 ** epoch)])

                # ── Augmenter update ────────────────────────────────────────
                modified_seq1, modified_seq2, _, _, _, _, _ = self.adv_model(input_ids, tau=tau)

                with torch.no_grad():
                    modified_sequence_output1 = self.model.transformer_encoder(modified_seq1)
                    modified_sequence_output2 = self.model.transformer_encoder(modified_seq2)

                modified_rec_loss1 = self.cross_entropy(modified_sequence_output1[:, -1, :], target_pos[:, -1:], target_neg[:, -1:])
                modified_rec_loss2 = self.cross_entropy(modified_sequence_output2[:, -1, :], target_pos[:, -1:], target_neg[:, -1:])
                modified_rec_loss  = (modified_rec_loss1 + modified_rec_loss2) / 2

                masks1, masks2, probs1, probs2, _ = self.adv_model.sample_masks(
                    input_ids, tau=tau, hard=False, return_probs=True)
                avg_prob1 = masks1.sum(dim=1) / seq_lengths.clamp(min=1)
                avg_prob2 = masks2.sum(dim=1) / seq_lengths.clamp(min=1)

                similarity_penalty = self._one_pair_contrastive_learning([modified_seq1, modified_seq2])
                reg_loss = (((avg_prob1 - self.target_rate) ** 2).mean() +
                            ((avg_prob2 - self.target_rate) ** 2).mean())

                if i == 2 and epoch >= self.args.warmup_epochs:
                    print(masks1, masks2, modified_seq1, modified_seq2)

                if epoch >= self.args.warmup_epochs:
                    adv_model_loss = (modified_rec_loss
                                    - self.args.penalty_weight * similarity_penalty
                                    + self.args.reg_weight * reg_loss)
                    self.optim_adv.zero_grad()
                    adv_model_loss.backward()
                    self.optim_adv.step()
                else:
                    adv_model_loss = torch.tensor(0.0)

                # ── Recommender update ──────────────────────────────────────
                sequence_output = self.model.transformer_encoder(input_ids)
                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)

                with torch.no_grad():
                    modified_seq1, modified_seq2, _, _ = self.adv_model.modify_sequence(input_ids, tau=tau)

                modified_sequence_output1 = self.model.transformer_encoder(modified_seq1)
                modified_sequence_output2 = self.model.transformer_encoder(modified_seq2)

                adv_loss   = self._one_pair_contrastive_learning([modified_seq1, modified_seq2])
                check_loss = (self.cross_entropy(sequence_output[:, -1, :], target_pos[:, -1:], target_neg[:, -1:]) +
                              self.cross_entropy(sequence_output[:, -1, :], target_pos[:, -1:], target_neg[:, -1:]))

                joint_loss = (self.args.rec_weight   * rec_loss +
                              self.args.cf_weight    * adv_loss +
                              self.args.check_weight * check_loss)

                self.optim_model.zero_grad()
                joint_loss.backward()
                self.optim_model.step()

                # ── Diagnostics ─────────────────────────────────────────────
                if self.diag is not None and epoch % 5 == 0 and i == len(dataloader) - 1:
                    pad_mask = (input_ids > 0)
                    self.diag.run(
                        epoch=epoch,
                        masks1=masks1.long(),
                        masks2=masks2.long(),
                        pad_mask=pad_mask,
                        seq_lengths=seq_lengths,
                        loss_A=adv_model_loss.item(),
                        loss_B=joint_loss.item(),
                        augmenter=self.adv_model,
                        recommender=self.model,
                        aug_seq1=modified_seq1,
                        aug_seq2=modified_seq2
                    )

                adv_avg_loss                    += adv_model_loss.item()
                adv_avg_loss_rec_loss           += modified_rec_loss.item()
                adv_avg_loss_similarity_penalty += similarity_penalty.item()
                adv_avg_loss_sparsity_bonus     += reg_loss.item()
                rec_avg_loss                    += rec_loss.item()
                cl_sum_avg_loss                 += adv_loss.item()
                joint_avg_loss                  += joint_loss.item()
                total_avg_prob1                 += avg_prob1.mean().item()
                total_avg_prob2                 += avg_prob2.mean().item()

            post_fix = {
                "epoch":                          epoch,
                "rec_avg_loss":                   '{:.4f}'.format(rec_avg_loss / len(rec_cf_data_iter)),
                "joint_avg_loss":                 '{:.4f}'.format(joint_avg_loss / len(rec_cf_data_iter)),
                "cl_avg_loss":                    '{:.4f}'.format(cl_sum_avg_loss / len(rec_cf_data_iter)),
                "adv_avg_loss":                   '{:.4f}'.format(adv_avg_loss / len(rec_cf_data_iter)),
                "adv_avg_loss_rec_loss":          '{:.4f}'.format(adv_avg_loss_rec_loss / len(rec_cf_data_iter)),
                "adv_avg_loss_similarity_penalty":'{:.4f}'.format(adv_avg_loss_similarity_penalty / len(rec_cf_data_iter)),
                "adv_avg_loss_sparsity_bonus":    '{:.4f}'.format(adv_avg_loss_sparsity_bonus / len(rec_cf_data_iter)),
                "avg_prob1":                      '{:.4f}'.format(total_avg_prob1 / len(rec_cf_data_iter)),
                "avg_prob2":                      '{:.4f}'.format(total_avg_prob2 / len(rec_cf_data_iter)),
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
                    rating_pred = self.predict_full(recommend_output)
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
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
                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                return self.get_sample_scores(epoch, pred_list)

class ASTARv2Trainer(Trainer):
    """Trainer for the ASTARv2 dual-view transport ablation.

    Training objective
    ------------------
    Recommender (B):
        L_B = L_rec(orig) + cl_weight * L_CL(view1, view2)

    Augmenter (A), after warmup:
        L_A = transport_reg_weight * L_reg(T1, T2)

    The recommender is encouraged to produce view-invariant representations
    via the view-to-view contrastive loss.  The augmenter is trained to
    produce diverse and balanced transport distributions via entropy and
    usage-balance regularisation (the only terms that carry gradient back
    through the discrete argmax selection).
    """

    def __init__(self, model, adv_model,
                 train_dataloader, eval_dataloader, test_dataloader,
                 args, diagnostics=None):
        super().__init__(model, adv_model,
                         train_dataloader, eval_dataloader, test_dataloader,
                         args, diagnostics)
        self.cl_weight            = getattr(args, 'v2_cl_weight', 0.2)
        self.transport_reg_weight = getattr(args, 'transport_reg_weight', 0.1)
        self.max_grad_norm        = 1.0

    def iteration(self, epoch, dataloader, full_sort=True, train=True):
        str_code = "train" if train else "test"

        if train:
            self.model.train()
            self.adv_model.train()

            metrics = {
                'rec_loss': 0.0, 'cl_loss': 0.0, 'total_B': 0.0,
                'reg_loss': 0.0, 'total_A': 0.0,
            }

            dataloader_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches) in dataloader_iter:
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                # ── Generate two views (no grad through augmenter here) ──────
                with torch.no_grad():
                    aug_seq1, aug_seq2, _ = self.adv_model.generate_views(input_ids)

                # ── Update Recommender (B) ───────────────────────────────────
                seq_out = self.model.transformer_encoder(input_ids)
                rec_loss = self.cross_entropy(seq_out, target_pos, target_neg)

                seq_out1 = self.model.transformer_encoder(aug_seq1)
                seq_out2 = self.model.transformer_encoder(aug_seq2)

                # View-to-view contrastive loss: recommender wants views to align
                cl_loss = self.cf_criterion(seq_out1[:, -1, :], seq_out2[:, -1, :])

                loss_B = rec_loss + self.cl_weight * cl_loss

                self.optim_model.zero_grad()
                loss_B.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optim_model.step()

                # ── Update Augmenter (A), after warmup ───────────────────────
                if epoch >= self.args.warmup_epochs:
                    # Re-run with grad to get differentiable reg_loss
                    _, _, reg_loss = self.adv_model.generate_views(input_ids)

                    loss_A = self.transport_reg_weight * reg_loss

                    self.optim_adv.zero_grad()
                    loss_A.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.adv_model.parameters(), self.max_grad_norm)
                    self.optim_adv.step()
                else:
                    loss_A = torch.tensor(0.0)
                    reg_loss = torch.tensor(0.0)

                # Decay tau once per step (mirrors ASTARAugmenter.step_tau)
                self.adv_model.step_tau()

                metrics['rec_loss'] += rec_loss.item()
                metrics['cl_loss']  += cl_loss.item()
                metrics['total_B']  += loss_B.item()
                metrics['reg_loss'] += reg_loss.item()
                metrics['total_A']  += loss_A.item()

                if i % 10 == 0:
                    dataloader_iter.set_description(
                        f"Epoch {epoch}: B={loss_B.item():.4f}, "
                        f"A={loss_A.item():.4f}, CL={cl_loss.item():.4f}"
                    )

            n = len(dataloader)
            for k in metrics:
                metrics[k] /= n

            post_fix = {
                "epoch":    epoch,
                "rec_loss": f"{metrics['rec_loss']:.4f}",
                "cl_loss":  f"{metrics['cl_loss']:.4f}",
                "total_B":  f"{metrics['total_B']:.4f}",
                "reg_loss": f"{metrics['reg_loss']:.4f}",
                "total_A":  f"{metrics['total_A']:.4f}",
            }
            print(f"ASTARv2 Training Metrics: {post_fix}")
            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

            return metrics

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
                    rating_pred = self.predict_full(recommend_output)
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
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
                    recommend_output = self.model.transformer_encoder(input_ids)
                    test_neg_items   = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]
                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                return self.get_sample_scores(epoch, pred_list)
