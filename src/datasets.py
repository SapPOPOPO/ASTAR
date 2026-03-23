import random
import torch
from torch.utils.data import Dataset
from data_augmentation import Crop, Mask, Reorder, Substitute, Insert, Random, CombinatorialEnumerate, RRandom
from utils import neg_sample, nCr
import copy
import numpy as np
from tqdm import tqdm

class RecWithContrastiveLearningDataset(Dataset):
    '''
    A dataset that gives recommendation task and a pair of contrastive learning tasks
    '''
    def __init__(self, args, user_seq, test_neg_items=None, data_type='train', 
                similarity_model_type='none'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

        if similarity_model_type=='offline':
            self.similarity_model = args.offline_similarity_model
        elif similarity_model_type=='online':
            self.similarity_model = args.online_similarity_model
        elif similarity_model_type=='hybrid':
            self.similarity_model = [args.offline_similarity_model, args.online_similarity_model]
        elif similarity_model_type=='none':
            self.similarity_model = None

        print("Similarity Model Type:", similarity_model_type)

        self.augmentations = {'crop': Crop(tao=args.tao),
                              'mask': Mask(gamma=args.gamma, mask_id=args.mask_id),
                              'reorder': Reorder(beta=args.beta),
                              'substitute': Substitute(self.similarity_model,
                                                substitute_rate=args.substitute_rate),
                              'insert': Insert(self.similarity_model, 
                                               insert_rate=args.insert_rate,
                                               max_insert_num_per_pos=args.max_insert_num_per_pos),
                              'random': RRandom(tao=args.tao, gamma=args.gamma, 
                                                beta=args.beta, item_similarity_model=self.similarity_model,
                                                insert_rate=args.insert_rate, 
                                                max_insert_num_per_pos=args.max_insert_num_per_pos,
                                                substitute_rate=args.substitute_rate,
                                                augment_threshold=self.args.augment_threshold,
                                                augment_type_for_short=self.args.augment_type_for_short,
                                                mask_id=args.mask_id),
                              'combinatorial_enumerate': CombinatorialEnumerate(tao=args.tao, gamma=args.gamma, 
                                                beta=args.beta, item_similarity_model=self.similarity_model,
                                                insert_rate=args.insert_rate, 
                                                max_insert_num_per_pos=args.max_insert_num_per_pos,
                                                substitute_rate=args.substitute_rate, n_views=args.n_views)
                            }

        if self.args.base_augment_type not in self.augmentations:
            raise ValueError(f"augmentation type: '{self.args.base_augment_type}' is invalided")
        print(f"Creating Contrastive Learning Dataset using '{self.args.base_augment_type}' data augmentation")

        # Define augmentation method
        self.base_transform = self.augmentations[self.args.base_augment_type]

        self.n_views = self.args.n_views

    def _one_pair_data_augmentation(self, input_ids):
        '''
        provides two positive samples given one sequence
        '''
        augmented_seqs = []
        aug_list = []
        length = len(input_ids)
        for i in range(2):
            augmented_input_ids = self.base_transform(input_ids[-self.max_len:])
            pad_len = self.max_len - len(augmented_input_ids)
            augmented_input_ids = [0] * pad_len + augmented_input_ids
            augmented_input_ids = augmented_input_ids[-self.max_len:]

            assert len(augmented_input_ids) == self.max_len
            aug_list.append(augmented_input_ids[-length:])
            cur_tensors = (
                torch.tensor(augmented_input_ids, dtype=torch.long)
            )
            augmented_seqs.append(cur_tensors) # We append the current tensors to the augmented_seqs list

        similarity = np.array(aug_list[0] == aug_list[1]).mean()
        return augmented_seqs, similarity # We return the augmented_seqs list

    def _metrics_for_sequence(self, aug_list):
        '''
        Given a list containing two augmented sequences
        (each of shape [seq_len] or a python list of ints),
        returns three ratios that describe how the two views align
        with respect to masked positions (mask_token == 0).

        Returns
        -------
        same_masked_ratio   : float  # both views masked at the same position
        same_unmasked_ratio : float  # both views un-masked at the same position
        masked_ratio        : float  # average fraction of positions that are masked
                            (averaged over the two views)

        The ratios are computed **only on the overlapping suffix** of the two
        sequences (right-aligned, as is conventional for rec-sys sequences).
        '''
    
        seq1 = torch.as_tensor(aug_list[0], dtype=torch.long, device=self.args.device)
        seq2 = torch.as_tensor(aug_list[1], dtype=torch.long, device=self.args.device)


        len1, len2 = seq1.size(0), seq2.size(0)
        min_len = min(len1, len2)

        # Take the *most recent* min_len items from each view
        s1 = seq1[-min_len:]      # shape [min_len]
        s2 = seq2[-min_len:]      # shape [min_len]

        mask_token = 0
        mask1 = (s1 == mask_token)      # [min_len] bool
        mask2 = (s2 == mask_token)      # [min_len] bool

        both_masked   = mask1 & mask2
        both_unmasked = (~mask1) & (~mask2)

        same_masked_ratio   = both_masked.float().mean().item()      # P(both masked)
        same_unmasked_ratio = both_unmasked.float().mean().item()    # P(both un-masked)
        masked_ratio        = (mask1.float().mean() + mask2.float().mean()).item() / 2.0

        return [same_masked_ratio, same_unmasked_ratio, masked_ratio]

    def _mask_from_prob(self, input_ids, prob, mask_token=0):
        """
        mask from the probability
        """
        augmented_seqs = []
        original_seq = input_ids[-self.max_len:]
        length = len(original_seq)
        pad_len = self.max_len - length
        original_seq = [0] * pad_len + original_seq
        masked_count = 0.0

        aug_list = []
        try:
            prob_np = prob.numpy()
            # Pad prob_np to match self.max_len (pads with 0.0 at the beginning)
            prob_padded = np.zeros(self.max_len)
            prob_padded[pad_len:] = prob_np[:length]  # Align with actual items
            for l in range(2):
                seq = original_seq.copy()  # Copy to avoid in-place accumulation
                mask_pos = (np.random.random(self.max_len) < prob_padded).astype(int)
                for i in range(self.max_len):
                    if mask_pos[i]:
                        seq[i] = mask_token
                        masked_count += 1.0 / length  # Use 1.0 for float division

                seq = seq[-length:]
                seq = self.base_transform(seq)
                aug_list.append(seq)

                pad_len = self.max_len - len(seq)
                seq = [0] * pad_len + seq
                seq = seq[-self.max_len:]

                cur_tensors = torch.tensor(seq, dtype=torch.long)
                augmented_seqs.append(cur_tensors)

            # Compute similarity on the unpadded parts (ignore padding)
            metrics = self._metrics_for_sequence(aug_list)

        except Exception as e:
            print(f"Error in augmenter: {e}")
            print("input_ids:", input_ids)
            print("prob shape:", prob.shape if hasattr(prob, 'shape') else "N/A")
            print('augmenter failed')
            # Fallback: return unpadded or dummy
            metrics = []

        return augmented_seqs, masked_count / 2, metrics
    
    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):        
        copied_input_ids = copy.deepcopy(input_ids)
        seq_set = set(items)

        target_neg = []
        for _ in copied_input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(copied_input_ids)
        copied_input_ids = [0] * pad_len + copied_input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        copied_input_ids = copied_input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(copied_input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items

            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long), # user_id for testing
                torch.tensor(copied_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long), # The answer is used for testing and validation
            )
        else:
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(copied_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long), # The answer is used for testing and validation
            )

        return cur_rec_tensors

    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio*len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k = insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size-2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size-2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0] # no use
        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            items_with_noise = self._add_noise_interactions(items)
            input_ids = items_with_noise[:-1]
            target_pos = items_with_noise[1:]
            answer = [items_with_noise[-1]]
            
        if self.data_type == "train":
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            cf_tensors_list = []

            total_augmentaion_pairs = nCr(self.n_views, 2)
            for i in range(total_augmentaion_pairs):
                aug_seq, similarity = self._one_pair_data_augmentation(input_ids)
                cf_tensors_list.append([aug_seq, 0, similarity])

            return (cur_rec_tensors, cf_tensors_list)

        elif self.data_type == 'valid':
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, \
                                target_pos, answer)
            return cur_rec_tensors
        
        else:
            cur_rec_tensors = self._data_sample_rec_task(user_id, items_with_noise, input_ids, \
                                target_pos, answer)
            return cur_rec_tensors

    def __len__(self):
        '''
        consider n_view of a single sequence as one sample
        '''
        return len(self.user_seq)