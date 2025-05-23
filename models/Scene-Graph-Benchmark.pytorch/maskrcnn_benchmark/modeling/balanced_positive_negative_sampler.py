# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx



# # # SINGLE GPU IMPLEMENTATION

# class BalancedPositiveNegativeSampler(object):
#     def __init__(self, batch_size_per_image, positive_fraction):
#         self.batch_size_per_image = batch_size_per_image
#         self.positive_fraction = positive_fraction

#     def __call__(self, matched_idxs):
#         pos_idx = []
#         neg_idx = []
#         for matched_idxs_per_image in matched_idxs:
#             positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
#             negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

#             num_pos = int(self.batch_size_per_image * self.positive_fraction)
#             num_pos = min(positive.numel(), num_pos)
#             num_neg = self.batch_size_per_image - num_pos
#             num_neg = min(negative.numel(), num_neg)

#             device = matched_idxs_per_image.device
#             if device.type == 'cuda' and device.index is None:
#                 device = torch.device('cuda:0')

#             print(f"positive.numel(): {positive.numel()}, num_pos: {num_pos}")
#             print(f"negative.numel(): {negative.numel()}, num_neg: {num_neg}")

#             # Randomly select positive examples
#             if positive.numel() > 0 and num_pos > 0:
#                 perm1 = torch.randperm(positive.numel(), device=device)[:num_pos]
#                 pos_idx_per_image = positive[perm1]
#             else:
#                 pos_idx_per_image = torch.tensor([], dtype=torch.long, device=device)

#             # Randomly select negative examples
#             if negative.numel() > 0 and num_neg > 0:
#                 perm2 = torch.randperm(negative.numel(), device=device)[:num_neg]
#                 neg_idx_per_image = negative[perm2]
#             else:
#                 neg_idx_per_image = torch.tensor([], dtype=torch.long, device=device)

#             # Create binary mask from indices
#             pos_idx_per_image_mask = torch.zeros_like(
#                 matched_idxs_per_image, dtype=torch.uint8
#             )
#             neg_idx_per_image_mask = torch.zeros_like(
#                 matched_idxs_per_image, dtype=torch.uint8
#             )
#             if pos_idx_per_image.numel() > 0:
#                 pos_idx_per_image_mask[pos_idx_per_image] = 1
#             if neg_idx_per_image.numel() > 0:
#                 neg_idx_per_image_mask[neg_idx_per_image] = 1

#             pos_idx.append(pos_idx_per_image_mask)
#             neg_idx.append(neg_idx_per_image_mask)

#         return pos_idx, neg_idx




# # debug version for small toy dataset
# class BalancedPositiveNegativeSampler(object):
#     """
#     This class samples batches, ensuring that they contain a fixed proportion of positives
#     """

#     def __init__(self, batch_size_per_image, positive_fraction):
#         """
#         Arguments:
#             batch_size_per_image (int): number of elements to be selected per image
#             positive_fraction (float): percentage of positive elements per batch
#         """
#         self.batch_size_per_image = batch_size_per_image
#         self.positive_fraction = positive_fraction

#     def __call__(self, matched_idxs):
#         """
#         Arguments:
#             matched_idxs: list of tensors containing -1, 0 or positive values.
#                 Each tensor corresponds to a specific image.
#                 -1 values are ignored, 0 are considered as negatives and > 0 as positives.

#         Returns:
#             pos_idx (list[tensor])
#             neg_idx (list[tensor])

#         Returns two lists of binary masks for each image.
#         The first list contains the positive elements that were selected,
#         and the second list the negative examples.
#         """
#         pos_idx = []
#         neg_idx = []
#         for matched_idxs_per_image in matched_idxs:
#             positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
#             negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

#             # Instead of sampling, use all available positives and negatives
#             pos_idx_per_image_mask = torch.zeros_like(
#                 matched_idxs_per_image, dtype=torch.uint8
#             )
#             neg_idx_per_image_mask = torch.zeros_like(
#                 matched_idxs_per_image, dtype=torch.uint8
#             )

#             if positive.numel() > 0:
#                 pos_idx_per_image_mask[positive] = 1
#             if negative.numel() > 0:
#                 neg_idx_per_image_mask[negative] = 1

#             pos_idx.append(pos_idx_per_image_mask)
#             neg_idx.append(neg_idx_per_image_mask)

#         return pos_idx, neg_idx
