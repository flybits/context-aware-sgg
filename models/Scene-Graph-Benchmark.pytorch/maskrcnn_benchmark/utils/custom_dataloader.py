import torch.utils.data


class CustomSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.dataset = dataset  # Store the original dataset

    def get_img_info(self, idx):
        # Access the get_img_info method of the original dataset
        real_idx = self.indices[idx]
        return self.dataset.get_img_info(real_idx)
