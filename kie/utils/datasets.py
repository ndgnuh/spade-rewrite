from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, index_file):
        super().__init__()
