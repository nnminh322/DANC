import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

def get_data_loader_BERT(config, data, shuffle = False, drop_last = False, batch_size = None):
    if batch_size == None:
        batch = min(config.batch_size, len(data))
    else:
        batch = min(batch_size, len(data))
    dataset = BERTDataset(data, config)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader

class BERTDataset(Dataset):    
    def __init__(self, data, config):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], idx)

    def collate_fn(self, data):
        batch_instance = {'ids': [], 'mask': []} 
        batch_label = []
        batch_idx = []

        batch_label = torch.tensor([item[0]['relation'] for item in data])
        batch_instance['ids'] = torch.tensor([item[0]['ids'] for item in data])
        batch_instance['mask'] = torch.tensor([item[0]['mask'] for item in data])
        batch_idx = torch.tensor([item[1] for item in data])
        
        return batch_instance, batch_label, batch_idx


def get_data_loader_BERTLLM(config, data, shuffle = False, drop_last = False, batch_size = None):
    if batch_size == None:
        batch = min(config.batch_size, len(data))
    else:
        batch = min(batch_size, len(data))
    dataset = BERTLLMDataset(data, config)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader


class BERTLLMDataset(Dataset):    
    def __init__(self, data, config):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], idx)

    # def collate_fn(self, data):
    #     # In ra data để kiểm tra cấu trúc
    #     print(data)
    #     batch_instance = {}
    #     batch_instance['input'] = [item[0]['input'] for item in data]
    #     # Thực hiện các thao tác khác nếu cần
    #     return batch_instance

    def collate_fn(self, data):
        # print(data)
        # print(f'config: {vars(self.config)}')
        tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)
        batch_instance = {'input': [],'ids': [], 'mask': []} 
        batch_label = []
        batch_idx = []

        batch_label = torch.tensor([item[0]['relation'] for item in data])
        batch_instance['ids'] = torch.tensor([item[0]['ids'] for item in data])
        batch_instance['mask'] = torch.tensor([item[0]['mask'] for item in data])
        # batch_instance['input'] = [item[0]['index'] for item in data]
        input_ids = [item[0]['ids'] for item in data]
        batch_instance['input'] = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]


        batch_idx = torch.tensor([item[1] for item in data])
        
        return batch_instance, batch_label, batch_idx
    
