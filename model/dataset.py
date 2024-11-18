import torch
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from random import sample

# class CustomDataset(Dataset):
#     def __init__(self, src_list: list, src_att_list: list, src_seg_list: list = list(),
#                  trg_list: list = None, min_len: int = 4, src_max_len: int = 300):
#         # List setting
#         if src_seg_list == list():
#             src_seg_list = [[0] for _ in range(len(src_list))]
#         self.tensor_list = []

#         # Tensor list
#         for src, src_att, src_seg, trg in zip(src_list, src_att_list, src_seg_list, trg_list):
#             if min_len <= len(src) <= src_max_len:
#                 # Source tensor
#                 src_tensor = torch.zeros(src_max_len, dtype=torch.long)
#                 src_tensor[:len(src)] = torch.tensor(src, dtype=torch.long)
#                 src_att_tensor = torch.zeros(src_max_len, dtype=torch.long)
#                 src_att_tensor[:len(src_att)] = torch.tensor(src_att, dtype=torch.long)
#                 src_seg_tensor = torch.zeros(src_max_len, dtype=torch.long)
#                 src_seg_tensor[:len(src_seg)] = torch.tensor(src_seg, dtype=torch.long)
#                 # Target tensor
#                 trg_tensor = torch.tensor(trg, dtype=torch.float)
#                 #
#                 self.tensor_list.append((src_tensor, src_att_tensor, src_seg_tensor, trg_tensor))

#         self.num_data = len(self.tensor_list)

#     def __getitem__(self, index):
#         return self.tensor_list[index]

#     def __len__(self):
#         return self.num_data

class CustomDataset(Dataset):
    def __init__(self, tokenizer, src_list: list = list(), src_list2: list = None, trg_list: list = None, min_len: int = 4, src_max_len: int = 300):

        self.tokenizer = tokenizer
        self.src_tensor_list = list()
        self.src_tensor_list2 = list()
        self.trg_tensor_list = list()

        self.min_len = min_len
        self.src_max_len = src_max_len

        for src in src_list:
            # if min_len <= len(src):
            self.src_tensor_list.append(src)

        if src_list2 is not None:
            for src in src_list2:
                # if min_len <= len(src):
                self.src_tensor_list2.append(src)

        self.trg_tensor_list = trg_list

        self.num_data = len(self.src_tensor_list)

    def __getitem__(self, index):
        if len(self.src_tensor_list2) == 0:
            encoded_dict = \
            self.tokenizer(
                self.src_tensor_list[index],
                max_length=self.src_max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded_dict['input_ids'].squeeze(0)
            attention_mask = encoded_dict['attention_mask'].squeeze(0)
            if len(encoded_dict.keys()) == 3:
                token_type_ids = encoded_dict['token_type_ids'].squeeze(0)
            else:
                token_type_ids = encoded_dict['attention_mask'].squeeze(0)

        else:
            try:
                encoded_dict = \
                self.tokenizer(
                    self.src_tensor_list[index], self.src_tensor_list2[index],
                    max_length=self.src_max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = encoded_dict['input_ids'].squeeze(0)
                attention_mask = encoded_dict['attention_mask'].squeeze(0)
                if len(encoded_dict.keys()) == 3:
                    token_type_ids = encoded_dict['token_type_ids'].squeeze(0)
                else:
                    token_type_ids = encoded_dict['attention_mask'].squeeze(0)
            except IndexError:
                print(index)
                print(self.src_tensor_list[index])

        trg_tensor = torch.tensor(self.trg_tensor_list[index], dtype=torch.float)
        # print(input_ids)
        # print()
        # print(trg_tensor)
        return (input_ids, attention_mask, token_type_ids, trg_tensor)

    def __len__(self):
        return self.num_data
    
class CustomMaskingDataset(Dataset):
    def __init__(self, tokenizer, src_list: list = list(), min_len: int = 4, src_max_len: int = 300):

        self.tokenizer = tokenizer
        self.src_tensor_list = list()

        self.min_len = min_len
        self.src_max_len = src_max_len

        for src in src_list:
            # if min_len <= len(src):
            self.src_tensor_list.append(src)

        self.num_data = len(self.src_tensor_list)

    def __getitem__(self, index):
        encoded_dict = \
        self.tokenizer(
            self.src_tensor_list[index],
            max_length=self.src_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded_dict['input_ids'].squeeze(0)
        attention_mask = encoded_dict['attention_mask'].squeeze(0)
        if len(encoded_dict.keys()) == 3:
            token_type_ids = encoded_dict['token_type_ids'].squeeze(0)
        else:
            token_type_ids = encoded_dict['attention_mask'].squeeze(0)

        sep_token_ix = (input_ids==self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0].item()
        ix_list = list(range(1, sep_token_ix))
        masking_index = sample(ix_list, int((sep_token_ix-1) * 0.15))
        input_ids[masking_index] = self.tokenizer.mask_token_id
        masked_input_ids = input_ids

        trg_tensor = torch.LongTensor([x if x in masking_index else -100 for x in range(self.src_max_len)])
        # print(input_ids)
        # print()
        # print(trg_tensor)
        return (masked_input_ids, attention_mask, token_type_ids, trg_tensor)

    def __len__(self):
        return self.num_data
    
class CustomSeq2seqDataset(Dataset):
    def __init__(self, tokenizer, src_list: list = list(), trg_list: list = None, min_len: int = 4, src_max_len: int = 300, trg_max_len: int = 300):

        self.tokenizer = tokenizer
        self.src_tensor_list = list()
        self.trg_tensor_list = list()

        self.min_len = min_len
        self.src_max_len = src_max_len
        self.trg_max_len = src_max_len

        for src in src_list:
            # if min_len <= len(src):
            self.src_tensor_list.append(src)

        self.trg_tensor_list = trg_list

        self.num_data = len(self.src_tensor_list)

    def __getitem__(self, index):
        encoded_dict = \
        self.tokenizer(
            self.src_tensor_list[index],
            max_length=self.src_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded_dict['input_ids'].squeeze(0)
        attention_mask = encoded_dict['attention_mask'].squeeze(0)

        encoded_dict_trg = \
        self.tokenizer(
            self.trg_tensor_list[index],
            max_length=self.trg_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        trg_tensor = encoded_dict_trg['input_ids'].squeeze(0)

        # print(input_ids)
        # print()
        # print(trg_tensor)
        return (input_ids, attention_mask, trg_tensor)

    def __len__(self):
        return self.num_data