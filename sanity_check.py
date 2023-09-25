import torch
from bert import BertModel
sanity_data = torch.load("./sanity_check.data")
# text_batch = ["hello world", "hello neural network for NLP"]
# tokenizer here
sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                         [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])
att_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],[1, 1, 1, 1, 1, 1, 1, 1]])

# load our model
bert = BertModel.from_pretrained('bert-base-uncased')
outputs = bert(sent_ids, att_mask)
att_mask = att_mask.unsqueeze(-1)
outputs['last_hidden_state'] = outputs['last_hidden_state'] * att_mask
sanity_data['last_hidden_state'] = sanity_data['last_hidden_state'] * att_mask

for k in ['last_hidden_state', 'pooler_output']:
    assert torch.allclose(outputs[k], sanity_data[k], atol=1e-5, rtol=1e-3)
print("Your BERT implementation is correct!")


from bert import BertSelfAttention
import torch

# class Config:
#     def __init__(self):
#         self.hidden_size = 512
#         self.num_attention_heads = 8
#
# config = Config()
# attention_module = BertSelfAttention(config)
# # Sample hidden states
# hidden_states = torch.randn(3, 7, config.hidden_size)
#
# # Sample attention mask (with padding for the second and third sequences)
# attention_mask = torch.tensor([
#     [[[0, 0, 0, 0, 0, 0, 0]]],
#     [[[0, 0, 0, 0, 0, 0, -1e9]]],  # Last token is padding
#     [[[0, 0, 0, 0, 0, -1e9, -1e9]]]  # Last two tokens are padding
# ])
# output = attention_module(hidden_states, attention_mask)
# print(output.shape)  # Expected: torch.Size([3, 7, 512])