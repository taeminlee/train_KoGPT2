import argparse
import logging
import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
from utils import get_tokenizer, download
from model.torch_gpt2 import GPT2Config, GPT2DoubleHeadsModel, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("--cuda", default="cuda", type=str,
                    help="set gpu")
parser.add_argument("--train_data_file", default="dataset/ratings_train.txt", type=str,
                    help="train file path")
parser.add_argument("--test_data_file", default="dataset/ratings_test.txt", type=str,
                    help="test file path")
parser.add_argument("--model_name_or_path", default="kogpt2", type=str,
                    help="The huggingface transformer model checkpoint for weights initialization."
                         "If you want to use SKT-KoGPT2, leave as default.")

args = parser.parse_args()

args.n_gpu = torch.cuda.device_count()
device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)


tokenizer_path = get_tokenizer()

tokenizer = {
    'url':
    'https://kobert.blob.core.windows.net/models/kogpt2/tokenizer/kogpt2_news_wiki_ko_cased_818bfa919d.spiece',
    'fname': 'kogpt2_news_wiki_ko_cased_818bfa919d.spiece',
    'chksum': '818bfa919d'
}

pytorch_kogpt2 = {
    'url':
    'https://kobert.blob.core.windows.net/models/kogpt2/pytorch/pytorch_kogpt2_676e9bcfa7.params',
    'fname': 'pytorch_kogpt2_676e9bcfa7.params',
    'chksum': '676e9bcfa7'
}

kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000,
    "activation_function": "gelu"
}

def get_kogpt2_vocab(cachedir='~/kogpt2/'):
    # download vocab
    vocab_info = tokenizer
    vocab_path = download(vocab_info['url'],
                          vocab_info['fname'],
                          vocab_info['chksum'],
                          cachedir=cachedir)
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                         mask_token=None,
                                                         sep_token=None,
                                                         cls_token=None,
                                                         unknown_token='<unk>',
                                                         padding_token='<pad>',
                                                         bos_token='<s>',
                                                         eos_token='</s>')
    return vocab


def get_kogpt2_model(model_name_or_path, ctx="cuda", cachedir='~/kogpt2/'):
    if model_name_or_path == "kogpt2":
        # download model
        model_info = pytorch_kogpt2
        model_path = download(model_info['url'],
                              model_info['fname'],
                              model_info['chksum'],
                              cachedir=cachedir)
        config = GPT2Config.from_dict(kogpt2_config)
        model = GPT2LMHeadModel(config=config)
        model.load_state_dict(torch.load(model_path), strict=False)

    else:
        config = GPT2Config.from_pretrained(args.model_name_or_path)
        model = GPT2LMHeadModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config)

    device = torch.device(ctx)
    model.to(device)
    model.train()

    return model, config


vocab = get_kogpt2_vocab()
model, config = get_kogpt2_model(args.model_name_or_path)

dataset_train = nlp.data.TSVDataset(args.train_data_file, field_indices=[1,2],
                                    num_discard_samples=1)
dataset_test = nlp.data.TSVDataset(args.test_data_file, field_indices=[1,2],
                                   num_discard_samples=1)


tok = nlp.data.BERTSPTokenizer(tokenizer_path, vocab, lower=False)

class GPT2Dataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, gpt2_tokenizer,
                 max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            gpt2_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

### Setting parameters ###
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

data_train = GPT2Dataset(dataset_train, 0, 1, tok, max_len, True, False)
data_test = GPT2Dataset(dataset_test, 0, 1, tok, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                                               num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size,
                                               num_workers=5)

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        last_token_tensor = hidden_states[:,-1]
        pooled_output = self.dense(last_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class GPT2Classifier(nn.Module):
    def __init__(self, gpt2, hidden_size=768, num_classes=2,
                 dr_rate=None, params=None):
        super(GPT2Classifier, self).__init__()
        self.gpt2 = gpt2
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        self.pooler = BertPooler(config)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        _, __ = self.gpt2(input_ids=token_ids, token_type_ids=segment_ids.long())

        __pooler = self.pooler(__)

        if self.dr_rate:
            out = self.dropout(__pooler)

        return self.classifier(out)


model = GPT2Classifier(model, dr_rate=0.5).to(device)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params':[p for n, p in model.named_parameters()
               if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
    {'params':[p for n, p in model.named_parameters()
               if any(nd in n for nd in no_decay)], 'weight_decay':0.0}]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X,1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
