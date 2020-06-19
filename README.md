# train_KoGPT2
Pre-train and fine-tune transformer models including KoGPT2

## Requirements
pip install -r requirements.txt

* python >= 3.6
* torch == 1.4.0                                                                                                          
* mxnet == 1.6.0                                                                                                          
* gluonnlp == 0.9.1                                                                                                       
* transformers == 2.9.1                                                                                                   
* wandb == 0.8.36
* jupyter==1.0.0

# Pre-train transformer: 'run_language_modeling.py'
* Modified [Huggingface-language_modeling.py](https://github.com/huggingface/transformers/tree/master/examples/language-modeling) and [SKT-KoGPT2](https://github.com/SKT-AI/KoGPT2).
The modified code is compatible for both transformer tokenizer(AutoTokenizer) and gluonnlp tokenizer(BERTSPTokenizer).
It is also compatible for both transformer model(AutoModelWithLMHead) and pre-trained KoGPT2 model(GP2LMHeadModel).
You can use either Transformer class (LineByLineText, TextDataset) or gluonnlp class (TSVDataset) for pre-processing based on what your dataset looks like.

* SKT-AI에서 공개한 KoGPT2를 포함한 허깅페이스 기반 트랜스포머 모델들(BERT,GPT2 등)을 훈련할 수 있는 코드입니다.
트랜스포머 기반 토크나이저 뿐만 아니라 gluonnlp의 BERTSPTokenizer도 사용할 수 있도록 하였습니다.
데이터셋 형태에 따라 트랜스포머의 LinebyLineText 클래스와 TextDataset 클래스, 혹은 gluonnlp의 TSVDataset을 선택해서 사용하시면 됩니다.

## How to use
* Current arguments are set up for pre-training KoGPT2 with gluonnlp-BERTSPTokenizer. Change arguments such as 'model_name_or_path', 'config_name' and 'use_gluonnlp_tokenizer' if you want to use other tokenizer and model for training.

* 코드는 기본적으로 gluonnlp-BERTSPTokenizer로 KoGPT2를 추가 훈련하도록 맞춰져있습니다. 트랜스포머의 모델이나 토크나이저를 사용하고 싶은 경우 'model_name_or_path', 'config_name' and 'use_gluonnlp_tokenizer' 등의 인자를 변경해주세요. 

## Example 
* Additional pre-train KoGPT2 with [NSMC dataset](https://github.com/e9t/nsmc)

* NSMC 데이터셋(라벨 제외)으로 KoGPT2를 추가 학습한 경우의 예시입니다. 커맨드는 아래와 같습니다.

> python3 run_language_modeling.py --do_train --output_dir=train_output --train_data_file=dataset/ratings_train.txt --do_eval --eval_data_file=dataset/ratings_test.txt

## Generate
```python
import torch
import gluonnlp as nlp
from utils import get_tokenizer
from transformers import AutoModelWithLMHead, AutoConfig

tok_path = get_tokenizer()
config = AutoConfig.from_pretrained("train_output")
model = AutoModelWithLMHead.from_pretrained(
                "train_output",
                from_tf=bool(".ckpt" in "train_output"),
                config=config
            )
model.to(torch.device("cpu"))
model.eval()
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tok_path,
                                               mask_token=None,
                                               sep_token=None,
                                               cls_token=None,
                                               unknown_token='<unk>',
                                               padding_token='<pad>',
                                               bos_token='<s>',
                                               eos_token='</s>')

tok = nlp.data.SentencepieceTokenizer(tok_path, num_best=0, alpha=0)
sent = '오늘 하루'
toked = tok(sent)

while 1:
  input_ids = torch.tensor([vocab[vocab.bos_token],]  + vocab[toked]).unsqueeze(0)
  pred = model(input_ids)[0][0]
  gen = vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())[-1]
  if gen in ['<unk>','</s>','▁.']:
      break
  sent += gen.replace('▁', ' ')
  toked = tok(sent)

print("Generated sentence: ", sent + ".")
```

### Output
```
> Generated sentence:  오늘 하루 수고가 많네요.
```

---

# Fine-tuning KoGPT2 for text classification: 'KoGPT2_finetuning_for_NSMC.py'
* 문장 분류 과제를 위해 KoGPT2를 NSMC 데이터셋으로 파인튜닝하는 코드입니다.
[SKT-AI의 KoGPT2](https://github.com/SKT-AI/KoGPT2) 및 [SKTBrain의 KoBERT 영화리뷰 분류 코드](https://github.com/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb)를 참고하고 수정하였습니다.

## How to use
* 기본적으로 SKT의 KoGPT2에 맞추어져 있습니다. 만약 트랜스포머 기반 pre-trained KoGPT2 모델(EX.[KoGPT2/Transformers](https://github.com/taeminlee/KoGPT2-Transformers))로 변경하고 싶으시면 '--model_name_or_path'를 바꿔주세요 (EX.'taeminlee/kogpt2').
마찬가지로 Pre-train을 진행한 후에 생성되는 output directory를 '--model_name_or_path'의 인자로 입력하면 직접 훈련한 모델을 파인튜닝할 수 있습니다.

## Example

> python3 KoGPT2_finetuning_for_NSMC.py





