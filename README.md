# train_KoGPT2
Pre-train and fine-tune transformer models including KoGPT2

# Requirements
pip install -r requirements.txt

## Pre-train transformer: 'run_language_modeling.py'
Modified [Huggingface-language_modeling.py](https://github.com/huggingface/transformers/tree/master/examples/language-modeling) and [SKT-KoGPT2](https://github.com/SKT-AI/KoGPT2).
The modified code is compatible for both transformer tokenizer(AutoTokenizer) and gluonnlp tokenizer(BERTSPTokenizer).
It is also compatible for both transformer model(AutoModelWithLMHead) and pre-trained KoGPT2 model(GP2LMHeadModel)

SKT-AI에서 공개한 KoGPT2를 포함한 허깅페이스 기반 트랜스포머 모델들(BERT,GPT2 등)을 훈련할 수 있는 코드입니다.
트랜스포머 기반 토크나이저 뿐만 아니라 gluonnlp의 BERTSPTokenizer도 사용할 수 있도록 하였습니다.

# How to use
Current arguments are set up for pre-training KoGPT2 with gluonnlp-BERTSPTokenizer. Change arguments such as 'model_name_or_path', 'config_name' and 'use_gluonnlp_tokenizer' if you want to use other tokenizer and model for training.

코드는 기본적으로 gluonnlp-BERTSPTokenizer로 KoGPT2를 추가 훈련하도록 맞춰져있습니다. 트랜스포머의 모델이나 토크나이저를 사용하고 싶은 경우 'model_name_or_path', 'config_name' and 'use_gluonnlp_tokenizer' 등의 인자를 변경해주세요. 

# Example 
Additional pre-train KoGPT2 for [NSMC dataset](https://github.com/e9t/nsmc)

NSMC 데이터셋(라벨 제외)으로 KoGPT2를 추가 학습한 경우의 예시입니다. 커맨드는 아래와 같습니다.

> python3 run_language_modeling.py --do_train --output_dir=train_output --train_data_file=dataset/ratings_train.txt --do_eval --eval_data_file=dataset/ratings_test.txt

## Fine-tuning KoGPT2 for text classification: 'KoGPT2_finetuning_for_NSMC.py'
문장 분류 과제를 위해 KoGPT2를 NSMC 데이터셋으로 파인튜닝하는 코드입니다.
[SKT-AI의 KoGPT2](https://github.com/SKT-AI/KoGPT2) 및 [SKTBrain의 KoBERT 영화리뷰 분류 코드](https://github.com/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb)를 참고하고 수정하였습니다.

# How to use
기본적으로 SKT의 KoGPT2에 맞추어져 있습니다. 만약 트랜스포머 기반 pre-trained KoGPT2 모델(EX.[KoGPT2/Transformers](https://github.com/taeminlee/KoGPT2-Transformers))로 변경하고 싶으시면 '--model_name_or_path'를 바꿔주세요 (EX.'taeminlee/kogpt2').
마찬가지로 Pre-train을 진행한 후에 생성되는 output directory를 '--model_name_or_path'의 인자로 입력하면 직접 훈련한 모델을 파인튜닝할 수 있습니다.

# Example

> python3 KoGPT2_finetuning_for_NSMC.py



