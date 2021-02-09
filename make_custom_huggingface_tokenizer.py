from tokenizers import BertWordPieceTokenizer
from transformers import ElectraTokenizer
import re


def text_normalization(text):
    text = text.strip()
    text = re.sub(r'([?.,!+-])', r' \1 ', text)
    text = re.sub(r'[" "]+', " ", text)

    text = re.sub(r'[^0-9a-zA-Z가-힣/?.,~!@#$%&*()+_-]+', ' ', text)
    text = text.strip()
    return text



if __name__ == '__main__':
    sample = '확인해 드릴게요, MVP_Hugging (참치) sdf세금을 #을 눌러주세요 "라고" 하고 example@google.com 포함해서 102만 원이라고 나오네요.'
    # tokenizer = BertWordPieceTokenizer()
    # tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    tokenizer = ElectraTokenizer.from_pretrained('./hugging_korean')

    sample_encode = tokenizer.encode(sample)
    print(tokenizer.tokenize(sample))
    print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample)))
    print(tokenizer.encode(sample))
    print(tokenizer.convert_ids_to_tokens(tokenizer.encode(sample)))
    # tokenizer.save_pretrained('./hugging_korean')
