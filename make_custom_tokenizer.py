import sentencepiece as spm
import pandas as pd
import urllib.request as request
import csv

SRC_DATA_PATH = '../integrated_data/korean_for_nlp.txt'

# request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv",
#                     filename="IMDb_Reviews.csv")

# train_df = pd.read_csv('IMDb_Reviews.csv')
# print(len(train_df))
# with open('imdb_review.txt', 'w', encoding='utf8') as f:
#     f.write('\n'.join(train_df['review']))


if __name__ == '__main__':
    spm.SentencePieceTrainer.Train('--input=%s --model_prefix=spm_unigram --vocab_size=32000'
                                   '--model_type=unigram --max_sentence_length=9999' % SRC_DATA_PATH)

    vocab_list = pd.read_csv('spm_unigram.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)
    print(vocab_list[:10])

    sp = spm.SentencePieceProcessor()
    vocab_file = "imdb.model"
    sp.load(vocab_file)

    lines = [
      "I didn't at all think of it this way.",
      "t t 쀏"
    ]
    for line in lines:
        line = sp.IdToPiece(1) + line + sp.IdToPiece(2)
        print(line)
        print(sp.encode_as_pieces(line))
        print(sp.encode_as_ids(line))
        print()
    print(sp.IdToPiece(1))
