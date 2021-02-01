import sentencepiece as spm
import pandas as pd
import urllib.request as request
import csv

SRC_DATA_PATH = '../integrated_data/korean_for_nlp.txt'


if __name__ == '__main__':
    input = SRC_DATA_PATH
    vocab_size = '1500'
    model_type = 'unigram'
    model_prefix = 'spm_%s_%s' % (model_type, vocab_size)
    max_sentence_length = '9999'


    spm.SentencePieceTrainer.Train('--input=%s --model_prefix=%s --vocab_size=%s'
                                   ' --model_type=%s --max_sentence_length=%s'
                                   ' --pad_id=0 --pad_piece=[PAD]'
                                   ' --unk_id=1 --unk_piece=[UNK]'
                                   ' --bos_id=2 --bos_piece=[BOS]'
                                   ' --eos_id=3 --eos_piece=[EOS]' 
                                   ' --user_defined_symbols=[CLS]' % (
                                    input, model_prefix, vocab_size, model_type, max_sentence_length))

    vocab_list = pd.read_csv('%s.vocab' % model_prefix, sep='\t', header=None, quoting=csv.QUOTE_NONE)
    print(vocab_list[:10])

    sp = spm.SentencePieceProcessor()
    vocab_file = "%s.model" % model_prefix
    sp.load(vocab_file)

    lines = [
      '나는 안녕하세요 1+1 이벤트 진행 중이다, 가격 1300원이야.',
      "t 값이 15,021원입니다."
    ]
    for line in lines:
        line = sp.IdToPiece(2) + line + sp.IdToPiece(3)
        print(line)
        print(sp.encode_as_pieces(line))
        print(sp.encode_as_ids(line))
        print()
    print(sp.IdToPiece(5))
    print(sp.piece_to_id('[BOS]'))
