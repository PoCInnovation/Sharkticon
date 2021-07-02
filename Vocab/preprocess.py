#######################
#    Tokenization     #
#######################

from tokenizers.pre_tokenizers import Whitespace
from tokenizers import BertWordPieceTokenizer
from tokenizers.trainers import BpeTrainer

tokenizer = BertWordPieceTokenizer(
    clean_text=False,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=True,
)

files = ['Dataset_final.csv']

tokenizer.train(
    files,
    vocab_size=30000,
    min_frequency=2,
    show_progress=True,
    special_tokens=['[START]', '[END]', '[PAD]',
                    '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
    limit_alphabet=1000,
    wordpieces_prefix="##"
)

tokenizer.save('./my-tokenizer.json')