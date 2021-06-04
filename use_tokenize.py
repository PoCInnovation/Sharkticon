##
## EPITECH PROJECT, 2021
## Sharkticon
## File description:
## use_tokenize
##

from tokenizers import BertWordPieceTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer.from_file("my-tokenizer.json")

print(output.tokens)