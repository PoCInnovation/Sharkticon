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
output = tokenizer.encode("192.168.17.1")
print(output.tokens)
inputt = tokenizer.decode(output.ids)
print(inputt)
print(tokenizer.get_vocab_size())
print(Tokenizer.input_ids)