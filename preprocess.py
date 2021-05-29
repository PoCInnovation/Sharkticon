#######################
#    Tokenization     #
#######################


from tokenizers import BertWordPieceTokenizer
#etape 1 : lire la dataset

tokenizer = BertWordPieceTokenizer(
  clean_text=False,
  handle_chinese_chars=False,
  strip_accents=False,
  lowercase=True,
)

files = ['light_dataset.csv']

tokenizer.train(
  files,
  vocab_size=100,
  min_frequency=2,
  show_progress=True,
  special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
  limit_alphabet=1000,
  wordpieces_prefix="##"
)

tokenizer.save('./my-tokenizer.json')

# with open('light_dataset.csv') as csvfile:
#     reader = csv.reader(csvfile, delimiter=';')
#     for line in reader:
#         for field in line:
#             tokens = word_tokenize(field)

# print(tokens)
# exit(0)

# for i in range(10):
#     print(srcip[i])
#     #print(srcport[i])
#     #print(destip[i])
#     #print(destport[i])

#print(df)

#etape 2 : Vectoriser la donnée
