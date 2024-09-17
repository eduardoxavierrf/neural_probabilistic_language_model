class CharTokenizer:
  def __init__(self):
    self.char_to_idx = {}
    self.idx_to_char = {}
    self.vocab_size = 0

  def fit(self, texts):
    unique_chars = set()
    for text in texts:
      unique_chars.update(text)
    
    for char in sorted(unique_chars):
      if char not in self.char_to_idx:
        self.char_to_idx[char] = self.vocab_size
        self.idx_to_char[self.vocab_size] = char
        self.vocab_size += 1

  def encode(self, text):
    return [self.char_to_idx.get(char, self.char_to_idx.get('<UNK>', -1)) for char in text]

  def decode(self, tokens):
    return ''.join([self.idx_to_char.get(idx, '<UNK>') for idx in tokens])

  def add_special_tokens(self, tokens):
    for token in tokens:
      if token not in self.char_to_idx:
        self.char_to_idx[token] = self.vocab_size
        self.idx_to_char[self.vocab_size] = token
        self.vocab_size += 1