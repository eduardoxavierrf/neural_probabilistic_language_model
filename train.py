from utils.tokenizer import CharTokenizer
from model import Model
import numpy as np

context_len = 16

text = open("data.txt", 'rb').read().decode(encoding='utf-8')

tokenizer = CharTokenizer()

tokenizer.fit([text])
tokenizer.add_special_tokens(['<UNK>'])

tokens = tokenizer.encode(text)

# Prepare data
x_data = []
y_data = []
for i in range(len(tokens) - context_len):
  context = tokens[i:i + context_len]

  target = tokens[i + context_len]

  x_data.append(context)
  y_data.append(target)

model = Model(context_len, 32, tokenizer.vocab_size)

# Shuffle data
x_data = np.array(x_data)
y_data = np.array(y_data)
y_data = np.array([model.one_hot(y, tokenizer.vocab_size) for y in y_data])

indices = np.random.permutation(len(x_data))

x_data = x_data[indices]
y_data = y_data[indices]

# Train model
model.train(x_data, y_data, 50, 32, 0.02)
model.save("saved_model/model.pkl")
