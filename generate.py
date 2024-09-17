from model import Model
from utils.tokenizer import CharTokenizer
import pickle
import numpy as np

text = open("data.txt", 'rb').read().decode(encoding='utf-8')

tokenizer = CharTokenizer()

tokenizer.fit([text])
tokenizer.add_special_tokens(['<UNK>'])

file = open("saved_model/model.pkl",'rb')
model = pickle.load(file)

generated = model.generate(10000, np.array([tokenizer.encode("José Dias amava ")]))

f = open("output.txt", "w")
f.write(tokenizer.decode(generated))
f.close()