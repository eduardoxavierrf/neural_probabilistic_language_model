import layers
import numpy as np
import pickle

class Model:
  def __init__(self, context_len: int, embedding_dim: int, vocab_size: int) -> None:
    self.context_len = context_len
    self.layers = [
      layers.Embedding(vocab_size, embedding_dim),
      layers.Flatten(),
      layers.Linear(context_len * embedding_dim, 512),
      layers.Tanh(),
      layers.Linear(512, vocab_size),
      layers.Softmax()
    ]

  def forward(self, inputs):
    for layer in self.layers:
      inputs = layer.forward(inputs)

    return inputs
  
  def backward(self, dout):
    for layer in reversed(self.layers):
      dout = layer.backward(dout)

    return dout
  
  def update_params(self, learning_rate):
    for layer in self.layers:
      layer.update_params(learning_rate)

  def cross_entropy_loss(self, y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)
    loss = -np.sum(y_true * np.log(y_pred), axis=1)
    
    return np.mean(loss)
  
  def avarage_loss(self, x_data, y_data):
    output = self.forward(x_data)

    return self.cross_entropy_loss(y_data, output)
  
  def predict(self, inputs):
    output = self.forward(np.array([inputs]))
    return np.argmax(output)
  
  def accuracy(self, x_data, y_data):
    results = [(self.predict(x),np.argmax(y)) for x, y in zip(x_data, y_data)]

    return (sum(int(x == y) for (x, y) in results)/len(y_data)) * 100
  
  def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int, learning_rate:float = 0.01):
    num_batches = x_train.shape[0] // batch_size
    for epoch in range(epochs):
      total_loss = 0
      for i in range(num_batches):
        start_idx = batch_size * i
        end_idx = start_idx + 32
        output = self.forward(x_train[start_idx:end_idx])

        delta = output - y_train[start_idx:end_idx]
        self.backward(delta)
        self.update_params(learning_rate)
        total_loss += self.cross_entropy_loss(y_train[start_idx:end_idx], output)

      print(f"Epoch {epoch}, Loss {total_loss/num_batches}, Accuracy {self.accuracy(x_train, y_train)}")

  def one_hot(self, y, num_classes):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1

    return one_hot.flatten()
  
  def save(self, filename):
    with open(filename, 'wb') as file:
      pickle.dump(self, file)

  def generate(self, max_new_tokens: int, context: np.ndarray):
    generated_text = context.copy()

    for _ in range(max_new_tokens):

      output = self.forward([generated_text[-self.context_len:]])
      output = output.reshape(-1)
      predicted = np.random.choice(np.arange(len(output)), size=1, p=output)

      generated_text = np.append(generated_text, predicted)

    return generated_text