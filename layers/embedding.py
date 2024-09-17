import numpy as np

class Embedding:
  def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.weight = np.random.randn(num_embeddings, embedding_dim) * 0.01

  def forward(self, tokens: np.array) -> np.ndarray:
    self.inputs = tokens
    return self.weight[tokens]

  def backward(self, dout: np.ndarray) -> None:
    self.dW = np.zeros_like(self.weight)
    np.add.at(self.dW, self.inputs, dout)
    return None

  def update_params(self, learning_rate: float) -> None:
    self.weight -= learning_rate * self.dW