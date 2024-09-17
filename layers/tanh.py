import numpy as np

class Tanh:
  def __init__(self):
    self.output = None

  def forward(self, inputs: np.ndarray) -> np.ndarray:
    self.output = np.tanh(inputs)
    return self.output
  
  def backward(self, dout) -> np.ndarray:
    return dout * (1 - np.square(self.output))
  
  def update_params(self, learning_rate):
    pass