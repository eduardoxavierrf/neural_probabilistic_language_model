import numpy as np

class Flatten:
  def forward(self, inputs: np.ndarray) -> np.ndarray:
    self.inputs_shape = inputs.shape
    return inputs.reshape(inputs.shape[0], -1)
  
  def backward(self, dout: np.ndarray) -> np.ndarray:
    return dout.reshape(self.inputs_shape)
  
  def update_params(self, learning_rate):
    pass