from base_autoencoder import AutoEncoder
import torch
import torch.nn as nn


class Encoder(AutoEncoder):
  def __init__(self,
               input_shape: int,
               hidden_units: int,
               *args, **kwargs) -> None:
    super().__init__(*args, **kwargs, )
    self.encoder_layer = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=input_shape, out_features=512),
        nn.ReLU(),
        nn.Tanh(),
        nn.Linear(in_features=512, out_features=128),
        nn.ReLU(),
        nn.Tanh(),
        nn.Linear(in_features=128, out_features=32),
        nn.Tanh(),
        nn.Linear(in_features=32, out_features=hidden_units),
    )

  def forward(self, x: torch.Tensor):
      return self.encoder_layer(x)


class Decoder(AutoEncoder):
  def __init__(self,
                input_shape: int,
                hidden_units: int,
                *args, **kwargs) -> None:
      super().__init__(*args, **kwargs)
      self.decoder_layer = nn.Sequential(
          nn.Linear(in_features=hidden_units, out_features=32),
          nn.Tanh(),
          nn.Linear(in_features=32, out_features=128),
          nn.Tanh(),
          nn.Linear(in_features=128, out_features=512),
          nn.Tanh(),
          nn.Linear(in_features=512, out_features=input_shape),
      )
      self.activation = nn.Sigmoid()

  def forward(self, x: torch.Tensor):
    out = self.activation(self.decoder_layer(x))
    return out.view(out.size(0), 1, 28, 28)


class DAC_AE(AutoEncoder):
  def __init__(
          self,
          input_shape: int,
          hidden_units: int,
  ) -> None:
      super().__init__()
      self.encoder = Encoder(
          input_shape=input_shape,
          hidden_units=hidden_units,
      )
      self.decoder = Decoder(
          input_shape=input_shape,
          hidden_units=hidden_units,
      )

  def forward(self, x):
      emb = self.encoder(x)
      out = self.decoder(emb)
      return out