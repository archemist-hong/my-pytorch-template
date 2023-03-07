from typing import List

from torch import nn


class MultiLayerPerceptron(nn.Module):
    """Multilayer Perceptron (MLP) Class."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: List[int] = [256, 256, 256],
        output_size: int = 10,
    ):
        super().__init__()
        dims = [input_size] + hidden_size
        self.layers: List[nn.Module] = []
        for idx in range(len(dims) - 1):
            self.layers += [*self.make_layers(dims[idx], dims[idx + 1])]
        self.layers += [nn.Linear(hidden_size[-1], output_size)]
        self.layers = nn.Sequential(*self.layers)
        self.init_param()

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.layers(x)

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, input_size: int, output_size: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        layers += [nn.Linear(input_size, output_size), nn.BatchNorm1d(output_size), nn.ReLU()]
        return nn.Sequential(*layers)


if __name__ == "__main__":
    _ = MultiLayerPerceptron()
