# How to Use

## Multi-Layer-Perceptron (MLP)

```yaml
# configs/model/mnist.yaml
net:
  _target_: src.models.components.multi_layer_perceptron.MultiLayerPerceptron
  input_size: 784
  hidden_size: [256, 256, 256]
  output_size: 10
```
