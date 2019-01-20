
## Usage of activations

Activations can either be used through an `Activation` layer, or through the `activation` argument supported by all forward layers:

```python
from keras.layers import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
```

This is equivalent to:

```python
model.add(Dense(64, activation='tanh'))
```

You can also pass an element-wise TensorFlow/Theano/CNTK function as an activation:

```python
from keras import backend as K

model.add(Dense(64, activation=K.tanh))
```

## Available activations

### softmax


```python
keras.activations.softmax(x, axis=-1)
```


Softmax activation function.

__Arguments__

- __x__: Input tensor.
- __axis__: Integer, axis along which the softmax normalization is applied.

__Returns__

Tensor, output of softmax transformation.

__Raises__

- __ValueError__: In case `dim(x) == 1`.
    
----

### elu


```python
keras.activations.elu(x, alpha=1.0)
```


Exponential linear unit.

__Arguments__

- __x__: Input tensor.
- __alpha__: A scalar, slope of negative section.

__Returns__

The exponential linear activation: `x` if `x > 0` and
`alpha * (exp(x)-1)` if `x < 0`.

__References__

- [Fast and Accurate Deep Network Learning by Exponential
   Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
    
----

### selu


```python
keras.activations.selu(x)
```


Scaled Exponential Linear Unit (SELU).

SELU is equal to: `scale * elu(x, alpha)`, where alpha and scale
are pre-defined constants. The values of `alpha` and `scale` are
chosen so that the mean and variance of the inputs are preserved
between two consecutive layers as long as the weights are initialized
correctly (see `lecun_normal` initialization) and the number of inputs
is "large enough" (see references for more information).

__Arguments__

- __x__: A tensor or variable to compute the activation function for.

__Returns__

   The scaled exponential unit activation: `scale * elu(x, alpha)`.

__Note__

- To be used together with the initialization "lecun_normal".
- To be used together with the dropout variant "AlphaDropout".

__References__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    
----

### softplus


```python
keras.activations.softplus(x)
```


Softplus activation function.

__Arguments__

- __x__: Input tensor.

__Returns__

The softplus activation: `log(exp(x) + 1)`.
    
----

### softsign


```python
keras.activations.softsign(x)
```


Softsign activation function.

__Arguments__

- __x__: Input tensor.

__Returns__

The softsign activation: `x / (abs(x) + 1)`.
    
----

### relu


```python
keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0.0)
```


Rectified Linear Unit.

With default values, it returns element-wise `max(x, 0)`.

Otherwise, it follows:
`f(x) = max_value` for `x >= max_value`,
`f(x) = x` for `threshold <= x < max_value`,
`f(x) = alpha * (x - threshold)` otherwise.

__Arguments__

- __x__: Input tensor.
- __alpha__: float. Slope of the negative part. Defaults to zero.
- __max_value__: float. Saturation threshold.
- __threshold__: float. Threshold value for thresholded activation.

__Returns__

A tensor.
    
----

### tanh


```python
keras.activations.tanh(x)
```


Hyperbolic tangent activation function.

----

### sigmoid


```python
keras.activations.sigmoid(x)
```


Sigmoid activation function.

----

### hard_sigmoid


```python
keras.activations.hard_sigmoid(x)
```


Hard sigmoid activation function.

Faster to compute than sigmoid activation.

__Arguments__

- __x__: Input tensor.

__Returns__

Hard sigmoid activation:

- `0` if `x < -2.5`
- `1` if `x > 2.5`
- `0.2 * x + 0.5` if `-2.5 <= x <= 2.5`.

----

### exponential


```python
keras.activations.exponential(x)
```


Exponential (base e) activation function.

----

### linear


```python
keras.activations.linear(x)
```


Linear (i.e. identity) activation function.


## On "Advanced Activations"

Activations that are more complex than a simple TensorFlow/Theano/CNTK function (eg. learnable activations, which maintain a state) are available as [Advanced Activation layers](layers/advanced-activations.md), and can be found in the module `keras.layers.advanced_activations`. These include `PReLU` and `LeakyReLU`.
