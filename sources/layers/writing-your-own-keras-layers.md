# 직접 케라스 레이어 구성하기

간단한 상태 비보존형 커스텀 작업을 하기 위해서는 'layers.core.Lambda' 레이어를 사용하는 것이 좋습니다. 하지만 학습 가능한 가중치를 사용해야 하는 커스텀 작업을 하기 위해서는 직접 레이어를 구성해야 합니다.

**Keras 2.0**을 기준으로 케라스 레이어의 골격은 세 가지 메서드를 필요로 합니다(기준 버전보다 낮은 버전을 사용하고 있다면, 업그레이드 해야합니다).

- 'build(input_shape)': 사용 할 가중치를 정의합니다. 이 메서드의 끝에서는 'self.built = True'로 지정하기 위해 반드시 'super([Layer], self).build()'를 호출해야만 합니다.
- 'call(x)': 레이어의 논리를 구성합니다. 마스킹을 지원하지 않는 레이어를 구성하기 원한다면, 'call'의 첫 번째 인자로 전달되는 인풋 텐서만 주의하면 됩니다.
- 'compute_output_shape(input_shape)': 레이어가 인풋의 형태를 수정하는 경우, 변형된 형태에 대한 논리를 지정해야 합니다. 이것은 케라스가 자동으로 형태를 추론하게 합니다.

```python
from keras import backend as K
from keras.layers import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 레이어에서 학습 가능한 가중치를 만듭니다.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # 마지막에 이것을 호출해야 합니다. 

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```

다중 인풋 텐서와 다중 아웃풋 텐서를 가진 케라스 레이어를 정의하는 것도 가능합니다. 이를 위해선 'build(input_shape)', 'call(x)' 그리고 'compute_output_shape(input_shape)'의 인풋과 아웃풋이 리스트라고 가정해야 합니다. 다음은 위와 비슷한 예시입니다.

```python
from keras import backend as K
from keras.layers import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 레이어에서 학습 가능한 가중치를 만듭니다.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # 마지막에 이것을 호출해야 합니다.

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]
```

케라스에서 사용할 수 있는 레이어는 어떻게 구현해야 되는지에 대한 모든 것을 예로서 제공합니다. 망설이지 말고 소스 코드를 읽어보세요!
