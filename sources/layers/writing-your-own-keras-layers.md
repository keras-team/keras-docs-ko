# 직접 케라스 레이어 만들기

간단한 상태 비보존형 커스텀 운용을 하려면 아마 `layers.core.Lambda` 레이어를 사용하는 편이 나을 것입니다. 하지만 학습가능한 가중치를 수반하는 커스텀 운용을 하려면 자신만의 레이어를 만들어야 합니다.

**Keras 2.0 현재** 케라스 레이어의 골조는 다음과 같습니다(만약 이전 버전을 사용하고 계시다면 업그레이드 해주십시오). 세 가지 메서드만 구현하시면 됩니다:

- `build(input_shape)`: 이 메서드에서 가중치를 정의합니다. 끝에서는 반드시 `super([Layer], self).build()`를 호출해 `self.built = True`로 지정해야 합니다.
- `call(x)`: 레이어의 논리의 핵심이 되는 메서드입니다. 마스킹을 지원하는 레이어를 만들 것이 아니라면 `call`의 첫 번째 인수에 전달되는 인풋 텐서만 주의하면 됩니다.
- `compute_output_shape(input_shape)`: 레이어가 인풋의 형태를 수정하는 경우, 형태 변형 논리를 지정해야 합니다. 이는 케라스가 자동 형태 추론을 할 수 있도록 합니다.

```python
from keras import backend as K
from keras.layers import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 이 레이어에 대해 학습가능한 가중치 변수를 만듭니다.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # 끝에서 꼭 이 함수를 호출하십시오

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```

다중 인풋 텐서와 다중 아웃풋 텐서를 가진 케라스 레이어를 정의하는 것도 가능합니다. 이를 해선 `build(input_shape)`, `call(x)` 그리고 `compute_output_shape(input_shape)`의 잇풋과 아웃풋이 리스트라고 가정해야 합니다. 다음은 위와 비슷한 예시입니다:

```python
from keras import backend as K
from keras.layers import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 이 레이어에 대해 학습가능한 가중치 변수를 만듭니다.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # 끝에서 꼭 이 함수를 호출하십시오

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]
```

기존의 케라스 레이어를 보시면 대부분의 기능을 어떻게 구현해야 하는지 참고할 수 있습니다. 망설이지 말고 소스코드를 읽어보세요!
