# 직접 케라스 층 만들기

가중치<sub>weight</sub>를 학습하지 않는 간단한 사용자 정의 연산이 필요한 경우에는 `layers.core.Lambda`층<sub>layer</sub>을 활용하는 편이 좋습니다. 하지만 가중치 학습이 필요한 경우라면 자신만의 층을 만들어야 합니다.

**케라스 2.0 이후의** 케라스 층의 골조는 다음과 같으며, 이하의 세 가지 메소드를 사용하여 사용자 정의 층을 만들 수 있습니다. (만약 이전 버전을 사용하고 계시다면 업그레이드가 필요합니다.)

- `build(input_shape)`: 이 메소드에서 가중치를 정의합니다. 메소드 코드의 끝에서는 반드시 `super([Layer], self).build()`를 호출해 `self.built = True`로 지정해야 합니다.
- `call(x)`: 해당 층이 수행할 연산을 정의하는 메소드입니다. 마스킹을 지원하는 층을 만들 것이 아니라면 입력 텐서를 받는 첫 번째 인자 외에 다른 인자는 필요하지 않습니다. 
- `compute_output_shape(input_shape)`: 해당 층의 연산을 거치면서 입력값의 형태<sub>shape</sub>가 변하는 경우 이 메소드를 통해 어떻게 변하는지 지정합니다. 이는 다음 층이 자동으로 입력값의 형태를 추론할 수 있도록 합니다.

```python
from keras import backend as K
from keras.layers import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 이 층에서 사용될 학습가능한 가중치 변수를 만듭니다.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # 끝에서 꼭 이 함수를 호출해야 합니다.

    def call(self, x):
        return K.dot(x, self.kernel) # 이 층이 실제로 수행할 연산입니다.

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```

여러 텐서를 입력받아 다시 여러 텐서를 출력하는 층을 정의할 수도 있습니다. `build(input_shape)`, `call(x)`, `compute_output_shape(input_shape)` 메소드의 입출력을 리스트로 설정하면 됩니다. 다음은 위와 동일한 연산에 리스트 입출력을 적용한 예시입니다:

```python
from keras import backend as K
from keras.layers import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list) # input_shape가 리스트 형식이 아닐 경우 오류를 출력합니다.
        # 이 층에서 사용될 학습가능한 가중치 변수를 만듭니다.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim), # shape에 입력 리스트의 0번째 텐서가 활용되었습니다.
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # 끝에서 꼭 이 함수를 호출해야 합니다.

    def call(self, x):
        assert isinstance(x, list) # 입력값이 리스트가 아닐 경우 오류를 출력합니다.
        a, b = x # 두 개의 텐서를 가진 리스트 입력을 가정하고 각각을 a, b에 할당합니다.
        return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)] # 두 텐서를 이용하여 연산을 수행하고 리스트로 반환합니다.

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list) # input_shape가 리스트 형식이 아닐 경우 오류를 출력합니다.
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]] # 출력 형태를 리스트 형식으로 반환합니다.
```

기존의 케라스 층을 열어보면 대부분의 기능을 어떻게 구현해야 하는지 참고할 수 있습니다. 망설이지 말고 소스코드를 읽어보세요!
