# 직접 케라스 층<sub>writing-your-own-keras-layers</sub> 만들기

가중치<sub>weight</sub>를 학습하지 않는 간단한 사용자 정의 연산이 필요한 경우에는  `layers.core.Lambda` 층을 사용하는 것이 좋습니다. 하지만 가중치 학습이 필요한 경우라면 직접 층을 만들어야 합니다.

**케라스 2.0** 이후의 케라스 층의 골격은 밑의 세 가지 메서드를 필요로 합니다(만약 이전 버전을 사용하고 계시다면 업그레이드가 필요합니다).

- 'build(input_shape)': 사용 할 가중치를 정의합니다. 이 메소드의 끝에서는 반드시 `super([Layer], self).build()`를 호출해 `self.built = True` 로 지정해야 합니다.
- 'call(x)': 해당 층이 수행할 연산을 정의하는 메소드입니다. 마스킹을 지원하는 층을 만드는 경우가 아니라면, 입력 텐서를 받는 첫 번째 인자 외에 다른 인자는 필요하지 않습니다. 
- 'compute_output_shape(input_shape)': 층이 수행하는 연산에서 입력의 형태를 수정하는 경우, 변형된 형태를 명시해주어야 합니다. 이는 다음 층이 자동으로 입력값의 형태를 추론할 수 있도록 합니다. 

```python
from keras import backend as K
from keras.layers import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 층에서 사용될 학습 가능한 가중치를 만듭니다.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # 끝에서 꼭 이 함수를 호출해야 합니다!

    def call(self, x):
        return K.dot(x, self.kernel) # 이 층이 실제로 수행할 연산입니다.

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```

여러 텐서를 입력받아 다시 여러 텐서를 출력하는 층을 정의할 수도 있습니다. 이를 위해선 `build(input_shape)`, `call(x)`,`compute_output_shape(input_shape)`의 입력과 출력이 리스트라고 가정해야 합니다. 다음은 위와 동일한 연산에 리스트 입출력을 적용한 예시입니다:

```python
from keras import backend as K
from keras.layers import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list) # input_shape가 리스트 형식이 아닐 경우 오류를 출력합니다.
        # 층에서 사용될 학습 가능한 가중치를 만듭니다.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim), # shape에 입력 리스트의 0번째 텐서가 활용되었습니다.
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # 끝에서 꼭 이 함수를 호출해야 합니다!

    def call(self, x):
        assert isinstance(x, list) # 입력값이 리스트가 아닐 경우 오류를 출력합니다.
        a, b = x # 두 개의 텐서를 가진 리스트 입력을 가정하고 각각을 a, b에 할당합니다.
        return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)] # 두 텐서를 이용하여 연산을 수행하고 리스트로 반환합니다.

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list) # input_shape가 리스트 형식이 아닐 경우 오류를 출력합니다.
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]] # 출력 형태를 리스트 형식으로 반환합니다.
```
케라스에서는 사용할 수 있는 모든 층에 대한 구현 방법이 예시로 준비되어 있습니다.
해당 층에 대한 소스 코드를 확인하세요!
