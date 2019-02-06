# Keras 모델에 관하여

Keras에는 두 개의 주된 모델 종류가 있습니다: [순차적 모델](/models/sequential), 그리고 [함수형 API와 함께 사용되는 Model 클래스](/models/model).

이 모델들은 많은 메소드와 속성들을 공통적으로 가지고 있습니다:

- `model.layers`는 모델을 구성하는 계층들의 flat화된 리스트 입니다.
- `model.inputs`는 모델의 입력 tensor의 리스트 입니다.
- `model.outputs`는 메돌의 출력 tensor의 리스트 입니다.
- `model.summary()`는 모델의 모양의 요약을 출력합니다. 간단하게는 [utils.print_summary](/utils/#print_summary)을 사용할 수 있습니다.
- `model.get_config()`는 모델의 설정이 저장된 딕셔너리를 반환 합니다. 모델은 다음과 같이 설정으로 부터 다시 인스턴스화 될 수 있습니다:

```python
config = model.get_config()
model = Model.from_config(config)
# 또는, 순차적 모델의 경우:
model = Sequential.from_config(config)
```

- `model.get_weights()`는 모델의 모든 가중치 tensor의 리스트를 Numpy 배열로서 반환 합니다.
- `model.set_weights(weights)`는 모델의 가중치 값을 Numpy 배열의 리스트로 부터 설정합니다. 리스트에 있는 배열들은 `get_wieghts()`로부터 반환된 것과 동일한 shape이어야만 합니다.
- `model.to_json()`는 모델의 모양을 JSON 문자열로서 반환합니다. 이 때, 모양에는 가중치가 포함되지 않고, 오로지 아키텍쳐만이 포함됩니다. 이 JSON 문자열로 부터 다음과 같이 동일한 모델을 (다시 초기화된 가중치와 함께) 다시 인스턴스화 할 수 있습니다:

```python
from keras.models import model_from_json

json_string = model.to_json()
model = model_from_json(json_string)
```
- `model.to_yaml()`는 모델의 모양을 YAML 문자열로서 반환합니다. 이 때, 모양에는 가중치가 포함되지 않고, 오로지 아키텍쳐만이 포함됩니다. 이 YAML 문자열로 부터 다음과 같이 동일한 모델을 (다시 초기화된 가중치와 함께) 다시 인스턴스화 할 수 있습니다:

```python
from keras.models import model_from_yaml

yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)
```

- `model.save_weights(filepath)`는 모델의 가중치를 HDF5 파일로서 저장 합니다.
- `model.load_weights(filepath, by_name=False)`는 모델의 가중치를 (`save_weights`에 의해 생성된)HDF5 파일로 부터 불러옵니다. 디폴트로는 아키텍쳐가 변하지 않았다는 것이 가정 됩니다. 가중치를 (몇 계층들을 공통적으로 포함한)다른 아키텍쳐에 불러오고 싶다면, `by_name=True`를 사용해서 동일한 이름을 가진 계층들에 대해서만 불러 오십시오.

메모: FAQ에 있는 `h5py`를 설치하는 지침인 [Keras에서 모델을 저장하기 위한 HDF5 또는 h5py를 어떻게 설치할 수 있는가?](/getting-started/faq/#how-can-i-install-HDF5-or-h5py-to-save-my-models-in-Keras)도 확인해 주시기 바랍니다.


## Model 하위 클래스 만들기

위의 언급된 두 종류의 모델외에 추가적으로, `Model`클래스의 하위클래스를 만들고 `call`메소드에 여러분 만의 forward pass를 구현하여 여러분 만의 완전히 커스텀화된 모델을 만들 수도 있습니다. (`Moddel`의 하위 클래스를 만드는 API는 Keras 2.2.0에서 소개되었습니다).

`Model`의 하위 클래스로서 만들어진 간단한 다중-계층 퍼셉트론의 예제입니다:

```python
import keras

class SimpleMLP(keras.Model):

    def __init__(self, use_bn=False, use_dp=False, num_classes=10):
        super(SimpleMLP, self).__init__(name='mlp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        self.num_classes = num_classes

        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(num_classes, activation='softmax')
        if self.use_dp:
            self.dp = keras.layers.Dropout(0.5)
        if self.use_bn:
            self.bn = keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs):
        x = self.dense1(inputs)
        if self.use_dp:
            x = self.dp(x)
        if self.use_bn:
            x = self.bn(x)
        return self.dense2(x)

model = SimpleMLP()
model.compile(...)
model.fit(...)
```

계층들은 `__init__(self, ...)`에 정의되어 있고, forward pass는 `call(self, inputs)`에 구체화 되어 있습니다. `call`내부에서 `self.add_loss(loss_tensor)`를 호출하면 (사용자 정의 계층에서 하는 것과 비슷하게)사용자 정의 손실을 명시해 줄 수도 있습니다. 

하위 클래스화된 모델에서, 모델의 토폴로지는 (계층들의 정적인 그래프 대신에) Python 코드에 의해 정의 됩니다.
모델의 토폴로지가 직렬화 또는 검사될 수 없다는 의미 입니다. 결과적으로, 다음의 메소드들과 속성들은 **하위 클래스화된 모델에서는 사용할 수 없습니다**:

- `model.inputs` 그리고 `model.outputs`.
- `model.to_yaml()` 그리고 `model.to_json()`
- `model.get_config()` 그리고 `model.save()`.

**중요한 점:** 작업에 적합한 API를 사용하십시오. `Model`의 하위 클래스를 만드는 API는 복잡한 모델을 구현하는데 있어서 높은 유연성을 제공해 줄 수 있지만, 거기에는 (위의 사라진 특징들에 더하여) 대가가 따릅니다:
좀 더 장황하고, 복잡하고, 그리고 사용자의 에러가 발생할 소지가 있습니다. 가능한한 사용자에게 친숙한 함수형 API 사용을 선호 하십시오.
