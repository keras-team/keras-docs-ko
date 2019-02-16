# Keras 모델에 관하여

Keras가 제공하는 모델에는 [Sequential 모델](./models/sequential.md)과 [함수형 API와 함께 사용되는 Model 클래스](/models/model) 두 가지 종류가 있습니다.

이 모델들은 아래의 메소드들과 속성들을 공통적으로 가지고 있습니다.

- `model.layers`: 모델을 구성하는 층들이 저장된 1차원 리스트 입니다.
- `model.inputs`: 모델의 입력 텐서들이 저장된 1차원 리스트 입니다.
- `model.outputs`: 모델의 출력 텐서들이 저장된 1차원 리스트 입니다.
- `model.summary()`: 모델의 구조를 요약해 출력해 줍니다. [`utils.print_summary(model)`](/utils/#print_summary)로도 동일한 출력을 얻을 수 있습니다.
- `model.get_config()`: 모델의 설정이 저장된 딕셔너리를 반환합니다. 모든 모델은 다음과 같이 설정 내용으로부터 다시 인스턴스화 될 수 있습니다.

```python
config = model.get_config()
model = Model.from_config(config)
# 또는, Sequential 모델의 경우:
model = Sequential.from_config(config)
```

- `model.get_weights()`: 모델의 가중치 텐서들이 NumPy 배열로 저장된 1차원 리스트 입니다.
- `model.set_weights(weights)`: 모델의 가중치 값을 NumPy 배열의 리스트로부터 설정합니다. 리스트에 있는 배열들의 크기는 `get_wieghts()`로부터 반환된 것과 동일해야 합니다.
- `model.to_json()`: 모델의 구조를 JSON 문자열로 반환합니다. 이때, 모델의 가중치는 제외되고 오로지 구조만이 포함됩니다. 이 JSON 문자열로부터 다음과 같이 동일한 모델을 (다시 초기화된 가중치와 함께)다시 인스턴스화 할 수 있습니다.

```python
from keras.models import model_from_json

json_string = model.to_json()
model = model_from_json(json_string)
```

- `model.to_yaml()`: 모델의 구조를 YAML 문자열로 반환합니다. 이때, 모델의 가중치는 제외되고 오로지 구조만이 포함됩니다. 이 YAML 문자열로부터 다음과 같이 동일한 모델을 (다시 초기화된 가중치와 함께)다시 인스턴스화 할 수 있습니다.

```python
from keras.models import model_from_yaml

yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)
```

- `model.save_weights(filepath)`: 모델의 가중치를 HDF5 파일로 저장 합니다.
- `model.load_weights(filepath, by_name=False)`: 모델의 가중치를 (`save_weights`에 의해 생성된)HDF5 파일로부터 불러옵니다. 기본 설정인 `by_name=False`는 모델과 가중치 파일의 네트워크 구조가 동일하다 가정합니다. 만약 구조가 다르다면, `by_name=True`를 사용해 동일한 이름을 가진 층들에 대해서만 가중치를 불러올 수도 있습니다.

**Note**: `h5py`의 설치는 FAQ의 [Keras 모델을 저장하기 위한 HDF5 또는 h5py는 어떻게 설치할 수 있나요?](/getting-started/faq/#how-can-i-install-HDF5-or-h5py-to-save-my-models-in-Keras)를 참조하시길 바랍니다.

## Model 하위 클래스 만들기

위에 언급된 두 가지 방법 외에도, Keras 2.2.0 버전부터 `Model` 클래스의 하위 클래스를 만들어 모델을 구현할 수 있는 API가 추가되었습니다. 해당 API를 사용해 `Model`의 하위 클래스를 만들고, `call` 메소드에 사용자의 요구에 맞는 포워드 패스 과정을 구현하여 모델을 만들 수도 있습니다.

다음은 `Model`의 하위 클래스로서 만들어진 간단한 다층 퍼셉트론의 예제입니다.

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

모델에 사용되는 각 층은 `__init__(self, ...)`에 정의되어 있고, 포워드 패스는 `call(self, inputs)`에 구현되어 있습니다. 임의의 층을 지정해 주는 것과 같이 사용자는 `call` 내부에서 `self.add_loss(loss_tensor)`를 호출함으로써 임의의 손실 함수를 지정해줄 수도 있습니다.

하지만, 위와 같이 모델을 구현할 때 모델의 구조는 정적 그래프가 아닌 Python 코드로서 정의되기 때문에, 모델의 구조를 확인하거나 저장할 수 없습니다. 결과적으로 해당 API를 사용하여 모델을 구현한 경우 다음의 메소드들과 속성들을 **사용할 수 없습니다**. 

- `model.inputs` 그리고 `model.outputs`.
- `model.to_yaml()` 그리고 `model.to_json()`
- `model.get_config()` 그리고 `model.save()`.

**중요한 점:** 작업에 적합한 API를 사용해야 합니다. `Model` 클래스의 하위 클래스를 만드는 API는 복잡한 모델을 구현하는데 있어서 높은 유연성을 제공해 줄 수는 있지만, 코드가 좀 더 길고 복잡해지며, 사용자 오류의 소지가 높아진다는 단점이 있습니다. 가능하면 사용자 친화적인 함수형 API를 사용하는 것이 좋습니다.
