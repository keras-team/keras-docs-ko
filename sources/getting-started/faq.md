# 케라스 FAQ: 자주 묻는 케라스 질문

- [케라스는 어떻게 인용해야 합니까?](#케라스는-어떻게-인용해야-합니까)
- [케라스를 GPU에서 실행하려면 어떻게 해야 합니까?](#케라스를-gpu에서-실행하려면-어떻게-해야-합니까)
- [케라스를 여러 대의 GPU에서 실행하려면 어떻게 해야 합니까?](#케라스를-여러-대의-gpu에서-실행하려면-어떻게-해야-합니까)
- ["표본", "배치", "에폭"은 무슨 뜻입니까?](#표본-배치-에폭은-무슨-뜻입니까)
- [케라스 모델은 어떻게 저장합니까?](#케라스-모델은-어떻게-저장합니까)
- [왜 훈련 손실이 시험 손실보다 훨씬 높습니까?](#왜-훈련-손실이-시험-손실보다-훨씬-높습니까)
- [중간층의 출력은 어떻게 얻을 수 있습니까?](#중간층의-출력은-어떻게-얻을-수-있습니까)
- [메모리를 초과하는 데이터 세트에 케라스를 사용하려면 어떻게 해야 합니까?](#메모리를-초과하는-데이터-세트에-케라스를-사용하려면-어떻게-해야-합니까)
- [검증 손실이 더이상 감소하지 않는 경우 훈련을 어떻게 중단할 수 있습니까?](#검증-손실이-더이상-감소하지-않는-경우-훈련을-어떻게-중단할-수-있습니까)
- [검증 데이터 분리는 어떻게 계산됩니까?](#검증-데이터-분리는-어떻게-계산됩니까)
- [훈련 중 데이터의 순서가 섞입니까?](#훈련-중-데이터의-순서가-섞입니까)
- [각 에폭별 훈련 손실/검증 손실/정확도는 어떻게 기록할 수 있습니까?](#각-에폭별-훈련-손실--검증-손실--정확도는-어떻게-기록할-수-있습니까)
- [특정한 층의 가중치를 고정시키려면 어떻게 해야 합니까?](#특정한-층의-가중치를-고정시키려면-어떻게-해야-합니까)
- [상태 저장 순환 신경망을 사용하려면 어떻게 해야 합니까?](#상태-저장-순환-신경망을-사용하려면-어떻게-해야-합니까)
- [Sequential 모델에서 층을 없애려면 어떻게 해야 합니까?](#sequential-모델에서-층을-없애려면-어떻게-해야-합니까)
- [케라스에서 선행 학습된 모델을 사용하려면 어떻게 해야 합니까?](#케라스에서-선행-학습된-모델을-사용하려면-어떻게-해야-합니까)
- [케라스에서 HDF5 입력을 사용하려면 어떻게 해야 합니까?](#케라스에서-hdf5-입력을-사용하려면-어떻게-해야-합니까)
- [케라스 구성 파일은 어디에 저장됩니까?](#케라스-구성-파일은-어디에-저장됩니까)
- [개발 중 케라스를 사용해 재현 가능한 결과를 얻으려면 어떻게 해야 합니까?](#개발-중-케라스를-사용해-재현-가능한-결과를-얻으려면-어떻게-해야-합니까)
- [HDF5나 h5py를 설치해 케라스 모델을 저장하려면 어떻게 해야 합니까?](#hdf5나-h5py를-설치해-케라스-모델을-저장하려면-어떻게-해야-합니까)

---

### 케라스는 어떻게 인용해야 합니까?

연구 중 케라스가 도움이 되었다면 출판 시 케라스를 인용해 주십시오. 다음은 BibTeX 등재 예시입니다.

```
@misc{chollet2015keras,
  title={Keras},
  author={Chollet, Fran\c{c}ois and others},
  year={2015},
  howpublished={\url{https://keras.io}},
}
```

---

### 케라스를 GPU에서 실행하려면 어떻게 해야 합니까?

GPU를 지원하는 **TensorFlow** 혹은 **CNTK** 백엔드를 사용하는 경우, 사용 가능한 GPU가 감지되면 코드가 자동으로 GPU에서 작동됩니다.

**Theano** 백엔드를 사용하는 경우, 다음 방법 중 하나를 사용하시면 됩니다:

**방법 1**: `THEANO_FLAGS`를 사용합니다.
```bash
THEANO_FLAGS=device=gpu,floatX=float32 python my_keras_script.py
```

본인 장치의 식별자에 따라 'gpu' 표기를 바꾸어야 할 수도 있습니다(예. `gpu0`, `gpu1`, 등).

**방법 2**: `.theanorc`를 설정합니다: [설명](http://deeplearning.net/software/theano/library/config.html)

**방법 3**: 코드 첫 부분에 수동으로`theano.config.device`, `theano.config.floatX`를 설정합니다:
```python
import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'
```

---

### 케라스를 여러 대의 GPU에서 실행하려면 어떻게 해야 합니까?

**TensorFlow** 백엔드를 사용하실 것을 권장합니다. 단일 모델을 여러 대의 GPU에 실행하려면 다음의 두 가지 방법이 있습니다: **데이터 병렬처리**와 **장치 병렬처리**.

대부분의 경우에는 데이터 병렬처리를 사용하면 됩니다.

#### 데이터 병렬처리

데이터 병렬처리는 목표 모델을 장치마다 복제하고, 각각의 복제본으로 입력 데이터의 각기 다른 부분을 처리하는 방식으로 구성됩니다.
케라스의 내장 유틸리티 `keras.utils.multi_gpu_model`은 어떤 모델도 병렬 데이터 버전으로 만들 수 있으며, 8대의 GPU까지는 거의 비례하여 속도를 증가시킬 수 있습니다.

더 자세한 정보는 [multi_gpu_model](/utils/#multi_gpu_model)문서를 참고하십시오. 다음은 간단한 예시입니다.

```python
from keras.utils import multi_gpu_model

# `model`을 8대의 GPU에 복제합니다.
# 이 예시에서는 장치에 8대의 GPU가 있는 것으로 가정합니다.
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# `fit` 호출이 8대 GPU에 분산됩니다.
# 배치 크기가 256이므로 각 GPU는 32개 표본을 처리합니다.
parallel_model.fit(x, y, epochs=20, batch_size=256)
```

#### 장치 병렬처리

장치 병렬처리는 같은 모델의 다른 부분을 각기 다른 장치에서 실행하는 방식입니다. 두 브랜치로 구성된 모델과 같은 병렬 구조의 모델에 가장 적합합니다.

TensorFlow 디바이스 스코프를 사용해 이를 구현할 수 있습니다. 다음은 간단한 예시입니다.

```python
# 공유된 LSTM이 두 가지 다른 시퀀스를 병렬로 인코딩하는 모델
input_a = keras.Input(shape=(140, 256))
input_b = keras.Input(shape=(140, 256))

shared_lstm = keras.layers.LSTM(64)

# 첫 번째 시퀀스를 1대의 GPU에서 처리합니다
with tf.device_scope('/gpu:0'):
    encoded_a = shared_lstm(tweet_a)
# 다음 시퀀스를 다른 GPU에서 처리합니다
with tf.device_scope('/gpu:1'):
    encoded_b = shared_lstm(tweet_b)

# CPU에 결과들을 합칩니다
with tf.device_scope('/cpu:0'):
    merged_vector = keras.layers.concatenate([encoded_a, encoded_b],
                                             axis=-1)
```

---

### "표본", "배치", "에폭"은 무슨 뜻입니까?

다음은 케라스를 올바르게 이용하기 위해 알아두어야 할 몇 가지 일반적인 정의입니다.

- **표본<sub>sample</sub>**: 데이터 세트의 요소 하나
  - *예:* 합성곱 신경망의 **표본** 하나는 이미지 하나입니다
  - *예:* 오디오 파일 하나는 음성 인식 모델의 **표본** 하나입니다.
- **배치<sub>batch</sub>**: *N*개 표본들의 모음. 데이터 세트가 대량의 표본으로 이루어져 있을 경우 세트를 전체를 한 번에 처리하기보다 작은 배치들 여러 개로 나누어 처리하게 됩니다. 하나의 **배치**에서의 표본들은 독립적으로 동시에 처리됩니다. 훈련 과정에서는 하나의 배치마다 한 번씩 가중치의 업데이트가 이루어집니다.
- 일반적으로 모델에 표본을 하나씩 입력할 때보다 **배치**를 통해 여러 데이터를 동시에 입력할 때 데이터의 분포를 더욱 정확히 추정합니다. 배치가 클수록 더 좋은 추정치를 예측할 수 있지만 입력값이 많은 만큼 1회 업데이트를 위한 연산 과정이 오래 걸립니다. 배치가 크면 보통 평가나 예측도 빨라지기에 메모리가 허용하는 선에서 배치 크기를 가능한 크게 하는 것을 권장합니다.
- **에폭<sub>epoch</sub>**: 임의로 설정할 수 있는 횟수지만, 보통 "전체 데이터 세트를 1회 통과하는 것"으로 정의합니다. 훈련을 별개의 단계로 구분하는 데에 사용되며, 이는 결과의 기록이나 주기별 평가에 유용합니다.
  - 케라스 모델의 `fit` 메소드와 함께 `validation_data`, 혹은 `validation_split`을 사용하는 경우, 각 **에폭**이 끝날 때마다 평가가 실행됩니다.
  - 케라스에서는 **에폭**의 끝에서 실행되도록 특별히 고안된 [콜백](https://keras.io/callbacks/)을 추가할 수 있습니다. 예를 들어, 학습률 변화와 모델 체크포인트 저장 등이 있습니다.

---

### 케라스 모델은 어떻게 저장합니까?

#### 전체 모델 저장하기/불러오기 (구조 + 가중치 + 최적화 함수 상태)

*케라스 모델 저장에 pickle 또는 cPickle 사용은 추천하지 않습니다.*

`model.save(filepath)` 명령으로 케라스 모델을 하나의 HDF5 파일로 저장할 수 있습니다. 저장한 파일은 다음을 포함합니다. 

- 모델 재구축을 위한 모델 구조<sub>architecture</sub> 정보
- 모델의 가중치<sub>weight</sub>
- 모델의 학습 설정 (손실 함수<sub>loss function</sub>, 최적화 함수<sub>optimizer</sub>)
- 학습을 중단한 지점에서 다시 시작하기 위한 최적화 함수의 상태

`keras.models.load_model(filepath)` 명령으로 저장한 모델 인스턴스를 다시 불러올 수 있습니다. `load_model` 메소드는 저장된 설정값을 활용해서 컴파일까지 완료합니다 (아직 컴파일하지 않은 모델은 제외).

예:

```python
from keras.models import load_model

model.save('my_model.h5')  # 'my_model.h5'라는 HDF5 파일을 생성합니다.
del model  # 현재 메모리에 있는 모델을 삭제합니다.

# 컴파일 된 모델을 복원합니다.
# 이전의 모델과 완전히 동일합니다.
model = load_model('my_model.h5')
```

`h5py` 설치에 대한 설명은 [HDF5나 h5py를 설치해 케라스 모델을 저장하려면 어떻게 해야 합니까?](#hdf5나-h5py를-설치해-케라스-모델을-저장하려면-어떻게-해야-합니까)에 있습니다.

#### 모델 구조만 저장하기/불러오기

가중치나 훈련 정보 없이 **모델의 구조**만 저장하고자 한다면 다음의 예시를 따릅니다.

```python
# JSON 형태로 저장합니다.
json_string = model.to_json()

# YAML 형태로 저장합니다.
yaml_string = model.to_yaml()
```

생성된 JSON 또는 YAML 파일은 사람이 읽을 수 있으며 필요하면 직접 고치는 것도 가능합니다. 

저장한 구조 정보를 활용해 새로운 모델을 만들 수 있습니다.

```python
# JSON 형식의 저장된 구조를 불러와 새로운 모델을 만듭니다.
from keras.models import model_from_json
model = model_from_json(json_string)

# YAML 형식의 저장된 구조를 불러와 새로운 모델을 만듭니다.
from keras.models import model_from_yaml
model = model_from_yaml(yaml_string)
```

#### 모델의 가중치만 저장하기/불러오기

**모델의 가중치**만 저장하고자 한다면 다음의 예시와 같이 HDF5 형식을 이용합니다.

```python
model.save_weights('my_model_weights.h5')
```

모델 인스턴스를 가동할 코드가 있음을 전제로, 해당 모델과 *동일한* 구조의 모델에서 저장한 가중치를 불러올 수 있습니다.

```python
model.load_weights('my_model_weights.h5')
```

가중치를 몇 개의 층<sub>layer</sub>만 같은 *다른* 구조의 모델로 불러들일 경우, 예를 들어 미세 조정<sub>fine-tuning</sub>이나 전이 학습<sub>transfer learning</sub>을 사용코자 할 경우, 필요한 층을 *층의 이름*을 사용하여 불러올 수 있습니다.

```python
model.load_weights('my_model_weights.h5', by_name=True)
```

예:

```python
"""
원래의 모델이 다음과 같다고 가정합시다.
    model = Sequential()
    model.add(Dense(2, input_dim=3, name='dense_1'))
    model.add(Dense(3, name='dense_2'))
    ...
    model.save_weights(fname)
"""

# 새로운 모델
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # 불러올 층입니다.
model.add(Dense(10, name='new_dense'))  # 불러오지 않을 층입니다.

# 첫 번째 모델에서 저장한 가중치를 불러올 경우 이름이 같은 층인 dense_1에만 적용됩니다.
model.load_weights(fname, by_name=True)
```

`h5py` 설치에 대한 설명은 [HDF5나 h5py를 설치해 케라스 모델을 저장하려면 어떻게 해야 합니까?](#hdf5나-h5py를-설치해-케라스-모델을-저장하려면-어떻게-해야-합니까)에 있습니다.

#### 저장된 모델의 사용자 설정 층 및 객체 다루기

불러오려는 모델이 사용자가 설정한 층과 객체<sub>objects</sub>, 함수<sub>function</sub>를 포함하는 경우 `custom_objects` 인자<sub>arguments</sub>를 통해서 불러오기 함수로 넘길 수 있습니다. 

```python
from keras.models import load_model
# 모델이 "AttentionLayer" 클래스의 인스턴스를 포함하고 있다고 가정합시다.
model = load_model('my_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
```

Utils의 [custom object scope](https://keras.io/utils/#customobjectscope) 클래스를 활용하는 방법도 있습니다.

```python
from keras.utils import CustomObjectScope

with CustomObjectScope({'AttentionLayer': AttentionLayer}):
    model = load_model('my_model.h5')
```

`custom_object`인자는 모델을 불러오는 세 가지 함수(`load_model`, `model_from_json`, `model_from_yaml`) 모두에서 동일하게 작동합니다.

```python
from keras.models import model_from_json
model = model_from_json(json_string, custom_objects={'AttentionLayer': AttentionLayer})
```

---

### 왜 훈련 손실이 테스트 손실보다 훨씬 높습니까?

케라스 모델에는 학습과 시험의 두 가지 모드가 있습니다. 그 중 시험모드에서는 드롭아웃이나 L1/L2 가중치 규제화<sub>regularization</sub>같은 규제화 메커니즘을 작동하지 않습니다. 이것이 하나의 이유가 될 수 있습니다.  
  
그밖에도 훈련 손실<sub>loss</sub>은 훈련 데이터로부터 생성된 각각의 배치에서 구한 손실의 평균이라는 점을 생각해볼 수 있습니다. 모델의 파라미터는 매 배치의 훈련이 끝날 때마다 바뀌기 때문에 한 에폭 안에서도 초기의 배치에서 나온 손실은 같은 에폭의 마지막 배치 손실보다 크기 마련입니다. 반면 시험 손실은 각 에폭이 끝난 다음에 측정되기 때문에 비교적 낮은 손실을 기록할 때도 있습니다.

---

### 중간층의 출력은 어떻게 얻을 수 있습니까?

가장 간단한 방법은 원하는 층을 출력하는 `Model`을 하나 만드는 것입니다.

```python
from keras.models import Model

model = ...  # 원래의 모델을 만듭니다.

layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
```

또는 데이터를 입력했을 때 특정 층에서 출력을 반환하는 케라스 함수를 만드는 것도 방법입니다.

```python
from keras import backend as K

# Sequential 모델을 이용합니다
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([x])[0]
```

비슷한 방식으로, Theano 또는 TensorFlow 함수를 직접 만들 수도 있습니다.  
  
만약 학습과 시험 모드에서 서로 다르게 작동하는 모델을 사용하고 있다면(예: `Dropout`, `BatchNormalization` 등) 만들려는 함수가 학습 상태(`K.learning_phase()`)로 작동하도록 지정해야 합니다. 

```python
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])

# 시험 모드의 출력을 원할 경우 = 0
layer_output = get_3rd_layer_output([x, 0])[0]

# 훈련 모드의 출력을 원할 경우 = 1
layer_output = get_3rd_layer_output([x, 1])[0]
```

---

### 메모리를 초과하는 데이터 세트에 케라스를 사용하려면 어떻게 해야 합니까?

`model.train_on_batch(x, y)`와 `model.test_on_batch(x, y)`를 사용하면 배치 단위로 학습시킬 수 있습니다. 자세한 내용은 [Sequential 모델 API](/models/sequential)문서에 있습니다.

그 밖에도 훈련 데이터의 배치를 생성하는 제너레이터 함수를 만들어서 `model.fit_generator(data_generator, steps_per_epoch, epochs)`메소드에 적용하는 방법도 있습니다.  

[CIFAR10 예시](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py)에서 배치 훈련 과정을 보실 수 있습니다.

---

### 검증 손실이 더이상 감소하지 않는 경우 훈련을 어떻게 중단할 수 있습니까?

`EarlyStopping`콜백을 사용합니다.

```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
```

더 자세한 내용은 [콜백](/callbacks) 문서를 참고하시면 됩니다.

---

### 검증 데이터 분리는 어떻게 계산됩니까?

예를 들어 `model.fit` 안의 `validation_split`인자를 `0.1`로 지정하는 경우 입력 데이터의 *마지막 10%가* 검증 데이터로 활용됩니다. `0.25`로 지정하면 마지막 25%가 검증 데이터가 됩니다. 검증 데이터를 분리하기 전에 별도로 뒤섞는 과정이 없으므로 말 그대로 입력 가운데 *마지막 x%* 표본이 검증 데이터가 된다는 것에 주의해야 합니다.  

동일한 `fit` 호출 안에서는 같은 검증 세트가 사용됩니다.

---

### 훈련 중 데이터의 순서가 섞입니까?

`model.fit`의 `shuffle`인자를 `True`로 설정할 경우(기본값) 각 에폭마다 훈련 데이터는 무작위로 뒤섞입니다. 

검증 데이터는 뒤섞이지 않습니다.

---


### 각 에폭별 훈련 손실 / 검증 손실 / 정확도는 어떻게 기록할 수 있습니까?

`model.fit`메소드는 `History`콜백을 통해 연속되는 손실값 및 사용자가 지정한 평가지표들이 기록된 리스트 형태의 `history` 속성을 출력합니다.

```python
hist = model.fit(x, y, validation_split=0.2)
print(hist.history)
```

---

### 특정한 층의 가중치를 고정시키려면 어떻게 해야 합니까?

특정한 층을 고정, 또는 동결<sub>freeze</sub>시킬 경우 그 층의 가중치는 학습되지 않습니다. 모델에 미세 조정을 적용하려는 경우나 사전에 학습된 단어의 임베딩을 가져와 사용하려는 경우 등에서 층의 가중치를 고정시키게 됩니다. 층을 정의할 때 `bool`값의 `trainable`인자를 `False`로 지정하면 층의 가중치를 고정시킬 수 있습니다.

```python
frozen_layer = Dense(32, trainable=False)
```

참고로 모델 인스턴스 객체를 만든 다음에도 `trainable`인자를 수정할 수 있습니다. 이때 바뀐 `trainable`인자를 적용하기 위해서는 다시 `compile()`을 실행시켜서 적용해야 합니다. 자세한 예는 아래에서 볼 수 있습니다. 

```python
x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)

frozen_model = Model(x, y)
# 아래 모델에서 `layer`층의 가중치는 학습 과정 중 업데이트 되지 않습니다.
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = True
trainable_model = Model(x, y)
# 이 모델에는 가중치를 학습하도록 설정을 변경한 `layer`층이 적용됩니다.
# 이 상태에서 `frozen_model`을 다시 `compile`하면 `frozen_model`의 `layer`층도 학습 가능한 상태로 바뀝니다.
# 하지만 본 예시에서는 `frozen_model`을 다시 `compile`하지 않았으므로 `frozen_model`의 `layer`층은 고정된 상태를 유지합니다. 
trainable_model.compile(optimizer='rmsprop', loss='mse')
frozen_model.fit(data, labels)  # 이 모델은 `layer`층의 가중치를 업데이트하지 않습니다.
trainable_model.fit(data, labels)  # 이 모델은 layer`층의 가중치를 업데이트 합니다.
```

---

### 상태 저장 순환 신경망을 사용하려면 어떻게 해야 합니까?

상태 저장 순환 신경망<sub>stateful RNN</sub>은 이전 배치 표본들로부터 얻은 마지막 상태<sub>state</sub>를 다음 배치의 초기 상태로 이어서 사용하는 순환 신경망입니다.  
  
상태 저장 순환 신경망 사용시에는 다음과 같은 가정이 전제됩니다.  

- 모든 배치의 표본 갯수가 같아야 합니다.
- 만약 `x1`과 `x2`가 서로 이어지는 배치일 경우, 모든 시계열 표본의 인덱스 `i`에 대해서 `x2[i]`는 `x1[i]`의 바로 다음에 이어지는 순서값이어야 합니다.  

상태 저장 순환 신경망을 사용하려면 다음의 설정을 따라야 합니다.  

- 모델의 첫 입력층에 `batch_size`인자를 사용해서 배치의 크기를 직접 지정해야 합니다. 예를 들어, 10단계의 시계열과 각 단계당 16개의 요인<sub>feature</sub>을 가진 데이터를 32개의 배치로 분할코자 할 경우 `batch_size=32`로 설정합니다.
- 해당 순환 신경망 층에서 `stateful=True`로 지정합니다.
- 학습을 위해 `fit()` 메소드 실행시 `shuffle=False`로 지정하여 배치가 뒤섞이지 않도록 합니다.  

이전 단계로부터 누적된 상태값을 초기화하고자 하는 경우 다음의 코드를 실행합니다.

- 모델 내의 모든 층의 상태값을 초기화할 경우 `model.reset_states()`를 실행합니다. 
- 특정 상태 저장 순환 신경망 층의 상태값만을 초기화 할 경우 `layer.reset_states()`를 실행합니다. 

예:

```python
x  # (32, 21, 16)의 형태를 가진 입력 데이터라고 가정합시다.
# 10단계의 시계열을 가진 모델에 이 데이터를 적용합니다.

model = Sequential()
model.add(LSTM(32, input_shape=(10, 16), batch_size=32, stateful=True))
model.add(Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 처음 10단계의 입력을 이용해서 11번째 단계의 값을 추정하도록 모델을 학습시킵니다.
model.train_on_batch(x[:, :10, :], np.reshape(x[:, 10, :], (32, 16)))

# 신경망의 상태값이 바뀌었으므로 이를 이용하여 이어지는 단계의 값을 훈련시킵니다. 
model.train_on_batch(x[:, 10:20, :], np.reshape(x[:, 20, :], (32, 16)))

# 모델 안의 LSTM 층의 상태값을 초기화합니다.
model.reset_states()

# 또 다른 방법으로, 이와 같이 층을 직접 지정해서 상태값을 초기화합니다.
model.layers[0].reset_states()
```

`predict`, `fit`, `train_on_batch`, `predict_classes`등의 메소드들은 *모두* 상태 저장 층의 상태값을 업데이트 한다는 점을 유의하시기 바랍니다. 따라서 상태 저장의 형태로 학습시키는 것 뿐만 아니라 상태 저장을 적용하여 예측값을 얻을 수 있습니다.

---

### Sequential 모델에서 층을 없애려면 어떻게 해야 합니까?

`.pop()` 메소드를 사용하면 현재 Sequential 모델의 가장 마지막 층을 제거할 수 있습니다.

```python
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(32, activation='relu'))

print(len(model.layers))  # "2"

model.pop()
print(len(model.layers))  # "1"
```

---

### 케라스에서 선행 학습된 모델을 사용하려면 어떻게 해야 합니까?

케라스는 아래 이미지 분류 모델에 대한 코드 및 선행 학습된 모델을 제공합니다.

- Xception
- VGG16
- VGG19
- ResNet
- ResNet v2
- ResNeXt
- Inception v3
- Inception-ResNet v2
- MobileNet v1
- MobileNet v2
- DenseNet
- NASNet

케라스가 제공하는 학습된 모델은 `keras.applications`모듈로부터 불러올 수 있습니다.

```python
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50
from keras.applications.resnet import ResNet101
from keras.applications.resnet import ResNet152
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.resnet_v2 import ResNet101V2
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.resnext import ResNeXt50
from keras.applications.resnext import ResNeXt101
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import NASNetMobile

model = VGG16(weights='imagenet', include_top=True)
```

[어플리케이션](/applications) 모듈 문서에 간단한 사용 방법이 나와있습니다.

제시된 모듈을 요인 추출<sub>feature extraction</sub> 또는 미세 조정과 같은 방식으로 활용하는 보다 자세한 예시는, 다음의 [블로그 문서](http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)를 참고하시면 됩니다.

특히 VGG16 모델은 다양한 케라스 예제의 기초로 사용되고 있습니다.

- [Style transfer](https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py)
- [Feature visualization](https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py)
- [Deep dream](https://github.com/keras-team/keras/blob/master/examples/deep_dream.py)

---

### 케라스에서 HDF5 입력을 사용하려면 어떻게 해야 합니까?

`keras.utils`모듈의 `HDF5Matrix`클래스를 사용합니다. 자세한 내용은 [HDF5Matrix](/utils/#hdf5matrix) 문서에 있습니다.

또한 다음과 같은 방법으로 HDF5 데이터 세트를 직접 사용할 수 있습니다.

```python
import h5py
with h5py.File('input/file.hdf5', 'r') as f:
    x_data = f['x_data']
    model.predict(x_data)
```

`h5py` 설치에 대한 설명은 [HDF5나 h5py를 설치해 케라스 모델을 저장하려면 어떻게 해야 합니까?](#hdf5나-h5py를-설치해-케라스-모델을-저장하려면-어떻게-해야-합니까)에 있습니다.

---

### 케라스 구성 파일은 어디에 저장됩니까?

케라스 데이터가 저장되는 디렉토리 기본값은 다음과 같습니다.

```bash
$HOME/.keras/
```

마이크로소프트 윈도우 사용자의 경우 `$HOME` 대신 `%USERPROFILE%`이 케라스의 상위 폴더 주소가 됩니다.
만약 사용자 권한 문제 등으로 케라스가 위의 디렉토리를 생성하지 못하는 경우 `/tmp/.keras/` 폴더가 백업으로 사용됩니다.

케라스 설정은 JSON 형식의 `$HOME/.keras/keras.json`에 저장됩니다. 설정 기본값은 다음과 같습니다.

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

설정 파일에는 이하 영역들의 값이 정의되어 있습니다.

- 각 층과 도구에서 사용될 이미지 데이터 형식의 기본값으로 `channels_last`인지 `channels_first`지를 설정.
- 일부 연산에서 값을 `0`으로 나누어 발생하는 오류를 방지하기 위해 더할 작은 `epsilon`값의 크기.
- 부동소수점<sub>float</sub> 데이터의 기본 타입.
- 케라스가 사용할 기본 백엔드. 자세한 사항은 [케라스 백엔드](/backend) 문서에 있습니다.

[`get_file()`](/utils/#get_file) 등의 명령으로 다운로드한 임시로 저장된 데이터 파일의 경우도 마찬가지로 `$HOME/.keras/datasets/`폴더에 저장됩니다.

---

### 개발 중 케라스를 사용해 재현 가능한 결과를 얻으려면 어떻게 해야 합니까?

모델을 개발하다보면 때때로 성능의 변화가 실제 모델 또는 데이터를 수정한 결과인지, 아니면 단순한 표본의 무작위 추출에 따른 결과인지 확인하기 위해 각 실행의 사이마다 재현 가능한 상태를 확보할 필요가 있습니다. 이 경우, 먼저 프로그램을 시작하기 전에(프로그램 내부에서가 아닙니다) `PYTHONHASHSEED` 환경 변수를 `0`으로 설정해야 합니다. 이는 3.2.3 이후 버전의 파이썬에서 세트 또는 딕셔너리 내의 개체의 순서와 같은 특정한 해시 기반 연산을 재현 가능케 하는데 필수적입니다 (자세한 내용은 [Python's documentation](https://docs.python.org/3.7/using/cmdline.html#envvar-PYTHONHASHSEED) 또는 [issue #2280](https://github.com/keras-team/keras/issues/2280#issuecomment-306959926)에 있습니다). 파이썬 시작 시 환경 변수를 설정하는 방법에는 다음이 있습니다.

```
$ cat test_hash.py
print(hash("keras"))
$ python3 test_hash.py                  # 재현 불가능한 해시 (Python 3.2.3+)
-8127205062320133199
$ python3 test_hash.py                  # 재현 불가능한 해시 (Python 3.2.3+)
3204480642156461591
$ PYTHONHASHSEED=0 python3 test_hash.py # 재현 가능한 해시
4883664951434749476
$ PYTHONHASHSEED=0 python3 test_hash.py # 재현 가능한 해시
4883664951434749476
```

그리고 TensorFlow 백엔드를 활용하여 GPU 연산을 할 경우 일부 연산은 다른 실행에서 다른 결과를 가져오는 비결정적<sub>non-deterministic</sub> 특징을 갖고 있습니다. 특히 `tf.reduce_sum()`과 같은 연산이 이에 해당됩니다. 이는 GPU가 많은 연산을 병렬로 처리하면서 그 실행 순서를 동일하게 보장하지 않기 때문입니다. 여기에 `float`자료형의 제한된 정확도 때문에 서로 다른 수치를 더하는 것조차 순서에 따라 아주 조금씩 다른 결과를 가져오기도 합니다. GPU를 사용하면서 비결정적인 연산을 피하고자 하더라도 일부는 그래디언트 계산 과정에서 TensorFlow에 의해 자동으로 생성되기 때문에 그냥 CPU를 활용하는 편이 훨씬 간편합니다. CPU만을 활용하고자 한다면 다음 예시와 같이 `CUDA_VISIBLE_DEVICES`환경변수를 비워둡니다.   

```
$ CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python your_program.py
```

아래 코드는 재현 가능한 결과를 얻는 방법의 예시입니다. 파이썬 3 및 텐서플로우 사용 환경에 맞춰져 있습니다. 

```python
import numpy as np
import tensorflow as tf
import random as rn

# 아래 줄은 NumPy가 재현 가능한 방식으로 무작위 숫자를 만들어내는 데 필요합니다. 

np.random.seed(42)

# 아래 줄은 파이썬 코어가 재현 가능한 방식으로 무작위 숫자를 만들어내는 데 필요합니다.

rn.seed(12345)

# TensorFlow로 하여금 단일 스레드만을 사용하도록 강제합니다.
# 다중 스레드를 사용할 경우 재현 불가능한 결과를 얻을 가능성이 있습니다.
# 이에 대한 자세한 내용은 https://stackoverflow.com/questions/42022950/에 있습니다.

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K

# 아래의 tf.set_random_seed()는 TensorFlow 백엔드가 재현 가능한 방식으로 무작위 숫자를 만들어내도록 합니다.
# 이에 대한 자세한 내용은 다음을 참고합니다.
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# 이 다음에 필요한 코드를 연결합니다 ...
```

---

### HDF5나 h5py를 설치해 케라스 모델을 저장하려면 어떻게 해야 합니까?

케라스 모델을 HDF5 형식으로 저장하고자 하는 경우 (예: `keras.callbacks.ModelCheckpoint` 등을 사용할 때), 케라스는 파이썬의 h5py 패키지를 이용합니다. h5py 패키지는 케라스의 의존성 목록에 있기 때문에 일반적으로 케라스 설치와 함께 자동으로 설치됩니다. 단, Debian 기반의 배포판에서는 아래와 같이 추가로 `libhdf5`를 설치해야 정상적으로 작동합니다.

```
sudo apt-get install libhdf5-serial-dev
```
h5py 모듈이 설치되었는지 확인하기 위해서는 파이썬 셸을 열어 다음 명령으로 모듈을 불러오면 됩니다. 

```
import h5py
```
오류 없이 불러오는 데 성공했다면 정상적으로 설치가 된 것입니다. 오류가 발생하는 경우 다음의 주소에서 해결 방법을 찾을 수 있습니다. http://docs.h5py.org/en/latest/build.html
