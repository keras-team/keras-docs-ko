# # 케라스 FAQ: 자주 묻는 케라스 질문

- [케라스는 어떻게 인용하나요?](#how-should-i-cite-keras)
- [케라스를 GPU에 실행하려면 어떻게 하나요?](#how-can-i-run-keras-on-gpu)
- [여러 개의 GPU에 케라스를 실행하려면 어떻게 하나요?](#how-can-i-run-a-keras-model-on-multiple-gpus)
- ["샘플", "배치", "세대"는 무슨 뜻인가요?](#what-does-sample-batch-epoch-mean)
- [케라스 모델은 어떻게 저장하나요?](#how-can-i-save-a-keras-model)
- [어째서 학습 손실이 테스트 손실보다 훨씬 높은가요?](#why-is-the-training-loss-much-higher-than-the-testing-loss)
- [중간 레이어의 아웃풋은 어떻게 얻을 수 있나요?](#how-can-i-obtain-the-output-of-an-intermediate-layer)
- [메모리를 초과하는 데이터셋에 케라스를 사용하려면 어떻게 하나요?](#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory)
- [검증 손실이 더 이상 감소하지 않는 경우 학습을 어떻게 중단할 수 있나요?](#how-can-i-interrupt-training-when-the-validation-loss-isnt-decreasing-anymore)
- [검증 분리는 어떻게 계산되나요?](#how-is-the-validation-split-computed)
- [학습 중 데이터가 섞이나요?](#is-the-data-shuffled-during-training)
- [각 세대에서의 학습 손실 / 검증 손실 / 정확성은 어떻게 기록하나요?](#how-can-i-record-the-training--validation-loss--accuracy-at-each-epoch)
- [레이어는 어떻게 "동결"시킬 수 있나요?](#how-can-i-freeze-keras-layers)
- [상태형 순환 신경망은 어떻게 사용할 수 있나요?](#how-can-i-use-stateful-rnns)
- [Sequential 모델에서 레이어를 없애려면 어떻게 하나요?](#how-can-i-remove-a-layer-from-a-sequential-model)
- [케라스에서 선행 학습된 모델을 사용하려면 어떻게 하나요?](#how-can-i-use-pre-trained-models-in-keras)
- [케라스에서 HDF5 인풋은 어떻게 사용하나요?](#how-can-i-use-hdf5-inputs-with-keras)
- [케라스 구성 파일은 어디에 저장되어 있나요?](#where-is-the-keras-configuration-file-stored)
- [개발 중 케라스를 사용해 재현 가능한 결과를 얻으려면 어떻게 하나요?](#how-can-i-obtain-reproducible-results-using-keras-during-development)
- [HDF5나 h5py를 설치해 케라스 모델을 저장하려면 어떻게 하나요?](#how-can-i-install-hdf5-or-h5py-to-save-my-models-in-keras)

---

### 케라스는 어떻게 인용하나요?

연구 중 케라스가 도움이 되었다면 케라스를 인용해 주십시오. 다음은 BibTeX 등재 예시입니다:

```
@misc{chollet2015keras,
  title={Keras},
  author={Chollet, Fran\c{c}ois and others},
  year={2015},
  howpublished={\url{https://keras.io}},
}
```

---

### 케라스를 GPU에 실행하려면 어떻게 하나요?

**텐서플로우** 혹은 **CNTK** 백엔드를 사용하는 경우, 사용 가능한 GPU가 감지되면 코드가 자동적으로 GPU에서 작동됩니다.

**Theano** 백엔드를 사용하는 경우, 다음 방법 중 하나를 사용하시면 됩니다:

**방법 1**: Theano 플래그를 사용합니다.
```bash
THEANO_FLAGS=device=gpu,floatX=float32 python my_keras_script.py
```

본인 장치의 식별자에 따라 'gpu'를 바꾸어야 할 수도 있습니다 (예. `gpu0`, `gpu1`, 등).

**방법 2**: `.theanorc`를 설정합니다: [설명](http://deeplearning.net/software/theano/library/config.html).

**방법 3**: 코드 첫 부분에 수동으로 `theano.config.device`와 `theano.config.floatX`를 설정합니다:
```python
import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'
```

---

### 여러 개의 GPU에 케라스를 실행하려면 어떻게 하나요?

이 용도로는 **텐서플로우** 백엔드를 사용하실 것을 권장합니다. 단일 모델을 여러 GPU에 실행하려면 두 가지 방법이 있습니다: **데이터 병렬처리**와 **장치 병렬처리**.

대부분의 경우, 데이터 병렬처리를 사용하면 됩니다.

#### 데이터 병렬처리

데이터 병렬처리는, 표적 모델을 각 장치에 복제하고, 각 복제본을 사용해 인풋 데이터의 각기 다른 부분을 처리하는 방식으로 구성됩니다.
케라스의 내장 유틸리티 `keras.utils.multi_gpu_model`은 어떤 모델도 병렬 데이터 버전으로 만들 수 있으며, 8개의 GPU까지는 유사-선형적 속도 증가를 보입니다.

보다 자세한 정보는 [multi_gpu_model](/utils/#multi_gpu_model)의 설명서를 참고하십시오. 다음은 간략한 예시입니다:

```python
from keras.utils import multi_gpu_model

# `model`을 8개의 GPU에 복제합니다.
# 이 예시는 장치에 8개의 GPU가 구비되어 있다고 가정합니다.
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# `fit` 호출이 8개의 GPU에 분산됩니다.
# 배치 크기가 256이므로 각 GPU는 32개의 샘플을 처리합니다.
parallel_model.fit(x, y, epochs=20, batch_size=256)
```

#### 장치 병렬처리

장치 병렬처리는 동일한 모델의 여러 가지 부분을 각기 다른 장치에서 실행하는 방식입니다. 두가지로 구성된 모델 등과 같이 병렬 구조를 가진 모델에 대해 효과가 뛰어납니다.

텐서플로우 장치 범위를 사용해서 이를 구현할 수 있습니다. 다음은 간략한 예시입니다:

```python
# 공유된 장단기 메모리가 두 가지 상이한 시퀀스를 병렬로 인코딩하는 모델
input_a = keras.Input(shape=(140, 256))
input_b = keras.Input(shape=(140, 256))

shared_lstm = keras.layers.LSTM(64)

# 첫 번째 시퀀스를 한 GPU에 처리합니다
with tf.device_scope('/gpu:0'):
    encoded_a = shared_lstm(tweet_a)
# 다음 시퀀스를 다른 GPU에 처리합니다
with tf.device_scope('/gpu:1'):
    encoded_b = shared_lstm(tweet_b)

# 결과를 CPU에서 연결합니다
with tf.device_scope('/cpu:0'):
    merged_vector = keras.layers.concatenate([encoded_a, encoded_b],
                                             axis=-1)
```

---

### "샘플", "배치", "세대"는 무슨 뜻인가요?

아래에서 케라스를 올바로 이용하기 위해 알아두어야 할 몇 가지 정의를 확인하십시오:

- **샘플**: 데이터셋의 성분 하나.
  - *예시:* 하나의 이미지는 컨볼루션 망의 **샘플**입니다
  - *예시:* 구어 인식 모델의 경우 오디오 파일 하나가 **샘플**입니다.
- **배치**: *N* 샘플의 세트. **배치**의 샘플은 독립적으로 동시에 처리됩니다. 학습한다면, 배치 하나는 모델을 1회만 업데이트합니다.
  - **배치**는 일반적으로 단일 인풋에 비해 인풋 데이터의 분포를 보다 정확히 추정합니다. 배치가 클수록, 추정치도 개선됩니다; 하지만 단지 업데이트 1회를 위해서 배치를 처리하는데 오랜 시간을 소모하는 것도 사실입니다. 유추(평가/예측)의 목적으로는 메모리를 초과하지 않는 선에서 배치 크기를 가능한 크게 하는 것이 권장됩니다 (배치가 크면 보통 평가/예측도 빨라지기 때문입니다).
- **세대**: 임의의 차단 장치로, 보통 "전체 데이터셋 1회 통과"로 정의됩니다. 학습을 뚜렷한 국면 단위로 나누는데 사용되며, 이는 로그 기록이나 정기적 평가에 유용합니다.
  - 케라스 모델의 `fit` 메서드와 함께 `validation_data`, 혹은 `validation_split`을 사용하는 경우, 모든 **세대**의 끝에서 평가가 실행됩니다.
  케라스는 **세대**의 끝에서 실행토록 특별히 고안된 [콜백](https://keras.io/callbacks/)을 추가할 수 있는 기능을 제공합니다. 학습 속도 변화와 모델 체크포인트(저장)가 그 사례입니다.

---

### 케라스 모델은 어떻게 저장하나요?

#### 전체 모델 저장하기/불러오기 (구조 + 가중치 + 옵티마이저 상태)

*pickle이나 cPickle을 사용하여 케라스 모델을 저장하는 것은 권장하지 않습니다.*

`model.save(filepath)`을 사용하여, 다음 요소를 갖는 하나의 HDF5 파일로 케라스 모델을 저장할 수 있습니다:

- 모델 재현에 사용할 모델의 구조
- 모델의 가중치
- 학습 구성 (손실, 옵티마이저)
- 학습을 멈춘 지점에서 재개할 수 있도록 해주는 옵티마이저의 상태

`keras.models.load_model(filepath)`을 사용해서 모델을 재인스턴스화할 수 있습니다.
`load_model`은 (모델이 애초에 컴파일되지 않았던 것이 아니라면) 저장된 학습 구성을 사용해서 모델 컴파일을 해결해 줍니다.

예시:

```python
from keras.models import load_model

model.save('my_model.h5')  # HDF5 파일 'my_model.h5'를 만듭니다
del model  # 기존의 모델을 삭제합니다

# 전의 모델과 동일한
# 컴파일된 모델을 반환합니다
model = load_model('my_model.h5')
```

`h5py`의 설치 방법에 대한 설명은 [HDF5나 h5py를 설치해 케라스 모델을 저장하려면 어떻게 하나요?](#how-can-i-install-hdf5-or-h5py-to-save-my-models-in-keras)를 참조해 주시기 바랍니다.

#### 모델의 구조만 저장하기/불러오기

모델의 가중치나 학습 구성이 아닌 **모델의 구조**만 저장해야 한다면, 다음과 같이 하면 됩니다:

```python
# JSON으로 저장합니다
json_string = model.to_json()

# YAML로 저장합니다
yaml_string = model.to_yaml()
```

생성된 JSON / YAML 파일은 사람이 해독 가능하며 필요에 따라 수동으로 수정할 수 있습니다.

이제 이 데이터를 이용해 새 모델을 만들 수 있습니다:

```python
# JSON으로부터 모델 재건축:
from keras.models import model_from_json
model = model_from_json(json_string)

# YAML로부터 모델 재건축:
from keras.models import model_from_yaml
model = model_from_yaml(yaml_string)
```

#### 모델의 가중치만 저장하기/불러오기

**모델의 가중치**만 저장해야 하는 경우, 밑의 코드로 HDF5를 사용해서 가중치를 저장할 수 있습니다:

```python
model.save_weights('my_model_weights.h5')
```

모델을 인스턴스화시키는 코드가 있다고 가정하면, 저장했던 가중치를 *동일한* 구조의 모델에 불러올 수 있습니다:

```python
model.load_weights('my_model_weights.h5')
```

가중치를 (몇몇 레이어만 공유하는) *다른* 구조에 불러 미세 조정을 하거나 전이 학습을 하려면 *레이어 이름*을 통해서 가중치를 불러와야 합니다:

```python
model.load_weights('my_model_weights.h5', by_name=True)
```

예시:

```python
"""
본래 모델이 다음과 같다고 가정합니다:
    model = Sequential()
    model.add(Dense(2, input_dim=3, name='dense_1'))
    model.add(Dense(3, name='dense_2'))
    ...
    model.save_weights(fname)
"""

# 새로운 모델
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # 불러옵니다
model.add(Dense(10, name='new_dense'))  # 불러올 수 없습니다

# 첫 번째 모델에서 가중치를 불러옵니다; 이는 첫 레이어인 dense_1에만 영향을 미칩니다
model.load_weights(fname, by_name=True)
```

`h5py`의 설치 방법에 대한 설명은 [HDF5나 h5py를 설치해 케라스 모델을 저장하려면 어떻게 하나요?](#how-can-i-install-hdf5-or-h5py-to-save-my-models-in-keras)를 참조해 주시기 바랍니다.

#### 저장된 모델에서 커스텀 레이어 (혹은 다른 커스텀 객체) 다루는 방법

불러오려는 모델에 커스텀 레이어나 다른 커스텀 클래스, 함수 등이 포함된 경우, 
`custom_objects` 인수를 통해서 로딩 메커니즘에 커스텀 객체를 전달할 수 있습니다: 

```python
from keras.models import load_model
# 모델이 "AttentionLayer" 클래스의 인스턴스를 포함한다고 가정합니다
model = load_model('my_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
```

혹은 [커스텀 객체 범위](https://keras.io/utils/#customobjectscope)를 사용하는 방법도 있습니다:

```python
from keras.utils import CustomObjectScope

with CustomObjectScope({'AttentionLayer': AttentionLayer}):
    model = load_model('my_model.h5')
```

`load_model`, `model_from_json`, `model_from_yaml`에 대해서도 같은 방식으로 커스텀 객체 조작을 할 수 있습니다:

```python
from keras.models import model_from_json
model = model_from_json(json_string, custom_objects={'AttentionLayer': AttentionLayer})
```

---

### 어째서 학습 손실이 테스트 손실보다 훨씬 높은가요?

케라스 모델에는 학습과 테스트의 두 가지 모드가 있습니다. 테스트 과정에서는 드롭아웃이나 L1/L2 가중치 정규화와 같은 정규화 메커니즘을 비활성화시키게 됩니다.

뿐만 아니라 학습 손실은 학습 데이터의 각 배치에 대한 손실의 평균입니다. 모델이 시간에 걸쳐 서서히 변하기 때문에, 세대 초반의 배치에 대한 손실이 후반의 배치에 대한 손실에 비해 높기 마련입니다. 그에 반해 한 세대의 테스트 손실은, 세대의 끝의 모델 상태에 기반해서 계산되기 때문에, 보다 낮은 손실이 발생합니다.

---

### 중간 레이어의 아웃풋은 어떻게 얻을 수 있나요?

간단한 방법의 하나는 새로운 `Model`을 만들어 원하는 레이어를 출력하도록 하는 것입니다:

```python
from keras.models import Model

model = ...  # 원래 모델을 생성합니다

layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
```

특정 인풋을 주었을 때 특정 레이어의 아웃풋을 반환하는 케라스 함수를 만드는 것 또한 대안이 될 수 있습니다. 다음은 예시입니다:

```python
from keras import backend as K

# Sequential 모델 사용
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([x])[0]
```

비슷한 방식으로, Theano와 텐서플로우 함수를 직접 만드는 수도 있습니다.

모델이 학습과 테스트 구간에서 각기 다르게 작동하도록 만들어진 경우 (예. `Dropout`이나 `BatchNormalization` 등을 사용하는 경우), 함수에 학습 구간 플래그를 전달해야 합니다.

```python
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])

# 테스트 모드의 아웃풋 = 0
layer_output = get_3rd_layer_output([x, 0])[0]

# 학습 모드의 아웃풋 = 1
layer_output = get_3rd_layer_output([x, 1])[0]
```

---

### 메모리를 초과하는 데이터셋에 케라스를 사용하려면 어떻게 하나요?

`model.train_on_batch(x, y)`와 `model.test_on_batch(x, y)`를 사용해서 배치 학습을 할 수 있습니다. [모델 설명서](/models/sequential)를 참고하십시오.

또는 학습 데이터 배치를 만들어 내는 생성기를 작성하여 `model.fit_generator(data_generator, steps_per_epoch, epochs)` 메서드를 사용하는 방법도 있습니다.

[CIFAR10 예시](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py)에서 실제로 배치 학습이 어떻게 적용되는지 확인 할 수 있습니다.

---

### 검증 손실이 더 이상 감소하지 않는 경우 학습을 어떻게 중단할 수 있나요?

`EarlyStopping` 콜백을 사용하면 됩니다:

```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
```

더 자세한 사항은 [콜백 설명서](/callbacks)를 참고하십시오.

---

### 검증 분리는 어떻게 계산되나요?

`model.fit`의 `validation_split` 인수를, 예를 들어, 0.1로 설정했다면, 데이터의 *마지막 10%* 를 검증 데이터로 사용하게 됩니다. 0.25로 설정한 경우 데이터의 마지막 25%가 되는 식입니다. 검증 분리를 추출하기 전에 데이터가 섞이지 않는다는 점을 참고하십시오. 검증 데이터는 말 그대로 전달한 인풋의 샘플 중 *마지막* x%일 뿐입니다.

(동일한 `fit` 호출 내에서는) 모든 세대에 대해 동일한 검증 세트가 사용됩니다.

---

### 학습 중 데이터가 섞이나요?

`model.fit`의 `shuffle` 인수가 `True`로 설정된 경우 (이는 디폴트 값입니다), 학습 데이터는 각 세대에서 임의로 섞입니다.

검증 데이터는 절대로 섞이지 않습니다.

---


### 각 세대에서의 학습 손실 / 검증 손실 / 정확성은 어떻게 기록하나요?

`model.fit` 메서드는 `History` 콜백을 반환하고, `History` 콜백의 `history` 속성은 연속되는 손실과 다른 측정 항목의 리스트를 담습니다.

```python
hist = model.fit(x, y, validation_split=0.2)
print(hist.history)
```

---

### 레이어는 어떻게 "동결"시킬 수 있나요?

레이어를 "동결"시킨다는 것은 학습에서 제외한다는 뜻으로, 다시 말해 가중치가 최신화되지 않는다는 의미입니다. 이는 모델을 미세-조정하는 경우나 텍스트 인풋에 대해 고정 임베딩을 사용하는 경우에 유용합니다.

레이어를 학습 불가로 설정하려면 `trainable` 인수(불리언)를 레이어의 생성자에 전달하면 됩니다:

```python
frozen_layer = Dense(32, trainable=False)
```

또한 레이어의 인스턴스화 후에 `trainable` 속성을 True`나 `False`로 설정할 수 있습니다. `trainable` 속성을 수정한 후 모델에 대해 `compile()`을 호출해야 수정 사항이 반영됩니다. 다음은 예시입니다:

```python
x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)

frozen_model = Model(x, y)
# 아래 모델에서는 `layer`의 가중치가 학습 도중 최신화되지 않습니다
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = True
trainable_model = Model(x, y)
# 다음 모델에서는 학습 중 레이어의 가중치가 최신화됩니다
# (이는 동일한 레이어 인스턴스를 사용하는 위의 모델에도 영향을 끼칩니다)
trainable_model.compile(optimizer='rmsprop', loss='mse')

frozen_model.fit(data, labels)  # `layer`의 가중치를 최신화하지 않습니다
trainable_model.fit(data, labels)  # `layer`의 가중치를 최신화합니다
```

---

### 상태형 순환 신경망은 어떻게 사용할 수 있나요?

순환 신경망을 상태형으로 만든다는 것은 각 배치의 샘플의 상태가 다음 배치의 샘플의 초기 상태로 재사용된다는 의미입니다.

따라서 상태형 순환 신경망을 사용하는 경우 다음을 가정합니다:

- 모든 배치가 같은 수의 샘플을 갖습니다
- `x1`와 `x2`가 연속된 배치의 샘플이라면, `x2[i]`는 모든 `i`에 대해서 `x1[i]`의 후속 시퀀스입니다.

순환 신경망에서 상태성을 사용하려면 다음이 필요합니다:

- `batch_size` 인수를 모델의 첫 레이어에 전달해서 배치 크기를 명확히 특정해야 합니다. 예. 시간 단계 당 16 특성과 10개의 시간 단계로 이루어진 시퀀스의 32-샘플 배치의 경우 `batch_size=32`가 됩니다.
- 순환 신경망에서 `stateful=True`로 설정하십시오.
- `fit()`을 호출할 때 `shuffle=False`로 특정하십시오.

축정된 상태를 다음과 같이 재설정할 수 있습니다:

- `model.reset_states()`를 사용해 모델 내 모든 레이어의 상태를 재설정하십시오
- `layer.reset_states()`를 사용해 특정 상태형 순환 신경망 레이어의 상태를 재설정하십시오

예시:

```python
x  # (32, 21, 16) 형태의 인풋 데이터입니다
# 길이 10인 시퀀스로 모델에 전달합니다

model = Sequential()
model.add(LSTM(32, input_shape=(10, 16), batch_size=32, stateful=True))
model.add(Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 첫 10 시간 단계를 받아 11번째 시간 단계를 예측하는 망을 학습시킵니다:
model.train_on_batch(x[:, :10, :], np.reshape(x[:, 10, :], (32, 16)))

# 망의 상태가 변했습니다. 후속 시퀀스를 전달할 수 있습니다:
model.train_on_batch(x[:, 10:20, :], np.reshape(x[:, 20, :], (32, 16)))

# 장단기 메모리 레이어의 상태를 재설정합시다:
model.reset_states()

# 이 경우 다음과 같이 하는 방법도 있습니다:
model.layers[0].reset_states()
```

`predict`, `fit`, `train_on_batch`, `predict_classes` 등의 메서드는 *전부* 모델의 상태형 레이어의 상태를 최신화한다는 점을 참고하십시오. 이는 상태형 학습뿐 아니라 상태형 예측까지 가능케 합니다.

---

### Sequential 모델에서 레이어를 없애려면 어떻게 하나요?

`.pop()`을 호출해서 Sequential 모델에 마지막으로 추가된 레이어를 제거할 수 있습니다:

```python
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(32, activation='relu'))

print(len(model.layers))  # "2"

model.pop()
print(len(model.layers))  # "1"
```

---

### 케라스에서 선행 학습된 모델을 사용하려면 어떻게 하나요?

다음의 이미지 분류 모델의 코드와 선행 학습된 가중치를 사용할 수 있습니다:

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

이러한 모델은 `keras.applications` 모듈에서 임포트됩니다:

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

몇 가지 간단한 예시를 보려면 [Applications 모듈 설명서](/applications)를 참고하십시오.

선행 학습된 모델을 특성 추출이나 미세-조정 용도로 사용하는 자세한 예시를 확인하려면, [이 블로그 포스트](http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)를 참고하십시오.

또한 VGG16 모델을 바탕으로 한 여러가지 케라스 예시 스크립트도 있습니다:

- [스타일 이전](https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py)
- [특성 시각화](https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py)
- [딥 드림](https://github.com/keras-team/keras/blob/master/examples/deep_dream.py)

---

### 케라스에서 HDF5 인풋은 어떻게 사용하나요?

`keras.utils`에서 `HDF5Matrix` 클래스를 사용할 수 있습니다. 세부 사항은 [HDF5Matrix 설명서](/utils/#hdf5matrix)를 참고하십시오.

HDF5 데이터셋을 직접 사용할 수도 있습니다:

```python
import h5py
with h5py.File('input/file.hdf5', 'r') as f:
    x_data = f['x_data']
    model.predict(x_data)
```

`h5py`의 설치 방법에 대한 설명은 [HDF5나 h5py를 설치해 케라스 모델을 저장하려면 어떻게 하나요?](#how-can-i-install-hdf5-or-h5py-to-save-my-models-in-keras)를 참조해 주시기 바랍니다.

---

### 케라스 구성 파일은 어디에 저장되어 있나요?

모든 케라스 데이터가 저장되는 디폴트 파일 경로는 다음과 같습니다:

```bash
$HOME/.keras/
```

윈도우의 경우 `$HOME`을 `%USERPROFILE%`로 대체해야 합니다.
케라스가 위의 파일 경로를 만들 수 없는 경우 (예. 권한 문제), `/tmp/.keras/`이 예비로 사용됩니다.

케라스 구성 파일은 `$HOME/.keras/keras.json`에 저장된 JSON 파일입니다. 디폴트 구성 파일은 다음과 같습니다:

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

이는 다음의 여러 영역을 포함합니다:

- 이미지 프로세싱 레이어와 유틸리티가 디폴트로 사용하는 이미지 데이터 형식 (`channels_last` 혹은 `channels_first`).
- 몇몇 연산에서 0으로 나누는 경우를 방지하기 위해 사용되는 매우 작은 `epsilon` 인자.
- 디폴트 부동소수점 자료형.
- 디폴트 백엔드. [백엔드 설명서](/backend)를 참고하십시오.

마찬가지로, [`get_file()`](/utils/#get_file)로 다운로드한 파일과 같은 캐싱된 데이터셋 파일은 디폴트로 `$HOME/.keras/datasets/`에 저장됩니다.

---

### 개발 중 케라스를 사용해 재현 가능한 결과를 얻으려면 어떻게 하나요?

모델 개발 과정에서 성능의 변화가 모델에 기인하는지, 데이터 수정 때문인지, 혹은 새로운 임의의 샘플 덕인지 구별하려면, 여러 실행에 걸쳐 재현 가능한 결과를 얻는 것이 중요합니다.

우선 (프로그램 자체의 내부가 아닌) 프로그램이 시작하기 전에 `PYTHONHASHSEED` 환경 변수를 `0`으로 설정해야 합니다. 이는 파이썬 3.2.3 이상에서 특정 해시-기반 연산에 대해 재현 가능한 행동 방식을 끌어내기 위해서 필요한 과정입니다 (예. 세트나 딕셔너리 내 아이템 순서, 보다 자세한 사항은 [파이썬 설명서](https://docs.python.org/3.7/using/cmdline.html#envvar-PYTHONHASHSEED)나 [이슈 #2280](https://github.com/keras-team/keras/issues/2280#issuecomment-306959926)를 참고하십시오). 다음처럼 파이썬을 시작할 때 환경변수를 설정할 수 있습니다:

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

게다가 텐서플로우 백엔드로 GPU에서 실행하는 경우, 몇몇 연산(특히 `tf.reduce_sum()`)은 불확정적 아웃풋을 반환합니다. 이는 GPU가 수 많은 연산을 병렬 처리하여 실행 순서가 확실히 보장되지 않기 때문입니다. 또한 부동소수점의 제한적인 정확성 때문에, 숫자 몇 개를 더할 때도 더하는 순서에 따라 그 결과가 미세하게 달라질 수 있습니다. 불확정적 연산을 피해볼 수는 있겠지만, 어느 정도는 텐서플로우가 경사를 계산하는 과정에서 자동적으로 발생하기 때문에, 그냥 CPU에 코드를 실행하는 편이 훨씬 간편합니다. `CUDA_VISIBLE_DEVICES` 환경 변수를 빈 문자열로 설정하면 됩니다. 다음은 예시입니다:

```
$ CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python your_program.py
```

아래의 간단한 코드는 재현 가능한 결과를 어떻게 얻을 수 있는지에 대한 예시를 제공합니다 –텐서플로우 백엔드와 파이썬 3 환경을 가정했습니다.

```python
import numpy as np
import tensorflow as tf
import random as rn

# 아래는명확히 정의된 초기 상태에서 Numpy로 난수를 생성하기 위해
# 필요한 과정입니다.

np.random.seed(42)

# 아래는 명확히 정의된 초기 상태에서 파이썬 코어로 난수를 생성하기 위해
# 필요한 과정입니다.

rn.seed(12345)

# 텐서플로우가 하나의 스레드만 사용하도록 강제합니다
# 다중 스레드는 재현 불가능한 결과의 잠재적 원인입니다
# 보다 자세한 사항은 https://stackoverflow.com/questions/42022950/을 참고하십시오

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K

# 텐서플로우 백엔드의 경우, 아래의 tf.set_random_seed()를 통해
# 명확히 정의된 초기 상태에서 난수 생성을 할 수 있습니다.
# 보다 자세한 사항은
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed를 참고하십시오

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# 나머지 코드...
```

---

### HDF5나 h5py를 설치해 케라스 모델을 저장하려면 어떻게 하나요?

케라스 모델을, 예를 들어 `keras.callbacks.ModelCheckpoint`를 통해
HDF5 파일로 저장하려면, h5py 파이썬 패키지가 필요합니다.
이 패키지는 케라스의 의존 요소로, 처음에 디폴트로 설치됩니다.
Debian-기반의 배포판에서는 추가적으로 `libhdf5`를 설치해야 합니다:

```
sudo apt-get install libhdf5-serial-dev
```

h5py가 설치되어 있는지 모르겠다면 파이썬 쉘을 열어 다음을 통해
모듈을 불러올 수 있습니다:

```
import h5py
```

만약 에러없이 모듈을 가져오면, h5py가 설치된 것이고, 아니라면 다음에서
자세한 설치 안내를 확인할 수 있습니다: http://docs.h5py.org/en/latest/build.html
