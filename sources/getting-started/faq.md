# Keras FAQ: 자주 문의되는 Keras관련 질문

- [Keras를 어떻게 인용해야 하는가?](#how-should-i-cite-keras)
- [Keras를 어떻게 GPU에서 실행할 수 있는가?](#how-can-i-run-keras-on-gpu)
- [Keras 모델을 어떻게 다중 GPU에서 실행할 수 있는가?](#how-can-i-run-a-keras-model-on-multiple-gpus)
- ["sample", "batch", "epoch"이 의미하는게 무엇인가?](#what-does-sample-batch-epoch-mean)
- [Keras 모델을 어떻게 저장할 수 있는가?](#how-can-i-save-a-keras-model)
- [왜 학습 손실이 테스트 손실보다 더 큰가?](#why-is-the-training-loss-much-higher-than-the-testing-loss)
- [어떻게 중간 계층의 출력을 얻을 수 있는가?](#how-can-i-obtain-the-output-of-an-intermediate-layer)
- [메모리 허용치 보다 큰 데이터셋을 Keras에서 어떻게 사용하 수 있는가?](#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory)
- [검증 손실이 더 이상 감소하지 않을때, 어떻게 학습을 중단할 수 있는가?](#how-can-i-interrupt-training-when-the-validation-loss-isnt-decreasing-anymore)
- [검증의 나뉨이 어떻게 계산 되는가?](#how-is-the-validation-split-computed)
- [학습 도중에 데이터가 뒤섞이는가?](#is-the-data-shuffled-during-training)
- [어떻게 매 epoch마다 학습 / 검증 손실 / 정확도를 기록할 수 있는가?](#how-can-i-record-the-training--validation-loss--accuracy-at-each-epoch)
- [어떻게 계층을 "freeze"할 수 있는가?](#how-can-i-freeze-keras-layers)
- [어떻게 상태 기반(stateful) RNNs를 사용할 수 있는가?](#how-can-i-use-stateful-rnns)
- [순차적 모델로 부터 어떻게 계층을 삭제할 수 있는가?](#how-can-i-remove-a-layer-from-a-sequential-model)
- [Keras에서 어떻게 미리 학습된 모델을 사용할 수 있는가?](#how-can-i-use-pre-trained-models-in-keras)
- [HDF5를 어떻게 Keras에 입력으로 사용할 수 있는가?](#how-can-i-use-hdf5-inputs-with-keras)
- [Keras 설정 파일은 어디에 저장되어 있는가?](#where-is-the-keras-configuration-file-stored)
- [개발 도중 Keras를 사용해서 어떻게 재생산 가능한 결과를 얻을 수 있는가?](#how-can-i-obtain-reproducible-results-using-keras-during-development)
- [Keras에서 모델을 저장하기 위한 HDF5 또는 h5py를 어떻게 설치할 수 있는가?](#how-can-i-install-hdf5-or-h5py-to-save-my-models-in-keras)

---

### Keras를 어떻게 인용해야 하는가?

연구에 도움이 되었다면, 출판물에 Keras를 인용해 주십시오. 다음은 BibTex 기입 방법의 예 입니다:

```
@misc{chollet2015keras,
  title={Keras},
  author={Chollet, Fran\c{c}ois and others},
  year={2015},
  howpublished={\url{https://keras.io}},
}
```

---

### Keras를 어떻게 GPU에서 실행할 수 있는가?

**Tensorflow** 또는 **CNTK**를 백엔드로서 사용하는 경우, 작성하신 코드는 가용한 GPU가 감지된다면 자동으로 GPU에서 실행될 것입니다.

**Theano**를 백엔드로서 사용 한다면, 다음의 방법 중 하나를 사용하십시오:

**방법 1**: Theano 플래그를 사용하십시오.
```bash
THEANO_FLAGS=device=gpu,floatX=float32 python my_keras_script.py
```

'gpu' 이름은 사용하시는 장치의 식별자에 따라서 변경해야만 합니다.

**방법 2**: `.theanorc`를 설정 하십시오: [설명](http://deeplearning.net/software/theano/library/config.html)

**방법 3**: 수동으로 `theano.config.device`와 `theano.config.floatX`를 코드 시작 부분에서 설정 하십시오:
```python
import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'
```

---

### Keras 모델을 어떻게 다중 GPU에서 실행할 수 있는가?

**TensorFlow**를 백엔드로서 사용하기를 권고 드립니다. 단일 모델을 다중 GPU에서 실행하기 위한 두 가지 방법이 있습니다: **data parallelism** 그리고 **device parallelism**가 있습니다.

대부분의 경우에서, 데이터의 병렬처리가 가장 필요한 것입니다.

#### 데이터의 병렬처리

데이터의 병렬처리는 각각의 기기(GPU)에 한번 대상 모델을 복사하는 것, 그리고 각 복사된 모델로 하여금 서로 다른 입력 데이터의 부분을 처리하는 것으로 구성됩니다.
Keras는 빌트인된 `keras.utils.multi_gpu_model` 유틸리티를 가지고 있는데, 어떤 모델이든 데이터-병렬처리 버전을 만들어낼 수 있고 최대 8개의 GPU에서 선형적으로 속도를 증가시킬 수 있습니다.

더 많은 정보를 위해서 [다중 GPU 모델](/utils/#multi_gpu_model) 문서를 참고 하십시오. 다음은 간단한 예 입니다:

```python
from keras.utils import multi_gpu_model

# 8개의 GPU에 `model`을 복제 합니다.
# 사용중인 기계에 8개의 가용 GPU가 있다는 것을 가정 합니다.
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# 다음의 `fit`을 호출은 8개의 GPU에 분산되어 집니다.
# 배치 크기가 256 이기 때문에, 각 GPU는 32개의 샘플을 작업하게 됩니다.
parallel_model.fit(x, y, epochs=20, batch_size=256)
```

#### 기기의 병렬처리

기기의 병렬처리는 서로다른 기기에서 동일한 모델의 서로다른 부분을 실행하는 것으로 구성됩니다. 예를 들어서, 두개의 가지(브랜치)를 가지는 모델과 같이 병렬적인 구조를 가지는 모델에서 최상의 동작을 합니다.

TensorFlow의 디바이스 범위(device scopes)를 사용해서 수행될 수 있습니다. 다음은 간단한 예 입니다:

```python
# 공유된 LSTM이 서로 다른 시퀀스를 병렬로 인코딩하는 모델 입니다
input_a = keras.Input(shape=(140, 256))
input_b = keras.Input(shape=(140, 256))

shared_lstm = keras.layers.LSTM(64)

# 첫 번째 시퀀스를 하나의 GPU에서 수행 합니다
with tf.device_scope('/gpu:0'):
    encoded_a = shared_lstm(tweet_a)
# 그 다음 시퀀스를 다른 하나의 GPU에서 수행 합니다
with tf.device_scope('/gpu:1'):
    encoded_b = shared_lstm(tweet_b)

# CPU에서 각 결과를 이어 붙입니다
with tf.device_scope('/cpu:0'):
    merged_vector = keras.layers.concatenate([encoded_a, encoded_b],
                                             axis=-1)
```

---

### "sample", "batch", "epoch"이 의미하는게 무엇인가?

아래의 내용은 Keras를 올바르게 활용하기 위해서 이해할 필요가 있는 몇 공통적인 정의 입니다.

- **Sample**: 데이터셋의 하나의 요소 입니다.
  - *예:* 하나의 이미지는 합성곱 신경망에서의 하나의 **sample** 입니다.
  - *예:* 하나의 오디오 파일은 음성 인식 모델에서의 하나의 **sample** 입니다.
- **Batch**: *N* 개의 sample 집합 입니다. 하나의 **batch**에 있는 sample들은 독립적으로 병렬 처리 됩니다. 만약 학습 중이라면, 하나의 batch가 모델을 한번 업데이트 하는 결과를 가져 옵니다.
  - 하나의 **batch**는 일반적으로 하나의 입력보다 입력 데이터의 분배에 대해서 더 좋은 근사치를 냅니다. batch가 크면 클 수록, 근사치가 더 좋아집니다; 하지만, 더 큰 batch는 수행하는데 더 많은 시간이 걸리고, 여전히 한번의 업데이트만의 결과를 거져옵니다. 추론 단계에서 (평가/예측), (일반적으로 큰 batch가 더 빠른 평가/예측의 결과를 낼 수 있기 때문에) 메모리 크기를 초과하지 않는 범위의 수용할 수 있는한 가장 큰 batch 크기를 선택하는 것이 권고 됩니다.
- **Epoch**: 무작위의 절단으로, 일반적으론 "전체 데이터셋에 대한 한번의 통과(수행)"으로 정의 됩니다. 로깅과 주기적인 평가에 유용한 학습을 별개의 단계로 분리하는 것을 위해서 사용됩니다.
  - `validation_data`또는 `validation_split`를 Keras 모델의 `fit`메소드와 함께 사용할 때, 평가는 매 **epoch**의 마지막에 수행 됩니다. 
  - Keras에서는 **epoch**의 마지막에 수행되도록 특별히 디자인된 [콜백](https://keras.io/callbacks/)을 추가할 수 있는 방법이 있습니다. 이에 대한 예로는 학습률의 변경과 모델의 체크포인트(저장)을 위한것이 있습니다.

---

### Keras 모델을 어떻게 저장할 수 있는가?

#### 전체 모델(구조 + 가중치 + optimizer 상태)을 저장/불러오기

*pickle 또는 cPickel을 사용해서 Keras 모델을 저장하는것은 권고되지 않습니다.*

`model.save(filepath)`을 사용해서 Keras 모델을 다음을 포함하는 단일 HDF5 파일로 저장할 수 있습니다:

- 모델을 재 생성하기 위한 모델의 아키텍처(구조).
- 모델의 가중치.
- 학습 설정 (손실, optimizer).
- 직전 학습에 이어서 학습하기 위한 optimizer의 상태.

그러면, `keras.models.load_model(filepath)`를 사용해서 모델을 다시 인스턴스화 할 수 있습니다.
또한 `load_model`은 (전에 컴파일이 한번도 이루어지지 않은 경우가 아니라면)저장된 학습 설정을 사용해서 모델의 컴파일을 처리 합니다.

ㅇ{:

```python
from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# 컴파일된 모델을 반환 합니다.
# 직전의 것과 완전히 동일 합니다.
model = load_model('my_model.h5')
```

[Keras에서 모델을 저장하기 위한 HDF5 또는 h5py를 어떻게 설치할 수 있는가?](#how-can-i-install-hdf5-or-h5py-to-save-my-models-in-keras) 에서 `h5py`를 설치하기 위한 설명을 참고 하십시오.

#### 모델의 구조만을 저장/불러오기

가중치 또는 학습 설정이 아니라 **모델의 구조**만을 저장해야 할 필요가 있다면, 다음을 해볼 수 있습니다:

```python
# JSON으로 저장 합니다.
json_string = model.to_json()

# YAML으로 저장 합니다.
yaml_string = model.to_yaml()
```

생성된 JSON / YAML 파일은 사람이 읽을 수 있고, 필요에 의해서 수작업으로 수정될 수 있습니다.

이 데이터로부터 새로운 모델을 만들 수 있습니다:

```python
# JSON으로 부터 모델을 복원 합니다:
from keras.models import model_from_json
model = model_from_json(json_string)

# YAML으로 부터 모델을 복원 합니다:
from keras.models import model_from_yaml
model = model_from_yaml(yaml_string)
```

#### 모델의 가중치만을 저장/불러오기

**모델의 가중치**를 저장해야할 필요가 있다면, 아래 코드에서 처럼 HDF5를 사용해서 수행할 수 있습니다:

```python
model.save_weights('my_model_weights.h5')
```

여러분의 모델을 인스턴스화 하기위한 코드가 있다고 가정해 봅시다. 그러면, *동일한* 구조를 가지는 모델로 저장된 가중치를 불러올 수 있습니다:

```python
model.load_weights('my_model_weights.h5')
```

미세 조정이나 전이 학습과 같이 (몇 계층들은 공통적으로 가지지만) *다른* 구조의 모델로 가중치를 불러올 필요가 있을 때, *계층 이름* 을 통해서 불러올 수 있습니다:

```python
model.load_weights('my_model_weights.h5', by_name=True)
```

예:

```python
"""
원본 모델이 다음과 같은 형태라고 가정해 봅시다:
    model = Sequential()
    model.add(Dense(2, input_dim=3, name='dense_1'))
    model.add(Dense(3, name='dense_2'))
    ...
    model.save_weights(fname)
"""

# 새로운 모델 입니다.
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # 불러와 지게 됩니다.
model.add(Dense(10, name='new_dense'))  # 불러와 지지지 않게 됩니다.

# 첫 번째 모델로 부터 가중치를 불러 옵니다; 첫 번째 계층인 dense_1에만 영향을 미칩니다.
model.load_weights(fname, by_name=True)
```

`h5py`를 설치는 방법에 대한 설명으로 [Keras에서 모델을 저장하기 위한 HDF5 또는 h5py를 어떻게 설치할 수 있는가?](#how-can-i-install-hdf5-or-h5py-to-save-my-models-in-keras)를 참조 하십시오.

#### 저장된 모델내의 사용자 정의 계층(또는 사용자 정의 객체)을 다루는 것

불러오고 싶은 모델이 사용자 정의 계층 또는 다른 사용자 정의 클래스나 함수를 포함할 때, 
`custom_objects`인자를 통한 불러오기 메커니즘에 그 대상을 전달해 줄 수 있습니다:

```python
from keras.models import load_model
# 모델이 "AttentionLayer" 클래스의 인스턴스를 포함한다고 가정해 봅시다.
model = load_model('my_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
```

다른 방법으로는, [사용자 정의 객체 스코프](https://keras.io/utils/#customobjectscope)를 사용할 수 있습니다:

```python
from keras.utils import CustomObjectScope

with CustomObjectScope({'AttentionLayer': AttentionLayer}):
    model = load_model('my_model.h5')
```

사용자 정의 객체를 다루는 것은 `load_model`, `model_from_json`, `model_from_yaml`와 동일한 방법으로 동작합니다:

```python
from keras.models import model_from_json
model = model_from_json(json_string, custom_objects={'AttentionLayer': AttentionLayer})
```

---

### 왜 학습 손실이 테스트 손실보다 더 큰가?

Keras 모델은 두 가지 모드를 가집니다: 학습과 테스트가 그것입니다. Dropout 그리고 L1/L2 가중치 Regularization과 같은 Regularization 기법들은 테스트시에는 사용되지 않습니다.

이와 더불어, 학습 손실은 학습 데이터의 각 batch 마다의 손실들의 평균입니다. 여러분의 모델이 시간이 경과함에 따라 바뀌기 때문에, 한 epoch의 첫 번째 batchㅇ 대한 손실은 일반적으로 마지막 batch의 것보다 높습니다. 반면에 각 epoch에 대한 테스트 손실은 epoch의 마지막에 도달한 모델을 사용해 계산되어 낮은 손실으 결과르 보여줍니다.

---

### 어떻게 중간 계층의 출력을 얻을 수 있는가?

한가지 간단한 방법은 여러분이 관심을 가지는 계층들을 출력하는 새로운 `Model`을 생성하는 것입니다.

```python
from keras.models import Model

model = ...  # 원본 모델을 생서 합니다.

layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
```

다른 방법으로는, 주어진 특정 입력에 대한 특정 계층의 결과를 반환하는 Keras 함수를 만들 수 있습니다. 예를 들어서:

```python
from keras import backend as K

# 순차적(Sequential) 모델
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([x])[0]
```

이와 유사하게, Theano와 TensorFlow 함수를 직접적으로 만들 수도 있습니다.

만약 여러분의 모델이 (예를 들어서, `Dropout`, `BatchNormalization`, 등과 같은 것을 사용하는 경우) 만약 학습과 테스트 시기에 서로다른 행동을 보여준다면, 함수에 학습 시기에 대한 플래그값을 넘겨줘야만 할 것입니다:

```python
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])

# 테스트 모드 = 0 에 대한 출력
layer_output = get_3rd_layer_output([x, 0])[0]

# 학습 모드 = 1 에 대하 출력
layer_output = get_3rd_layer_output([x, 1])[0]
```

---

### 메모리 허용치 보다 큰 데이터셋을 Keras에서 어떻게 사용하 수 있는가?

`model.train_on_batch(x, y)`그리고 `model.test_on_batch(x, y)`를 사용해서 batch 학습을 수행할 수 있습니다. [모델 문서](/models/sequential) 를 참조 하십시오.

다른 방법으로, 학습 데이터의 batch를 수확(yeild)하는 생성자(generator)를 만들고, `model.fit_generator(data_generator, steps_per_epoch, epochs)`메소드를 사용 하실 수 있습니다.

[CIFAR10 예제](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py)에서의 batch 학습이 어떻게 동작하는지를 확인해 보실 수 있습니다.

---

### 검증 손실이 더 이상 감소핮 않을때, 어떻게 학습을 중단할 수 있는가?

`EarlyStopping`콜백을 사용할 수 있습니다:

```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
```

[콜백 문서](/callbacks)에서 좀 더 자세한 내용을 확인하십시오.

---

### 검증의 나뉨이 어떻게 계산 되는가?
 
`model.fit`에 있는 `validation_split` 인자를 예를 들어 0.1로 설정한 경우, 사용되는 검증 데이터는 데이터의 *마지막 10%* 가 됩니다. 만약 그 값을 0.25로 설정 한다면, 데이터의 마지막 25%가 될 것입니다. 데이터는 검증의 나뉨을 추출하기 전엔 뒤섞잊 않아서 검증이란 말 그대로 전달받은 입력 sample들의 *마지막* x%가 됨을 알아 두십시오.

(동일한 `fit`호출 안에서)모든 epoch에 대해서 동일한 검증 데이터가 사용 됩니다. 

---

### 학습 도중에 데이터가 뒤섞이는가?

네, `model.fit`에 있는 `suffle`인자가 (디폴트)`True`로 설정 되었다면, 학습 데이터는 매 epoch마 무작위로 뒤섞이게 됩니다.

검증 데이터는 뒤섞이지 않습니다.

---


### 어떻게 매 epoch마다 학습 / 검증 손실 / 정확도를 기록할 수 있는가?

`model.fit`메소드는 `History`라는 콜백을 반환합니다. `History`콜백은 연속적인 손실값과 다른 메트릭스에 대한 리스트를 포함하는 `history` 속성을 가집니다.

```python
hist = model.fit(x, y, validation_split=0.2)
print(hist.history)
```

---

### 어떻게 계층을 "freeze"할 수 있는가?

계층을 "freeze" 한다는 것은 학습의 대상에서 제외시킴을 의미 합니다. 예를 들어서, 그 계층의 가중치는 업데이트 되지 않습니다. 모델을 미세조정하거나 텍스트 입력에 대해서 고정된 임베딩을 사용해야하는 경우에 유용합니다.

계층의 생성자의 `trainable`인자에 학습가능하지 않은 계층의 집합을 넘겨줄 수 있습니다.

```python
frozen_layer = Dense(32, trainable=False)
```

추가적으로, 계층을 인스턴스호 한 후에 `trainable` 속성을 `True`또는 `False`로 설정할 수 있습니다. 설정 내용이 반영되려면, `trainable`속성을 변경한 후에 모델에 대하여 `compile()`을 호출해줘야 합니다. 다음은 그에대하 예 입니다:

```python
x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)

frozen_model = Model(x, y)
# 아래의 모델에서, `layer`에 대하 가중치는 학습도중 업데이트 되지 않습니다.
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = True
trainable_model = Model(x, y)
# 이번의 모델에서는 layer의 가중치가 학습도중 업데이트 됩니다.
# (동일한 계층 인스턴스를 사용하기 때문에, 그 결과는 위의 모델에도 영향을 미칩니다.)
trainable_model.compile(optimizer='rmsprop', loss='mse')

frozen_model.fit(data, labels)  # this does NOT update the weights of `layer`
trainable_model.fit(data, labels)  # this updates the weights of `layer`
```

---

### 어떻게 상태 기반(stateful) RNNs를 사용할 수 있는가?

RNN을 상태기반으로 만든다는 것은 각 batch의 샘플에 대한 상태가 다음 batch의 샘플에 대한 초기 상태로서 재사용된다는 의미를 가집니다.

따라서, 상태 기반 RNNs을 사용할 때, 다음과 같은것이 가정됩니다:

- 모든 batch는 동일한 수의 샘플을 가집니다.
- 만야 `x1`와 `x2`가 연속적인 샘플에 대한 batch라면, 모든 i에 대해서 `x2[i]`가 `x1[i]`의 다음 순서가 됩니다.

RNNs에서 상태 정보를 사용하려면, 다음을 수행해야 합니다:

- 모델의 첫 번째 계층에 `batch_size`인자에 사용하고자 하는 batch 크기를 명시적으로 전달해 줘야 합니다. 예를 들어서, `batch_size=32`는 32개의 샘플로 구성된 batch의 순서를 16개의 특징(features)을 가지는 10번의 시간 단계로 구성합니다.
- RNN 계층에 대하여 `stateful=True`을 설정해 주십시오.
- `fit()`을 호출할 때, `suffle=False`을 설정해 주십시오.

누적된 상태를 리셋하기 위해서는:

- `model.reset_states()`을 사용해서 모델의 모든 계층에 대한 상태를 리셋 하십시오.
- use `layer.reset_states()`을 사용해서 상태 기반의 특정 RNN 계층의 상태를 리셋 하십시오.

Example:

```python
x  # 입력 데이터로, 그 shape은 (32, 12, 16) 입니다.
# 길이 10로 구성된 순서들을 모델에 주입하게 됩니다.

model = Sequential()
model.add(LSTM(32, input_shape=(10, 16), batch_size=32, stateful=True))
model.add(Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 주어진 처음 10 시간 순서로부터, 11번째 시간 순서를 예측할 수 있도록 네트워크를 학습시킵니다.
model.train_on_batch(x[:, :10, :], np.reshape(x[:, 10, :], (32, 16)))

# 네트워크의 상태가 변화 했고, 그 다음의 시간 순서를 주입할 수 있습니다.
model.train_on_batch(x[:, 10:20, :], np.reshape(x[:, 20, :], (32, 16)))

# LSTM 계층의 상태를 리셋 합니다:
model.reset_states()

# 다음과 같이 리셋을위한 다른 방법도 있습니다:
model.layers[0].reset_states()
```

`predict`, `fit`, `train_on_batch`, `predict_classes`, 등의 모든 메소드는 모델내의 상태기반 계층의 상태를 업데이트 합니다. 상태 기반 학습을 할 수 있게 해줄 뿐만 아니라, 상태기반 예측도 가능하게 해 줍니다.

---

### 순차적 모델로 부터 어떻게 계층을 삭제할 수 있는가?

`.pop()`을 호출하여 순차적 모델에 마지막에 추가된 계층을 삭제하 수 있습니다:

```python
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(32, activation='relu'))

print(len(model.layers))  # "2"

model.pop()
print(len(model.layers))  # "1"
```

---

### Keras에서 어떻게 미리 학습된 모델을 사용할 수 있는가?

다음의 이미지 분류 모델에 대한 코드와 미리 학습된 가중치를 활용할 수 있습니다:

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

 `keras.applications`모듈로 부터 불러와져서 사용될 수 있습니다:

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

몇 간단한 사용법 예는 [Applications 모듈에 대하 문서](/applications)를 확인하시기 바랍니다.

미세 조정이나 특징 추출을 위해서 미리 학습된 모델을 사용하는 방법에 대한 자세한 내용은 [여기 블로그 포스트](http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)를 확인하시기 바랍니다.

VGG16 모델은 몇 Keras의 예제 스크립트의 기본으로 사용되고 있습니다:

- [Style transfer](https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py)
- [Feature visualization](https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py)
- [Deep dream](https://github.com/keras-team/keras/blob/master/examples/deep_dream.py)

---

### HDF5를 어떻게 Keras에 입력으로 사용할 수 있는가?

`keras.utils`에 있는 `HDF5Matrix`클래스를 사용할 수 있습니다. 자세한 내용은 [the HDF5Matrix documentation](/utils/#hdf5matrix)를 참고하시기 바랍니다.

HDF5 데이터셋을 직접적으로 사용할 수 있습니다:

```python
import h5py
with h5py.File('input/file.hdf5', 'r') as f:
    x_data = f['x_data']
    model.predict(x_data)
```

[Keras에서 모델을 저장하기 위한 HDF5 또는 h5py를 어떻게 설치할 수 있는가?](#how-can-i-install-hdf5-or-h5py-to-save-my-models-in-keras) 에서 `h5py`를 설치하기 위한 설명을 참고 하십시오.

---

### Keras 설정 파일은 어디에 저장되어 있는가?

Keras의 데이터가 저장되는 디폴트 디렉토리의 위치는 다음과 같습니다:

```bash
$HOME/.keras/
```

Windows 사용자는 `$HOME`을 `%USERPROFILE%`로 교체하십시오.
(권한 문제등으로 인해서) Keras가 위 디렉토리를 생성할 수 없는 경우라면, `/tmp/.keras`가 백업으로 사용 됩니다.

Keras 설정 파일은 JSON 파일로, `$HOME/.keras/keras.json`에 저장되어 있습니다. 디폴트 설정 파일은 다음과 같은 형태를 가집니다:

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

다음과 같은 필드 요소를 포함합니다:

- (`channels_last`또는 `channels_first`) 유틸리티와 이미지 철 계층에서 디폴트로 사용될 이미지 데이터의 포맷을 나타냅니다.
- `epsilon`는 숫자의 fuzz 요소로, 몇 연산자에서 0으로 나눠지는 것을 방지하기 위해서 사용 됩니다.
- 디폴트 자료형을 부동소수(float)로 명시 합니다.
- 디폴트 백엔드를 명시합니다. [백엔드 문서](/backend)를 참고 하십시오.

이와 비슷하게, [`get_file()`](/utils/#get_file)같은것에 의해 다운로드된 캐쉬된 데이터셋의 파일은 디폴트로 `$HOME/.keras/datasets/`에 저장됩니다.

---

### [개발 도중 Keras를 사용해서 어떻게 재생산 가능한 결과를 얻을 수 있는가?]

모델을 개발하는 도중에 성능에서의 어떠 변화가 실제 모델의 의한 것인지, 데이터 변형에 의한 것인지, 또는 새로운 무작위 샘플들에 의한 결과 때문인지를 판단하 위해서 재생산이 가능한 결과를 실행할 때 마다 얻는것이 때로는 유용할 수 있습니다. 

우선, `PYTHONHASHSEED`환경 변수를 프로그램이 시작하기 전 `0`으로 설정하십시오. Python 3.2.3 이후의 버전에서는 (set이나 dict 자료구조의 저장 순서와 같은, 더 깊은 이해를 위해서는 [Python 문서](https://docs.python.org/3.7/using/cmdline.html#envvar-PYTHONHASHSEED) 또는 [이슈 #2280](https://github.com/keras-team/keras/issues/2280#issuecomment-306959926)를 확인하십시오)해쉬 기반의 연산의 행동을 재생산 하기 위한  필수 과정 입니다. 환경 변수를 설정하기 위한 한 가지 방법은 다음과 같이 Python이 실행되 때 해주는 것입니다:

```
$ cat test_hash.py
print(hash("keras"))
$ python3 test_hash.py                  # 재생산이 불가능한 해쉬 (Python 3.2.3+)
-8127205062320133199
$ python3 test_hash.py                  # 재생산이 불가능한 해쉬 (Python 3.2.3+)
3204480642156461591
$ PYTHONHASHSEED=0 python3 test_hash.py # 재생산이 가능한 해쉬
4883664951434749476
$ PYTHONHASHSEED=0 python3 test_hash.py # 재생산이 가능한 해쉬
4883664951434749476
```

더욱이 GPU에서 구동하는 TensorFlow를 백엔드 사용한다면, `tf.reduce_sum()`와 같은 몇 연산들은 비결정적(non-deterministic)인 출력을 만들어 냅니다. 많은 연산들이 병렬로 GPU에서 수행되기 때문이며, 그렇ㄱ 때문에 실행의 순서가 항상 보장되 않습니다. 부동소수에 대한 제한된 정확성 때문에, 몇 숫자들을 더하는것 조차 더해지는 순서에 따라서 약간 다른 결과를 보여줄 수 있습니다. 여러분 스스로가 비결정적 연산을 피해보려고 노력할 수는 있지만, 경사도를 계산하기 위해서 TensorFlow가 자동으로 생성하는 부분이 있기 때문에, CPU에서 코드를 실행해 보는 것이 훨씬 더 간단합니다. 그러 위해서, 다음과 같이 `CUDA_VISIBLE_DEVICES`환경 변수에 빈 문자열을 설정해 줄 수 있습니다:

```
$ CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python your_program.py
```

아래의 코드 스니펫은 재생산이 가능하 결과를 얻어오기 위한 예를 보여줍니다 - 사용된 환경은 Python3와 TensorFlow를 백엔드 사용하고 있습니다:

```python
import numpy as np
import tensorflow as tf
import random as rn

# 아래는 Numpy가 무작위 숫자의 발생을 
# 잘 정의된 초기 상태에 따라서 시작하기 위해서 필수입니다.

np.random.seed(42)

# 아래는 Python 코어가 무작위 숫자의 발생을
# 잘 정의된 초기 상태에 따라서 시작하 위해서 필수입니다.

rn.seed(12345)

# TensorFlow가 단을 쓰레드를 사용하도록 강제 합니다.
# 다중 쓰레드는 재생산할 수 없는 결과를 만들어내는 잠재적 요소 입니다.
# 더 자세한 내용은 https://stackoverflow.com/questions/42022950/를 확인하십시오.

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K

# 아래의 tf.set_random_seed()는 TensorFlow를 백엔드로 하는 
# 잘 정의된 초기 상태에 따른 무작위 숫자 발생을 만들어 냅니다.
# 더 자세한 내용은 다음을 확인하십시오:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# 나머 코드가 기술됩니다 ...
```

---

### Keras에서 모델을 저장하기 위한 HDF5 또는 h5py를 어떻게 설치할 수 있는가?

`keras.callbacks.ModelCheckpoint`와 같은 것을 사용하여 Keras 모델을 HDF5 파일로서 저장학 위해서,
Keras는 h5py라는 Python 패키지를 사용합니다. Keras의 의존성(dependency)로, 디폴트 설치가 되어 합니다.
데비안 기반의 환경에서는 `libhdf5`가 추가적으로 설치 되어야만 합니다:

```
sudo apt-get install libhdf5-serial-dev
```

h5py가 설치되었는지 잘 모르겠다면, Python 쉘을 열어 아래처럼 모듈을 불러와볼 수 있습니다:

```
import h5py
```

import 수행시 에러가 없다면, 설치가 된 것입니다. 그렇지 않다면, 더 상세한 
설치에 대한 설명을 여기서:http://docs.h5py.org/en/latest/build.html 참고하십시오.
