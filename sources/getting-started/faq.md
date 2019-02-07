# Keras FAQ: 자주 문의되는 Keras관련 질문

- [Keras를 어떻게 인용해야 하는가?](#how-should-i-cite-keras)
- [Keras를 어떻게 GPU에서 실행할 수 있는가?](#how-can-i-run-keras-on-gpu)
- [Keras 모델을 어떻게 다중 GPU에서 실행할 수 있는가?](#how-can-i-run-a-keras-model-on-multiple-gpus)
- ["sample", "batch", "epoch"이 의미하는게 무엇인가?](#what-does-sample-batch-epoch-mean)
- [Keras 모델을 어떻게 저장할 수 있는가?](#how-can-i-save-a-keras-model)
- [왜 학습 손실이 테스트 손실보다 더 큰가?](#why-is-the-training-loss-much-higher-than-the-testing-loss)
- [어떻게 중간 계층의 출력을 얻을 수 있는가?](#how-can-i-obtain-the-output-of-an-intermediate-layer)
- [메모리 허용치 보다 큰 데이터셋을 Keras에서 어떻게 사용하 수 있는가?](#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory)
- [검증 손실이 더 이상 감소핮 않을때, 어떻게 학습을 중단할 수 있는가?](#how-can-i-interrupt-training-when-the-validation-loss-isnt-decreasing-anymore)
- [검증의 나뉨이 어떻게 계산 되는가?](#how-is-the-validation-split-computed)
- [학습 도중에 데이터가 뒤섞이는가?](#is-the-data-shuffled-during-training)
- [어떻게 매 epoch마다 학습 / 검증 손실 / 정확도를 기록할 수 있는가?](#how-can-i-record-the-training--validation-loss--accuracy-at-each-epoch)
- [How can I "freeze" layers?](#how-can-i-freeze-keras-layers)
- [How can I use stateful RNNs?](#how-can-i-use-stateful-rnns)
- [How can I remove a layer from a Sequential model?](#how-can-i-remove-a-layer-from-a-sequential-model)
- [How can I use pre-trained models in Keras?](#how-can-i-use-pre-trained-models-in-keras)
- [How can I use HDF5 inputs with Keras?](#how-can-i-use-hdf5-inputs-with-keras)
- [Where is the Keras configuration file stored?](#where-is-the-keras-configuration-file-stored)
- [How can I obtain reproducible results using Keras during development?](#how-can-i-obtain-reproducible-results-using-keras-during-development)
- [How can I install HDF5 or h5py to save my models in Keras?](#how-can-i-install-hdf5-or-h5py-to-save-my-models-in-keras)

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

### How can I "freeze" Keras layers?

To "freeze" a layer means to exclude it from training, i.e. its weights will never be updated. This is useful in the context of fine-tuning a model, or using fixed embeddings for a text input.

You can pass a `trainable` argument (boolean) to a layer constructor to set a layer to be non-trainable:

```python
frozen_layer = Dense(32, trainable=False)
```

Additionally, you can set the `trainable` property of a layer to `True` or `False` after instantiation. For this to take effect, you will need to call `compile()` on your model after modifying the `trainable` property. Here's an example:

```python
x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)

frozen_model = Model(x, y)
# in the model below, the weights of `layer` will not be updated during training
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = True
trainable_model = Model(x, y)
# with this model the weights of the layer will be updated during training
# (which will also affect the above model since it uses the same layer instance)
trainable_model.compile(optimizer='rmsprop', loss='mse')

frozen_model.fit(data, labels)  # this does NOT update the weights of `layer`
trainable_model.fit(data, labels)  # this updates the weights of `layer`
```

---

### How can I use stateful RNNs?

Making a RNN stateful means that the states for the samples of each batch will be reused as initial states for the samples in the next batch.

When using stateful RNNs, it is therefore assumed that:

- all batches have the same number of samples
- If `x1` and `x2` are successive batches of samples, then `x2[i]` is the follow-up sequence to `x1[i]`, for every `i`.

To use statefulness in RNNs, you need to:

- explicitly specify the batch size you are using, by passing a `batch_size` argument to the first layer in your model. E.g. `batch_size=32` for a 32-samples batch of sequences of 10 timesteps with 16 features per timestep.
- set `stateful=True` in your RNN layer(s).
- specify `shuffle=False` when calling `fit()`.

To reset the states accumulated:

- use `model.reset_states()` to reset the states of all layers in the model
- use `layer.reset_states()` to reset the states of a specific stateful RNN layer

Example:

```python
x  # this is our input data, of shape (32, 21, 16)
# we will feed it to our model in sequences of length 10

model = Sequential()
model.add(LSTM(32, input_shape=(10, 16), batch_size=32, stateful=True))
model.add(Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# we train the network to predict the 11th timestep given the first 10:
model.train_on_batch(x[:, :10, :], np.reshape(x[:, 10, :], (32, 16)))

# the state of the network has changed. We can feed the follow-up sequences:
model.train_on_batch(x[:, 10:20, :], np.reshape(x[:, 20, :], (32, 16)))

# let's reset the states of the LSTM layer:
model.reset_states()

# another way to do it in this case:
model.layers[0].reset_states()
```

Note that the methods `predict`, `fit`, `train_on_batch`, `predict_classes`, etc. will *all* update the states of the stateful layers in a model. This allows you to do not only stateful training, but also stateful prediction.

---

### How can I remove a layer from a Sequential model?

You can remove the last added layer in a Sequential model by calling `.pop()`:

```python
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(32, activation='relu'))

print(len(model.layers))  # "2"

model.pop()
print(len(model.layers))  # "1"
```

---

### How can I use pre-trained models in Keras?

Code and pre-trained weights are available for the following image classification models:

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

They can be imported from the module `keras.applications`:

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

For a few simple usage examples, see [the documentation for the Applications module](/applications).

For a detailed example of how to use such a pre-trained model for feature extraction or for fine-tuning, see [this blog post](http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).

The VGG16 model is also the basis for several Keras example scripts:

- [Style transfer](https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py)
- [Feature visualization](https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py)
- [Deep dream](https://github.com/keras-team/keras/blob/master/examples/deep_dream.py)

---

### How can I use HDF5 inputs with Keras?

You can use the `HDF5Matrix` class from `keras.utils`. See [the HDF5Matrix documentation](/utils/#hdf5matrix) for details.

You can also directly use a HDF5 dataset:

```python
import h5py
with h5py.File('input/file.hdf5', 'r') as f:
    x_data = f['x_data']
    model.predict(x_data)
```

Please also see [How can I install HDF5 or h5py to save my models in Keras?](#how-can-i-install-hdf5-or-h5py-to-save-my-models-in-keras) for instructions on how to install `h5py`.

---

### Where is the Keras configuration file stored?

The default directory where all Keras data is stored is:

```bash
$HOME/.keras/
```

Note that Windows users should replace `$HOME` with `%USERPROFILE%`.
In case Keras cannot create the above directory (e.g. due to permission issues), `/tmp/.keras/` is used as a backup.

The Keras configuration file is a JSON file stored at `$HOME/.keras/keras.json`. The default configuration file looks like this:

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

It contains the following fields:

- The image data format to be used as default by image processing layers and utilities (either `channels_last` or `channels_first`).
- The `epsilon` numerical fuzz factor to be used to prevent division by zero in some operations.
- The default float data type.
- The default backend. See the [backend documentation](/backend).

Likewise, cached dataset files, such as those downloaded with [`get_file()`](/utils/#get_file), are stored by default in `$HOME/.keras/datasets/`.

---

### How can I obtain reproducible results using Keras during development?

During development of a model, sometimes it is useful to be able to obtain reproducible results from run to run in order to determine if a change in performance is due to an actual model or data modification, or merely a result of a new random sample.

First, you need to set the `PYTHONHASHSEED` environment variable to `0` before the program starts (not within the program itself). This is necessary in Python 3.2.3 onwards to have reproducible behavior for certain hash-based operations (e.g., the item order in a set or a dict, see [Python's documentation](https://docs.python.org/3.7/using/cmdline.html#envvar-PYTHONHASHSEED) or [issue #2280](https://github.com/keras-team/keras/issues/2280#issuecomment-306959926) for further details). One way to set the environment variable is when starting python like this:

```
$ cat test_hash.py
print(hash("keras"))
$ python3 test_hash.py                  # non-reproducible hash (Python 3.2.3+)
-8127205062320133199
$ python3 test_hash.py                  # non-reproducible hash (Python 3.2.3+)
3204480642156461591
$ PYTHONHASHSEED=0 python3 test_hash.py # reproducible hash
4883664951434749476
$ PYTHONHASHSEED=0 python3 test_hash.py # reproducible hash
4883664951434749476
```

Moreover, when using the TensorFlow backend and running on a GPU, some operations have non-deterministic outputs, in particular `tf.reduce_sum()`. This is due to the fact that GPUs run many operations in parallel, so the order of execution is not always guaranteed. Due to the limited precision of floats, even adding several numbers together may give slightly different results depending on the order in which you add them. You can try to avoid the non-deterministic operations, but some may be created automatically by TensorFlow to compute the gradients, so it is much simpler to just run the code on the CPU. For this, you can set the `CUDA_VISIBLE_DEVICES` environment variable to an empty string, for example:

```
$ CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python your_program.py
```

The below snippet of code provides an example of how to obtain reproducible results - this is geared towards a TensorFlow backend for a Python 3 environment:

```python
import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Rest of code follows ...
```

---

### How can I install HDF5 or h5py to save my models in Keras?

In order to save your Keras models as HDF5 files, e.g. via
`keras.callbacks.ModelCheckpoint`, Keras uses the h5py Python package. It is
 a dependency of Keras and should be installed by default. On Debian-based
 distributions, you will have to additionally install `libhdf5`:

```
sudo apt-get install libhdf5-serial-dev
```

If you are unsure if h5py is installed you can open a Python shell and load the
module via

```
import h5py
```

If it imports without error it is installed, otherwise you can find detailed
installation instructions here: http://docs.h5py.org/en/latest/build.html
