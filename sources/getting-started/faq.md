# 케라스 FAQ: 자주 묻는 케라스 질문

- [케라스는 어떻게 인용해야 합니까?](#케라스는-어떻게-인용해야-합니까)
- [케라스를 GPU에서 실행하려면 어떻게 해야 합니까?](#케라스를-gpu에서-실행하려면-어떻게-해야-합니까)
- [케라스를 여러 대의 GPU에서 실행하려면 어떻게 해야 합니까?](#케라스를-여러-대의-gpu에서-실행하려면-어떻게-해야-합니까)
- ["샘플", "배치", "에폭"은 무슨 뜻입니까?](#샘플-배치-에폭은-무슨-뜻입니까)
- [케라스 모델은 어떻게 저장합니까?](#케라스-모델은-어떻게-저장합니까)
- [왜 훈련 손실이 테스트 손실보다 훨씬 높습니까?](#왜-훈련-손실이-테스트-손실보다-훨씬-높습니까)
- [중간층의 아웃풋은 어떻게 얻을 수 있습니까?](#중간층의-아웃풋은-어떻게-얻을-수-있습니까)
- [메모리를 초과하는 데이터셋에 케라스를 사용하려면 어떻게 해야 합니까?](#메모리를-초과하는-데이터셋에-케라스를-사용하려면-어떻게-해야-합니까)
- [검증 손실이 더이상 감소하지 않는 경우 훈련을 어떻게 중단할 수 있습니까?](#검증-손실이-더이상-감소하지-않는-경우-훈련을-어떻게-중단할-수-있습니까)
- [검증 분리는 어떻게 계산됩니까?](#검증-분리는-어떻게-계산됩니까)
- [훈련 중 데이터가 섞입니까?](#훈련-중-데이터가-섞입니까)
- [각 에폭별 훈련 손실 / 검증 손실 / 정확도는 어떻게 기록할 수 있습니까?](#각-에폭별-훈련-손실--검증-손실--정확도는-어떻게-기록할-수-있습니까)
- [케라스 층들을 "동결" 시키려면 어떻게 해야 합니까?](#케라스-층들을-동결-시키려면-어떻게-해야-합니까)
- [상태형 순환 신경망을 사용하려면 어떻게 해야 합니까?](#상태형-순환-신경망을-사용하려면-어떻게-해야-합니까)
- [Sequential 모델에서 층을 없애려면 어떻게 해야 합니까?](#sequential-모델에서-층을-없애려면-어떻게-해야-합니까)
- [케라스에서 선행 훈련된 모델을 사용하려면 어떻게 해야 합니까?](#케라스에서-선행-훈련된-모델을-사용하려면-어떻게-해야-합니까)
- [케라스에서 HDF5 인풋을 사용하려면 어떻게 해야 합니까?](#케라스에서-hdf5-인풋을-사용하려면-어떻게-해야-합니까)
- [케라스 구성 파일은 어디에 저장됩니까?](#케라스-구성-파일은-어디에-저장됩니까)
- [개발 중 케라스를 사용해 재현 가능한 결과를 얻으려면 어떻게 해야 합니까?](#개발-중-케라스를-사용해-재현-가능한-결과를-얻으려면-어떻게-해야-합니까)
- [HDF5나 h5py를 설치해 케라스 모델을 저장하려면 어떻게 해야 합니까?](#hdf5나-h5py를-설치해-케라스-모델을-저장하려면-어떻게-해야-합니까)

---

### 케라스는 어떻게 인용해야 합니까?

연구 중 케라스가 도움이 되었다면 출판 시 케라스를 인용해 주십시오. 다음은 BibTeX 등재 예시입니다:

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

**텐서플로** 혹은 **CNTK** 백엔드를 사용하는 경우, 사용 가능한 GPU가 감지되면 코드가 자동으로 GPU에서 작동됩니다.

**Theano** 백엔드를 사용하는 경우, 다음 방법 중 하나를 사용하시면 됩니다:

**방법 1**: Theano 플래그를 사용합니다.
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

### 케라스를 여러 대의 GPU에서 실행하려면 어떻게 해야 합니까??

**텐서플로** 백엔드를 사용하실 것을 권장합니다. 단일 모델을 여러 대의 GPU에 실행하려면 다음의 두 가지 방법이 있습니다: **데이터 병렬처리**와 **장치 병렬처리**.

대부분의 경우에는 데이터 병렬처리를 사용하면 됩니다.

#### 데이터 병렬처리

데이터 병렬처리는 표적 모델을 장치마다 복제하고, 각각의 복제본으로 인풋 데이터의 각기 다른 부분을 처리하는 방식으로 구성됩니다.
케라스의 내장 유틸리티 `keras.utils.multi_gpu_model`은 어떤 모델도 병렬 데이터 버전으로 만들 수 있으며, 8대의 GPU까지는 거의 선형적으로 속도를 증가시킬 수 있습니다.

더 자세한 정보는 [multi_gpu_model](/utils/#multi_gpu_model)문서를 참고하십시오. 다음은 간단한 예시입니다:

```python
from keras.utils import multi_gpu_model

# `model`을 8대의 GPU에 복제합니다.
# 이 예시에서는 장치에 8대의 GPU가 있는 것으로 가정합니다.
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# `fit` 호출이 8대 GPU에 분산됩니다.
# 배치 크기가 256이므로 각 GPU는 32개 샘플을 처리합니다.
parallel_model.fit(x, y, epochs=20, batch_size=256)
```

#### 장치 병렬처리

장치 병렬처리는 같은 모델의 다른 부분을 각기 다른 장치에서 실행하는 방식으로 구성됩니다. 두 브랜치로 구성된 모델과 같은 병렬 구조의 모델에 가장 적합합니다.

텐서플로 디바이스 스코프를 사용해 이를 구현할 수 있습니다. 다음은 간단한 예시입니다:

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

### "샘플", "배치", "에폭"은 무슨 뜻입니까?

다음은 케라스를 올바르게 이용하기 위해 알아두어야 할 몇 가지 일반적인 정의입니다:

- **샘플**: 데이터셋의 요소 하나
  - *예시:* 하나의 이미지는 합성곱 망의 **샘플**입니다
  - *예시:* 오디오 파일 하나는 음성 인식 모델의 **샘플**입니다.
- **배치**: *N* 샘플들의 모음. 하나의 **배치**에서의 샘플들은 독립적으로 동시에 처리됩니다. 훈련에서 하나의 배치는 모델을 1회만 업데이트합니다.
  - **배치**는 일반적으로 단일 인풋보다 인풋 데이터에서의 분포를 더욱 정확히 추정합니다. 배치가 클수록 추정치도 개선됩니다; 하지만 단 1회의 업데이트를 위해 배치 처리에 더 오랜 시간을 소모하는 것도 사실입니다. 유추(평가/예측)를 위해서는 메모리를 초과하지 않는 선에서 배치 크기를 가능한 한 크게 하는 것을 권장합니다(배치가 크면 보통 평가/예측도 빨라지기 때문입니다).
- **에폭**: 임의의 차단 장치로, 보통 "전체 데이터셋을 1회 통과하는 것"으로 정의됩니다. 훈련을 국면 단위로 구분하는 데에 사용되며, 이는 로그 기록이나 주기적 평가에 유용합니다.
  - 케라스 모델의 `fit` 메소드와 함께 `validation_data`, 혹은 `validation_split`을 사용하는 경우, 각 **에폭**이 끝날 때마다 평가가 실행됩니다.
  - 케라스에서는 **에폭**의 끝에서 실행되도록 특별히 고안된 [콜백](https://keras.io/callbacks/)을 추가할 수 있습니다. 학습률 변화와 모델 체크포인트(저장)가 그 사례입니다.

---

### 케라스 모델은 어떻게 저장합니까?

#### Saving/loading whole models (architecture + weights + optimizer state)

*It is not recommended to use pickle or cPickle to save a Keras model.*

You can use `model.save(filepath)` to save a Keras model into a single HDF5 file which will contain:

- the architecture of the model, allowing to re-create the model
- the weights of the model
- the training configuration (loss, optimizer)
- the state of the optimizer, allowing to resume training exactly where you left off.

You can then use `keras.models.load_model(filepath)` to reinstantiate your model.
`load_model` will also take care of compiling the model using the saved training configuration (unless the model was never compiled in the first place).

Example:

```python
from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
```

Please also see [How can I install HDF5 or h5py to save my models in Keras?](#how-can-i-install-hdf5-or-h5py-to-save-my-models-in-keras) for instructions on how to install `h5py`.

#### Saving/loading only a model's architecture

If you only need to save the **architecture of a model**, and not its weights or its training configuration, you can do:

```python
# save as JSON
json_string = model.to_json()

# save as YAML
yaml_string = model.to_yaml()
```

The generated JSON / YAML files are human-readable and can be manually edited if needed.

You can then build a fresh model from this data:

```python
# model reconstruction from JSON:
from keras.models import model_from_json
model = model_from_json(json_string)

# model reconstruction from YAML:
from keras.models import model_from_yaml
model = model_from_yaml(yaml_string)
```

#### Saving/loading only a model's weights

If you need to save the **weights of a model**, you can do so in HDF5 with the code below:

```python
model.save_weights('my_model_weights.h5')
```

Assuming you have code for instantiating your model, you can then load the weights you saved into a model with the *same* architecture:

```python
model.load_weights('my_model_weights.h5')
```

If you need to load the weights into a *different* architecture (with some layers in common), for instance for fine-tuning or transfer-learning, you can load them by *layer name*:

```python
model.load_weights('my_model_weights.h5', by_name=True)
```

Example:

```python
"""
Assuming the original model looks like this:
    model = Sequential()
    model.add(Dense(2, input_dim=3, name='dense_1'))
    model.add(Dense(3, name='dense_2'))
    ...
    model.save_weights(fname)
"""

# new model
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # will be loaded
model.add(Dense(10, name='new_dense'))  # will not be loaded

# load weights from first model; will only affect the first layer, dense_1.
model.load_weights(fname, by_name=True)
```

Please also see [How can I install HDF5 or h5py to save my models in Keras?](#how-can-i-install-hdf5-or-h5py-to-save-my-models-in-keras) for instructions on how to install `h5py`.

#### Handling custom layers (or other custom objects) in saved models

If the model you want to load includes custom layers or other custom classes or functions, 
you can pass them to the loading mechanism via the `custom_objects` argument: 

```python
from keras.models import load_model
# Assuming your model includes instance of an "AttentionLayer" class
model = load_model('my_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
```

Alternatively, you can use a [custom object scope](https://keras.io/utils/#customobjectscope):

```python
from keras.utils import CustomObjectScope

with CustomObjectScope({'AttentionLayer': AttentionLayer}):
    model = load_model('my_model.h5')
```

Custom objects handling works the same way for `load_model`, `model_from_json`, `model_from_yaml`:

```python
from keras.models import model_from_json
model = model_from_json(json_string, custom_objects={'AttentionLayer': AttentionLayer})
```

---

### 왜 훈련 손실이 테스트 손실보다 훨씬 높습니까?

A Keras model has two modes: training and testing. Regularization mechanisms, such as Dropout and L1/L2 weight regularization, are turned off at testing time.

Besides, the training loss is the average of the losses over each batch of training data. Because your model is changing over time, the loss over the first batches of an epoch is generally higher than over the last batches. On the other hand, the testing loss for an epoch is computed using the model as it is at the end of the epoch, resulting in a lower loss.

---

### 중간층의 아웃풋은 어떻게 얻을 수 있습니까?

One simple way is to create a new `Model` that will output the layers that you are interested in:

```python
from keras.models import Model

model = ...  # create the original model

layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
```

Alternatively, you can build a Keras function that will return the output of a certain layer given a certain input, for example:

```python
from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([x])[0]
```

Similarly, you could build a Theano and TensorFlow function directly.

Note that if your model has a different behavior in training and testing phase (e.g. if it uses `Dropout`, `BatchNormalization`, etc.), you will need to pass the learning phase flag to your function:

```python
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])

# output in test mode = 0
layer_output = get_3rd_layer_output([x, 0])[0]

# output in train mode = 1
layer_output = get_3rd_layer_output([x, 1])[0]
```

---

### 메모리를 초과하는 데이터셋에 케라스를 사용하려면 어떻게 해야 합니까?

You can do batch training using `model.train_on_batch(x, y)` and `model.test_on_batch(x, y)`. See the [models documentation](/models/sequential).

Alternatively, you can write a generator that yields batches of training data and use the method `model.fit_generator(data_generator, steps_per_epoch, epochs)`.

You can see batch training in action in our [CIFAR10 example](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py).

---

### 검증 손실이 더이상 감소하지 않는 경우 훈련을 어떻게 중단할 수 있습니까?

You can use an `EarlyStopping` callback:

```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
```

Find out more in the [callbacks documentation](/callbacks).

---

### 검증 분리는 어떻게 계산됩니까?

If you set the `validation_split` argument in `model.fit` to e.g. 0.1, then the validation data used will be the *last 10%* of the data. If you set it to 0.25, it will be the last 25% of the data, etc. Note that the data isn't shuffled before extracting the validation split, so the validation is literally just the *last* x% of samples in the input you passed.

The same validation set is used for all epochs (within a same call to `fit`).

---

### 훈련 중 데이터가 섞입니까?

Yes, if the `shuffle` argument in `model.fit` is set to `True` (which is the default), the training data will be randomly shuffled at each epoch.

Validation data is never shuffled.

---


### 각 에폭별 훈련 손실 / 검증 손실 / 정확도는 어떻게 기록할 수 있습니까?

The `model.fit` method returns a `History` callback, which has a `history` attribute containing the lists of successive losses and other metrics.

```python
hist = model.fit(x, y, validation_split=0.2)
print(hist.history)
```

---

### 케라스 층들을 "동결" 시키려면 어떻게 해야 합니까?

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

### 상태형 순환 신경망을 사용하려면 어떻게 해야 합니까?

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

### Sequential 모델에서 층을 없애려면 어떻게 해야 합니까?

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

### 케라스에서 선행 훈련된 모델을 사용하려면 어떻게 해야 합니까?

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

### 케라스에서 HDF5 인풋을 사용하려면 어떻게 해야 합니까?

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

### 케라스 구성 파일은 어디에 저장됩니까?

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

### 개발 중 케라스를 사용해 재현 가능한 결과를 얻으려면 어떻게 해야 합니까?

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

### HDF5나 h5py를 설치해 케라스 모델을 저장하려면 어떻게 해야 합니까?

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
