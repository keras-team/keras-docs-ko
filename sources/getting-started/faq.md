# Keras FAQ: 자주 등장하는 케라스 질문들 모음입니다.

- [Keras를 어떻게 인용해야 합니까?](#how-should-i-cite-keras)
- [GPU로 Keras를 실행하려면 어떻게 해야 합니까?](#how-can-i-run-keras-on-gpu)
- [여러 대의 GPU로 Keras를 실행하려면 어떻게 해야 합니까?](#how-can-i-run-a-keras-model-on-multiple-gpus)
- ["sample", "batch", "epoch"가 의미하는 것이 무엇입니까?](#what-does-sample-batch-epoch-mean)
- [어떻게 Keras 모델을 저장할 수 있습니까?](#how-can-i-save-a-keras-model)
- [왜 training loss가 testing loss보다 훨씬 높습니까?](#why-is-the-training-loss-much-higher-than-the-testing-loss)
- [어떻게 중간층의 출력을 얻을 수 있습니까?](#how-can-i-obtain-the-output-of-an-intermediate-layer)
- [메모리 사이즈에 맞지 않는 데이터 셋을 Keras에서 사용하려면 어떻게 해야 합니까?](#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory)
- [validation loss가 더이상 감소하지 않을때 어떻게 하면 학습을 중단시킬 수 있습니까?](#how-can-i-interrupt-training-when-the-validation-loss-isnt-decreasing-anymore)
- [validation 세트는 어떤 식으로 분할됩니까?](#how-is-the-validation-split-computed)
- [학습 수행 시 데이터가 무작위로 섞입니까?](#is-the-data-shuffled-during-training)
- [어떻게 각 epoch별 training / validation loss / accuracy를 기록할 수 있습니까?](#how-can-i-record-the-training--validation-loss--accuracy-at-each-epoch)
- [어떻게 레이어를 잠글 수 있습니까?](#how-can-i-freeze-keras-layers)
- [stateful RNNs는 어떻게 사용할 수 있습니까?](#how-can-i-use-stateful-rnns)
- [Sequential model에서 레이어를 삭제하려면 어떻게 해야 합니까?](#how-can-i-remove-a-layer-from-a-sequential-model)
- [Keras에서 사전 훈련된 모델을 사용하려면 어떻게 해야 합니까?](#how-can-i-use-pre-trained-models-in-keras)
- [Keras에서 HDF5 입력을 사용하려면 어떻게 해야 합니까?](#how-can-i-use-hdf5-inputs-with-keras)
- [Keras 설정 파일은 어디에 저장됩니까?](#where-is-the-keras-configuration-file-stored)
- [Keras를 사용하여 개발하는 중에 어떻게 다시 재현 가능한 결과를 얻을 수 있습니까?](#how-can-i-obtain-reproducible-results-using-keras-during-development)
- [Keras에서 내 모델을 저장하기 위해서 어떻게 하면 HDF5 또는 h5py를 설치할 수 있습니까?](#how-can-i-install-hdf5-or-h5py-to-save-my-models-in-keras)

---

### Keras를 어떻게 인용해야 합니까?

귀하의 연구에 Keras가 도움이 되셨다면 귀하의 출판물에 Keras를 인용하십시오. 다음은 BibTeX entry 예시입니다:

```
@misc{chollet2015keras,
  title={Keras},
  author={Chollet, Fran\c{c}ois and others},
  year={2015},
  howpublished={\url{https://keras.io}},
}
```

---

### GPU로 Keras를 실행하려면 어떻게 해야 합니까?

만일 **TensorFlow** 또는 **CNTK** 백엔드에서 실행 중이라면, 여러분의 코드는 사용 가능한 GPU가 있을 시 자동으로 GPU상에서 구동됩니다. 

만일 **Theano** 백엔드에서 실행 중이라면, 여러분은 다음 중 하나의 방법을 따라야 합니다. 

**방법 1**: Theano flags 사용하기.
```bash
THEANO_FLAGS=device=gpu,floatX=float32 python my_keras_script.py
```

기기의 식별자에 따라 'gpu' 이름을 변경해야 할 수 있습니다. (e.g. `gpu0`, `gpu1`, 등등).

**방법 2**: 여러분의 `.theanorc`를 설정합니다: [설명서](http://deeplearning.net/software/theano/library/config.html)

**방법 3**: 여러분의 코드가 시작하는 부분에 수동으로 `theano.config.device`, `theano.config.floatX`를 설정하십시오:
```python
import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'
```

---

### 여러 대의 GPU로 Keras를 실행하려면 어떻게 해야 합니까?

여러 대의 GPU를 사용하는 경우 저희는 **TensorFlow** 백엔드 사용을 권장합니다. 여러 GPU에서 단일 모델을 실행하는 데엔 **데이터 병렬 처리**와 **장치 병렬 처리** 두 가지 방법이 있습니다.

대부분의 경우에서 여러분이 필요한 것은 데이터 병렬 처리일 것입니다. 

#### 데이터 병렬 처리

데이터 병렬 처리는 각 장치에서 타겟 모델을 한번 복제한 후 각 복제본을 이용하여 입력 데이터의 각각 다른 일부분을 처리하게끔 되어 있습니다. 
Keras는 `keras.utils.multi_gpu_model`라는 내부 유틸리티 함수를 내장하고 있는데, 이것은 어떠한 모델이던지 간에 데이터를 병렬로 처리할 수 있도록 해 줍니다. 또한 최대 8 GPU의 준 선형 속도 향상 기능을 제공합니다. 

더 많은 정보를 얻고 싶으시다면, 다음의 문서를 참고하십시오: [multi_gpu_model](/utils/#multi_gpu_model). 다음은 쉬운 사용법 예시입니다:

```python
from keras.utils import multi_gpu_model

# 8 GPUs에서 `model`를 복제합니다.
# 여기선 컴퓨터에 8개의 사용 가능한 GPU가 있다고 가정합니다.
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# 이 `fit` 함수 호출은 8 GPUs에 대해 분배됩니다.
# 배치 크기가 256이므로, 각 GPU는 32개의 샘플들을 처리할 것입니다. 
parallel_model.fit(x, y, epochs=20, batch_size=256)
```

#### 장치 병렬 처리

장치 병렬 처리는 다른 장치에서 동일한 모델의 각각 다른 일부분을 실행합니다. 이 방법은 병렬 구조를 가진 모델에서 가장 잘 작동합니다. (예: 2개의 분기가 있는 모델)

이 작업은 텐서플로의 device scopes를 통해 수행할 수 있습니다. 다음은 쉬운 사용법 예시입니다:

```python
# Model where a shared LSTM is used to encode two different sequences in parallel
input_a = keras.Input(shape=(140, 256))
input_b = keras.Input(shape=(140, 256))

shared_lstm = keras.layers.LSTM(64)

# Process the first sequence on one GPU
with tf.device_scope('/gpu:0'):
    encoded_a = shared_lstm(tweet_a)
# Process the next sequence on another GPU
with tf.device_scope('/gpu:1'):
    encoded_b = shared_lstm(tweet_b)

# Concatenate results on CPU
with tf.device_scope('/cpu:0'):
    merged_vector = keras.layers.concatenate([encoded_a, encoded_b],
                                             axis=-1)
```

---

### "sample", "batch", "epoch"가 의미하는 것이 무엇입니까?

Below are some common definitions that are necessary to know and understand to correctly utilize Keras:

- **Sample**: one element of a dataset.
  - *Example:* one image is a **sample** in a convolutional network
  - *Example:* one audio file is a **sample** for a speech recognition model
- **Batch**: a set of *N* samples. The samples in a **batch** are processed independently, in parallel. If training, a batch results in only one update to the model.
  - A **batch** generally approximates the distribution of the input data better than a single input. The larger the batch, the better the approximation; however, it is also true that the batch will take longer to process and will still result in only one update. For inference (evaluate/predict), it is recommended to pick a batch size that is as large as you can afford without going out of memory (since larger batches will usually result in faster evaluation/prediction).
- **Epoch**: an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
  - When using `validation_data` or `validation_split` with the `fit` method of Keras models, evaluation will be run at the end of every **epoch**.
  - Within Keras, there is the ability to add [callbacks](https://keras.io/callbacks/) specifically designed to be run at the end of an **epoch**. Examples of these are learning rate changes and model checkpointing (saving).

---

### 어떻게 Keras 모델을 저장할 수 있습니까?

#### 전체 모델을 저장/로딩하기 (구조 + 가중치들 + 옵티마이저의 상태)

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

#### 모델의 구조만 저장/로딩하는 방법

**모델의 구조**만 저장하고 싶다면, 가중치나 훈련과 관련한 설정들은 제외하고 싶다면, 이렇게 하시면 됩니다:

```python
# JSON 형태로 저장
json_string = model.to_json()

# YAML 형태로 저장
yaml_string = model.to_yaml()
```

생성된 JSON / YAML 파일은 사람이 읽을 수 있는 형태이며, 필요한 경우 수동으로 편집할 수 있습니다. 

그런 다음 이 데이터로 새로운 모델을 만들 수 있습니다:

```python
# JSON 파일로부터 새로운 모델 재구축하기:
from keras.models import model_from_json
model = model_from_json(json_string)

# YAML 파일로부터 새로운 모델 재구축하기:
from keras.models import model_from_yaml
model = model_from_yaml(yaml_string)
```

#### 모델의 가중치들만 저장/로딩하는 방법

**모델의 가중치**들만 저장하고 싶다면, HDF5를 통해 아래처럼 진행할 수 있습니다:

```python
model.save_weights('my_model_weights.h5')
```

여러분의 모델이 인스턴싱되어 있는 상태라면, 여러분은 저장되어 있는 **모델의 가중치**들을 *동일한* 구조를 가진 모델에 로드할 수 있습니다:

```python
model.load_weights('my_model_weights.h5')
```

여러분이 미세 조정이나 전이 학습을 위하여 *다른* 구조를 가진 모델에 저장되어 있는 가중치를 로드해야 한다면(일부 공통되는 레이어와 함께), 여러분은  *레이어 이름*을 통해서 이 가중치들을 로드할 수 있습니다:

```python
model.load_weights('my_model_weights.h5', by_name=True)
```

예시:

```python
"""
원래 모델이 다음과 같이 생겼다고 가정합시다:
    model = Sequential()
    model.add(Dense(2, input_dim=3, name='dense_1'))
    model.add(Dense(3, name='dense_2'))
    ...
    model.save_weights(fname)
"""

# 새로운 모델을 만들겠습니다.
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # 로드될 것입니다. 
model.add(Dense(10, name='new_dense'))  # 로드되지 않을 것입니다. 

# 첫 번째 모델에서 가중치를 로드합니다. 이것은 첫 번째 레이어, dense_1에만 영향을 끼칩니다.
model.load_weights(fname, by_name=True)
```

`h5py`를 설치하는 방법에 대해서는 [Keras에서 내 모델을 저장하기 위해서 어떻게 하면 HDF5 또는 h5py를 설치할 수 있습니까?](#how-can-i-install-hdf5-or-h5py-to-save-my-models-in-keras)를 참조하십시오.

#### 저장된 모델 속의 커스텀 레이어(또는 다른 커스텀 오브젝트)를 제어하는 방법

여러분이 로드하고자 하는 모델이 커스텀 레이어 또는 커스텀 클래스, 함수를 포함한다면, 그것들을 `custom_objects`인자를 통해 로딩 매커니즘에 전달할 수 있습니다. 

```python
from keras.models import load_model
# 여러분의 모델에 "AttentionLayer" 클래스의 인스턴스가 포함되어 있다고 가정합니다. 
model = load_model('my_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
```

또는, [custom object scope](https://keras.io/utils/#customobjectscope)를 사용할 :

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

### 왜 training loss가 testing loss보다 훨씬 높습니까?

A Keras model has two modes: training and testing. Regularization mechanisms, such as Dropout and L1/L2 weight regularization, are turned off at testing time.

Besides, the training loss is the average of the losses over each batch of training data. Because your model is changing over time, the loss over the first batches of an epoch is generally higher than over the last batches. On the other hand, the testing loss for an epoch is computed using the model as it is at the end of the epoch, resulting in a lower loss.

---

### 어떻게 중간층의 출력을 얻을 수 있습니까?

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

### 메모리 사이즈에 맞지 않는 데이터 셋을 Keras에서 사용하려면 어떻게 해야 합니까?

You can do batch training using `model.train_on_batch(x, y)` and `model.test_on_batch(x, y)`. See the [models documentation](/models/sequential).

Alternatively, you can write a generator that yields batches of training data and use the method `model.fit_generator(data_generator, steps_per_epoch, epochs)`.

You can see batch training in action in our [CIFAR10 example](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py).

---

### Validation loss가 더이상 감소하지 않을때 어떻게 하면 학습을 중단시킬 수 있습니까?

You can use an `EarlyStopping` callback:

```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
```

Find out more in the [callbacks documentation](/callbacks).

---

### Validation 세트는 어떤 식으로 분할됩니까?

If you set the `validation_split` argument in `model.fit` to e.g. 0.1, then the validation data used will be the *last 10%* of the data. If you set it to 0.25, it will be the last 25% of the data, etc. Note that the data isn't shuffled before extracting the validation split, so the validation is literally just the *last* x% of samples in the input you passed.

The same validation set is used for all epochs (within a same call to `fit`).

---

### 학습 수행 시 데이터가 무작위로 섞입니까?

Yes, if the `shuffle` argument in `model.fit` is set to `True` (which is the default), the training data will be randomly shuffled at each epoch.

Validation data is never shuffled.

---


### 어떻게 각 epoch별 training / validation loss / accuracy를 기록할 수 있습니까?

The `model.fit` method returns a `History` callback, which has a `history` attribute containing the lists of successive losses and other metrics.

```python
hist = model.fit(x, y, validation_split=0.2)
print(hist.history)
```

---

### 어떻게 Keras 레이어를 잠글 수 있습니까?

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

### Stateful RNNs는 어떻게 사용할 수 있습니까?

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

### Sequential model에서 레이어를 삭제하려면 어떻게 해야 합니까?

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

### Keras에서 사전 훈련된 모델을 사용하려면 어떻게 해야 합니까?

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

### Keras에서 HDF5 입력을 사용하려면 어떻게 해야 합니까?

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

### Keras 설정 파일은 어디에 저장됩니까?

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

### Keras를 사용하여 개발하는 중에 어떻게 다시 재현 가능한 결과를 얻을 수 있습니까?

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

### Keras에서 내 모델을 저장하기 위해서 어떻게 하면 HDF5 또는 h5py를 설치할 수 있습니까?

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
