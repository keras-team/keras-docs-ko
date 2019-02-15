## callbacks의 용도

callback은 주어진 학습 단계에서 적용될 수 있는 함수들의 집합입니다. 여러분은 모델의 훈련 도중 모델 내의 상태와 통계를 보기 위해 callbacks를 이용할 수 있습니다. callbacks의 리스트(키워드 인수 `callbacks`로)를 `Sequential` 또는 `Model` 클래스의 `.fit()` 메소드에 전달할 수 있습니다. 주어진 callbacks의 메소드는 훈련의 각 단계별로 호출될 것입니다.

---

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L274)</span>
### Callback

```python
keras.callbacks.Callback()
```

새로운 callbacks를 빌드하는데 사용되는 추상화 기초 클래스(abstract base class)입니다.

__Properties__

- __params__: 딕셔너리. 훈련 매개변수
    (eg. 자세한 정도, 배치 크기, 에포크 수 등...).
- __model__: 'keras.models.Model'의 인스턴스.
    훈련되고 있는 모델의 기준

callbacks의 메소드들이 인수로 받아들이는 'logs' 딕셔너리는 
현재 배치, 또는 에포크에 해당하는 키 값들을 갖고 있습니다.

지금은 'Sequential' 모델 클래스의 '.fit()' 메소드가 'logs'
에 다음과 같은 값들을 포함하고 있을것이며, 이 값들을 callbacks
에 넘깁니다.:

on_epoch_end: 한 에포크가 끝날 때 logs에는 'acc' 와 'loss',
그리고 선택적으로 'val_loss'(만약 validation이 'fit'에 활성화되어있다면)
와 'val_acc'(만약 validation과 accuracy 모니터링이 활성화 되어있다면)
를 포함합니다.
on_batch_begin: 한 배치의 시작점에 logs는 현재 배치에 들어 있는
샘플의 수인 'size'를 포함합니다.
on_batch_end: 한 배치가 끝나는 지점에 logs는 'loss'를 포함하고,
선택적으로 'acc'(만약 accuracy 모니터링이 활성화 되어있다면)를
포함합니다.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L476)</span>
### BaseLogger

```python
keras.callbacks.BaseLogger(stateful_metrics=None)
```
측정 단위의 에포크별 평균을 누적시키는 콜백

이 콜백은 모든 케라스 모델에 자동으로 적용됩니다.

__Arguments__

- __stateful_metrics__: 에포크 단위별로 평균을 취하지 *말아야* 할
    반복될 수 있는 측정 단위의 이름(스트링)
    이 리스트에 해당하는 측정 단위들은 `on_epoch_end` log에 값 그대로
    포함 되며, 다른 측정 단위들은 `on_epoch_end`에 평균되어 들어갑니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L523)</span>
### TerminateOnNaN

```python
keras.callbacks.TerminateOnNaN()
```

훈련 도중 loss 값이 NaN이 나왔을 때 종결시키는 콜백.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L536)</span>
### ProgbarLogger

```python
keras.callbacks.ProgbarLogger(count_mode='samples', stateful_metrics=None)
```

stdout 에 측정단위들을 프린트하는 콜백

__Arguments__

- __count_mode__: "steps"나 "samples"중 하나.
    진행 표시줄이 샘플 갯수를 셀 지 배치 단계를 셀 지 정해줍니다.
- __stateful_metrics__: 에포크 단위별로 평균을 취하지 *말아야* 할
    반복될 수 있는 측정 단위의 이름(스트링)
    이 리스트에 해당하는 측정 단위들은 값 그대로
    포함 되며, 다른 측정 단위들은 평균되어 들어갑니다.(eg. loss, etc..)

__Raises__

- __ValueError__: 적절치 않은 'count_mode'가 주어졌을 때.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L613)</span>
### History

```python
keras.callbacks.History()
```

`History` 오브젝트에 이벤트를 기록하는 콜백.

이 콜백은 모든 케라스 모델에 자동적으로 적용됩니다.
`History` 오브젝트는 모델의 `fit` 메소드에 의해
반환됩니다.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L632)</span>
### ModelCheckpoint

```python
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
```

모든 에포크 이후에 모델을 저장합니다.

`filepath`는 포매팅 옵션으로 경로를 지정할 수 있으며,
`epoch`의 값들과 `logs`(`on_epoch_end`로부터 전달된)
값들을 갖고 있게 됩니다.

예를 들어: 만약 `filepath` 가 `weights.{epoch:02d}-{val_loss:.2f}.hdf5`라면,
모델 체크포인트는 파일명에 에포크의 수와 validation loss를 포함하게 됩니다.


__Arguments__

- __filepath__: 스트링, 모델을 저장하는 경로
- __monitor__: 모니터하는 값
- __verbose__: 자세함의 정도를 설정하는 모드, 0 또는 1.
- __save_best_only__: `save_best_only=True` 라면,
    측정된 값 기준 가장 좋은 모델 중 마지막 모델이 더
    새로운 모델에 의해 대체되지 않습니다.
- __mode__: {auto, min, max}중 하나
    `save_best_only=True`라면, 현재 저장된 파일을 겹쳐 쓸지에
    대한 결정이 모니터되는 값의 최대화, 또는 최소화에 의해 결정
    됩니다. `val_acc`의 경우 `max`가 되어야 하고, `val_loss`의
    경우 `min`이 되어야 하는 식입니다. `auto` 모드에서는 측정되는
    값의 이름에 따라 자동으로 이 방향이 결정됩니다.
- __save_weights_only__: True 라면, 모델의 가중치만 
    (`model.save_weights(filepath)`)로 저장되고, False 라면 전체
    모드가 (`model.save(filepath)`)로 저장이 됩니다.
- __period__: 체크포인트 간극(에포크의 수).
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L732)</span>
### EarlyStopping

```python
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
```

측정되는 수치가 더 이상 개선되지 않을 때 훈련을 멈춥니다.

__Arguments__

- __monitor__: 모니터되는 측정값
- __min_delta__: 모니터되는 측정값이 개선이라고 판단될 수 있는
    최소한의 변화, i.e. min_delta의 절댓값보다 작은 값이 바뀔
    경우, 개선이라고 판단되지 않습니다.
- __patience__: 값이 더 이상 개선이 되지 않을 때 훈련이 중단
    되기 전까지의 에포크 수
- __verbose__: 자세함의 정도를 설정하는 모드.
- __mode__: {auto, min, max}중 하나. `min` 모드에서는,
    수치가 더 이상 감소하지 않을 때 훈련이 멈춥니다;
    `max` 모드에서는, 수치가 더 이상 증가하지 않을 때
    훈련이 멈춥니다; `auto` 모드에서는, 측정되는 수치의
    이름에 따라 해당하는 방향이 자동으로 결정됩니다.
- __baseline__: 모니터되는 값이 가져야 할 기준치.
    이 기준치 이상 모델이 더 좋아지지 않을 때 훈련은 멈춥니다.
- __restore_best_weights__: 측정된 수치가 가장 좋았을 때의 모델
    가중치를 다시 불러올 지를 정해줍니다. False 라면, 훈련의
    마지막 단계에서 얻은 가중치를 사용합니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L846)</span>
### RemoteMonitor

```python
keras.callbacks.RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None, send_as_json=False)
```

서버로 스트리밍 할 때 이용되는 콜백.

`requests` 라이브러리를 필요로 합니다.
사건들은 디폴트로 `root + '/publish/epoch/end/'` 로 보내집니다. 콜은 HTTP 포스트
로 지정되고, `data` 전달인자에 JSON으로 인코딩된 데이터가 들어갑니다. send_as_json이
True로 설정되면 내용물의 형식이 application/json이 될 것이고, 그렇지 않다면 JSON이
직렬화되어 전송됩니다.

__Arguments__

- __root__: 스트링; 타겟 서버의 루트 url
- __path__: 스트링; `root` 에 상대적인 경로로, 이벤트가 보내지는 경로
- __field__: 스트링; 데이터가 저장되는 JSON의 field.
    정보가 형식에 맞춰 보내질 때(send_as_json이 False일 때)만 field가 이용됩니다.
- __headers__: 딕셔너리; 선택적인 커스텀 HTTP 헤더.
- __send_as_json__: 불린 자료형; 데이터가 application/json으로 보내질지의 여부.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L905)</span>
### LearningRateScheduler

```python
keras.callbacks.LearningRateScheduler(schedule, verbose=0)
```

학습 속도 스케쥴러.

__Arguments__

- __schedule__: 입력으로 에포크 인덱스(정수, 인덱스는 0에서 시작)
    와 학습 속도를 받고 새로운 학습 속도를 출력합니다(부동소수점).
- __verbose__: 정수. 0: 메세지를 출력하지 않습니다, 1: 메세지를 업데이트 합니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L941)</span>
### TensorBoard

```python
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
```

텐서보드의 기본적인 시각화.

[텐서보드](https://www.tensorflow.org/guide/summaries_and_tensorboard)
는 텐서플로우와 함께 제공되는 시각화 도구입니다.

이 콜백은 텐서보드를 위한 로그를 작성하며, 여러분의 훈련과 테스트 척도에 대한
동적 
This callback writes a log for TensorBoard, which allows
you to visualize dynamic graphs of your training and test
metrics, as well as activation histograms for the different
layers in your model.

If you have installed TensorFlow with pip, you should be able
to launch TensorBoard from the command line:
```sh
tensorboard --logdir=/full_path_to_your_logs
```

When using a backend other than TensorFlow, TensorBoard will still work
(if you have TensorFlow installed), but the only feature available will
be the display of the losses and metrics plots.

__Arguments__

- __log_dir__: the path of the directory where to save the log
    files to be parsed by TensorBoard.
- __histogram_freq__: frequency (in epochs) at which to compute activation
    and weight histograms for the layers of the model. If set to 0,
    histograms won't be computed. Validation data (or split) must be
    specified for histogram visualizations.
- __write_graph__: whether to visualize the graph in TensorBoard.
    The log file can become quite large when
    write_graph is set to True.
- __write_grads__: whether to visualize gradient histograms in TensorBoard.
    `histogram_freq` must be greater than 0.
- __batch_size__: size of batch of inputs to feed to the network
    for histograms computation.
- __write_images__: whether to write model weights to visualize as
    image in TensorBoard.
- __embeddings_freq__: frequency (in epochs) at which selected embedding
    layers will be saved. If set to 0, embeddings won't be computed.
    Data to be visualized in TensorBoard's Embedding tab must be passed
    as `embeddings_data`.
- __embeddings_layer_names__: a list of names of layers to keep eye on. If
    None or empty list all the embedding layer will be watched.
- __embeddings_metadata__: a dictionary which maps layer name to a file name
    in which metadata for this embedding layer is saved. See the
    [details](https://www.tensorflow.org/guide/embedding#metadata)
    about metadata files format. In case if the same metadata file is
    used for all embedding layers, string can be passed.
- __embeddings_data__: data to be embedded at layers specified in
    `embeddings_layer_names`. Numpy array (if the model has a single
    input) or list of Numpy arrays (if the model has multiple inputs).
    Learn [more about embeddings](
    https://www.tensorflow.org/guide/embedding).
- __update_freq__: `'batch'` or `'epoch'` or integer. When using `'batch'`, writes
    the losses and metrics to TensorBoard after each batch. The same
    applies for `'epoch'`. If using an integer, let's say `10000`,
    the callback will write the metrics and losses to TensorBoard every
    10000 samples. Note that writing too frequently to TensorBoard
    can slow down your training.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L1286)</span>
### ReduceLROnPlateau

```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
```

Reduce learning rate when a metric has stopped improving.

Models often benefit from reducing the learning rate by a factor
of 2-10 once learning stagnates. This callback monitors a
quantity and if no improvement is seen for a 'patience' number
of epochs, the learning rate is reduced.

__Example__


```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(X_train, Y_train, callbacks=[reduce_lr])
```

__Arguments__

- __monitor__: quantity to be monitored.
- __factor__: factor by which the learning rate will
    be reduced. new_lr = lr * factor
- __patience__: number of epochs with no improvement
    after which learning rate will be reduced.
- __verbose__: int. 0: quiet, 1: update messages.
- __mode__: one of {auto, min, max}. In `min` mode,
    lr will be reduced when the quantity
    monitored has stopped decreasing; in `max`
    mode it will be reduced when the quantity
    monitored has stopped increasing; in `auto`
    mode, the direction is automatically inferred
    from the name of the monitored quantity.
- __min_delta__: threshold for measuring the new optimum,
    to only focus on significant changes.
- __cooldown__: number of epochs to wait before resuming
    normal operation after lr has been reduced.
- __min_lr__: lower bound on the learning rate.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L1407)</span>
### CSVLogger

```python
keras.callbacks.CSVLogger(filename, separator=',', append=False)
```

Callback that streams epoch results to a csv file.

Supports all values that can be represented as a string,
including 1D iterables such as np.ndarray.

__Example__


```python
csv_logger = CSVLogger('training.log')
model.fit(X_train, Y_train, callbacks=[csv_logger])
```

__Arguments__

- __filename__: filename of the csv file, e.g. 'run/log.csv'.
- __separator__: string used to separate elements in the csv file.
- __append__: True: append if file exists (useful for continuing
    training). False: overwrite existing file,
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L1495)</span>
### LambdaCallback

```python
keras.callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
```

Callback for creating simple, custom callbacks on-the-fly.

This callback is constructed with anonymous functions that will be called
at the appropriate time. Note that the callbacks expects positional
arguments, as:

- `on_epoch_begin` and `on_epoch_end` expect two positional arguments:
`epoch`, `logs`
- `on_batch_begin` and `on_batch_end` expect two positional arguments:
`batch`, `logs`
- `on_train_begin` and `on_train_end` expect one positional argument:
`logs`

__Arguments__

- __on_epoch_begin__: called at the beginning of every epoch.
- __on_epoch_end__: called at the end of every epoch.
- __on_batch_begin__: called at the beginning of every batch.
- __on_batch_end__: called at the end of every batch.
- __on_train_begin__: called at the beginning of model training.
- __on_train_end__: called at the end of model training.

__Example__


```python
# Print the batch number at the beginning of every batch.
batch_print_callback = LambdaCallback(
    on_batch_begin=lambda batch,logs: print(batch))

# Stream the epoch loss to a file in JSON format. The file content
# is not well-formed JSON but rather has a JSON object per line.
import json
json_log = open('loss_log.json', mode='wt', buffering=1)
json_logging_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
    on_train_end=lambda logs: json_log.close()
)

# Terminate some processes after having finished model training.
processes = ...
cleanup_callback = LambdaCallback(
    on_train_end=lambda logs: [
        p.terminate() for p in processes if p.is_alive()])

model.fit(...,
          callbacks=[batch_print_callback,
                     json_logging_callback,
                     cleanup_callback])
```
    

---


# Create a callback

You can create a custom callback by extending the base class `keras.callbacks.Callback`. A callback has access to its associated model through the class property `self.model`.

Here's a simple example saving a list of losses over each batch during training:
```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
```

---

### Example: recording loss history

```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = LossHistory()
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, callbacks=[history])

print(history.losses)
# outputs
'''
[0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
'''
```

---

### Example: model checkpoints

```python
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

'''
saves the model weights after each epoch if the validation loss decreased
'''
checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])
```
