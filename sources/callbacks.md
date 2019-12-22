## 콜백<sub>callbacks</sub>의 사용법

콜백은 학습 과정의 에폭<sub>Epoch</sub> 또는 배치<sub>Batch</sub>와 같은 각 단계마다 작동할 함수의 세트입니다. 콜백을 사용하면 학습이 진행되는 동안 모델의 내부 상태 및 통계적 지표의 변화를 관찰할 수 있습니다. 사용방법은 `.fit()` 메소드의 `callbacks` 인자에 콜백으로 이루어진 리스트를 전달하는 것입니다. Sequential 모델과 Model 클래스 모두에서 작동하며, 관련된 메소드들이 훈련의 각 단계마다 호출됩니다.  


---
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L275)</span>
### Callback
새로운 콜백 생성에 필요한 기본 클래스입니다.

__특성<sub>properties</sub>__

- __params__: 딕셔너리. 훈련 파라미터들 (예: 출력 여부<sub>verbosity</sub>, 배치 크기<sub>batch size</sub>, 에폭 수<sub>number of epochs</sub> 등).
- __model__: `keras.models.Model` 인스턴스. 훈련할 모델의 레퍼런스.  

콜백 메소드는 `logs`라는 이름의 딕셔너리를 인자로 받습니다. 이 딕셔너리는 매 에폭 또는 배치와 관련된 각종 수치들을 저장하는 값<sub>values</sub>과 이를 호출하는 키<sub>keys</sub>들로 이루어지게 됩니다.  

(현재 버전까지) Sequential 모델 클래스의 `.fit()`메소드의 경우 아래와 같은 수치를 콜백에 전달, `logs`에 기록합니다.  
- on_epoch_end: 기본적으로 `logs`에 `acc`와 `loss`를 기록하며, 교차검증<sub>validation</sub>이 적용된 경우 `val_loss`, 교차검증 및 정확도 추적이 적용된 경우 `val_acc`까지 저장됩니다.
- on_batch_begin: 현재 배치의 크기를 나타내는 `size`값이 `logs`에 저장됩니다.
- on_batch_end: `logs`에 `loss`를 기록하며 정확도 추적이 적용된 경우 `acc`도 함께 저장합니다. 


----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L477)</span>
### BaseLogger

```python
keras.callbacks.BaseLogger(stateful_metrics=None)
```

각 에폭별로 평가 지표들의 평균을 기록하는 콜백으로 모든 케라스 모델에 자동적으로 적용됩니다.

__인자__

- __stateful_metrics__: 에폭마다 기록할 때 평균이 아니라 원래의 값을 그대로 유지해야 하는 평가 지표들을 지정하는 인자입니다. 리스트나 튜플, 문자열과 같은 파이썬의 iterable 타입을 입력받습니다. 이 목록에 포함된 지표들은 각 `on_epoch_end` 단계에서 원래의 값 그대로 기록되며, 그밖의 값들은 평균값으로 기록됩니다.


----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L524)</span>
### TerminateOnNaN

```python
keras.callbacks.TerminateOnNaN()
```
손실<sub>loss</sub> 값으로 NaN이 발생했을 때 학습을 종료시키는 콜백입니다.


---
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L537)</span>
### ProgbarLogger

```python
keras.callbacks.ProgbarLogger(count_mode='samples', stateful_metrics=None)
```

평가 지표를 stdout(일반적으로 콘솔창)에 출력하는 콜백입니다.

__인자__

- __count_mode__: `'steps'` 혹은 `'samples'` 중 하나를 입력합니다. 진행 표시줄이 검사한 샘플의 수에 기반하는지 검사한 단계(배치)의 수에 기반하는지 여부를 결정합니다.
- __stateful_metrics__: 에폭마다 기록할 때 평균이 아니라 원래의 값을 그대로 유지해야 하는 평가 지표들을 지정하는 인자입니다. 리스트나 튜플, 문자열과 같은 파이썬의 iterable 타입을 입력받습니다. 이 목록에 포함된 지표들은 각 `on_epoch_end` 단계에서 원래의 값 그대로 기록되며, 그밖의 값들은 평균값으로 기록됩니다.

__오류__

- __ValueError__: 유효하지 않은 `count_mode`의 경우 오류를 알립니다.
    

----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L614)</span>
### History

```python
keras.callbacks.History()
```

`History` 객체에 이벤트를 기록하는 콜백.

이 콜백은 모든 케라스 모델에 자동적으로 적용됩니다. `History` 객체는 모델의 `fit` 메소드를 통해 반환됩니다.
    
    
----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L633)</span>
### ModelCheckpoint

```python
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
```

각 에폭이 끝날 때 모델을 저장합니다.

__인자__

- __filepath__: `str`. 모델 파일을 저장할 경로를 지정합니다. `logs`에 포함된 키와 `on_epoch_end`에서 해당 키에 부여하는 값을 이용해 파일 이름의 형식을 지정할 수 있습니다. 예를 들어, `filepath`가 `weights.{epoch:02d}-{val_loss:.2f}.hdf5`라면 모델 체크포인트는 에폭 횟수와 검증 손실이 들어간 파일 이름으로 저장됩니다. 
- __monitor__: 조건을 지정해서 저장 여부를 정할 때 해당 조건으로 사용할 지표를 지정합니다. 기본값은 `'val_loss'`입니다.
- __verbose__: 진행과정을 출력할지의 여부를 결정합니다. 0 혹은 1의 값을 갖습니다.
- __save_best_only__: `save_best_only=True`인 경우 `monitor`로 지정된 지표를 기준으로 현재까지 가운데 가장 성능이 뛰어난 결과
- __save_weights_only__: 참인 경우 모델의 가중치만 저장되고(`model.save_weights(filepath)`), 아닌 경우 전체 모델이 저장됩니다(`model.save(filepath)`).
- __mode__: `'auto', 'min', 'max'`. `save_best_only=True`를 사용할 때 `monitor`로 지정한 지표가 현재까지 최대일 때 저장할지 최소일 때 저장할지를 결정합니다. 예를 들어 `val_acc`의 경우 `mode`는 `'max'`가 되어야 하며, `val_loss`라면 `'min'`이 되어야 합니다. `auto`의 경우 지표의 이름에서 방향성을 자동으로 유추합니다.
- __period__: 체크포인트 사이의 간격을 에폭 수를 기준으로 지정합니다.


----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L733)</span>
### EarlyStopping

```python
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
```

지표가 나아지지 않는 경우 학습을 멈추도록 합니다.

__인자__

- __monitor__: 조건으로 사용할 지표를 지정합니다. 기본값은 `'val_loss'`입니다.
- __min_delta__: 관찰하는 수량이 향상되었다고 판단할 최소한의 변화. 다시 말해 변화량이 min_delta 보다 작을 경우 향상된 것으로 간주하지 않습니다.
- __patience__: 지표가 나아지지 않는 상태에서 추가적으로 학습할 최대 횟수를 지정합니다. 지정한 횟수에 도달할 때까지 지표의 개선이 없으면 학습을 멈춥니다. 검증 지표를 조건으로 사용할 때 검증 빈도가 1보다 큰 경우 (예: `model.fit(validation_freq=5)`) 매 에폭마다 검증 지표가 생성되지 않는다는 점에 유의하기 바랍니다.
- __verbose__: 진행과정을 출력할지의 여부를 결정합니다. 0 혹은 1의 값을 갖습니다.
- __mode__: `'auto', 'min', 'max'`. `save_best_only=True`를 사용할 때 `monitor`로 지정한 지표가 현재까지 최대일 때 저장할지 최소일 때 저장할지를 결정합니다. 예를 들어 `val_acc`의 경우 `mode`는 `'max'`가 되어야 하며, `val_loss`라면 `'min'`이 되어야 합니다. `auto`의 경우 지표의 이름에서 방향성을 자동으로 유추합니다.
- __baseline__: 관찰하는 수량이 도달해야 하는 최저 기준 값. 모델이 지정한 값 이상의 향상을 이루지 못하는 경우 학습을 멈춥니다.
- __restore_best_weights__: 학습을 끝낼 때 가장 좋은 결과를 기록한 에폭의 가중치를 복원할지 여부를 정합니다. 기본값은 `'False'`이며, 이 경우 가장 마지막 학습 단계의 가중치를 반환합니다.


----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L851)</span>
### RemoteMonitor

```python
keras.callbacks.RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None, send_as_json=False)
```

이벤트를 서버에 스트림할 콜백

`requests` 라이브러리를 필요로 합니다.
디폴트 설정으로 이벤트는 `root + '/publish/epoch/end/'`으로 보내집니다.
호출은 HTTP POST와 JSON-인코딩 이벤트 데이터 딕셔너리인
`data` 인자로 합니다.
send_as_json이 참으로 설정된 경우, 요청의 내용 종류는
application/json입니다. 그 외는 형식 내 직렬화된 JSON이 보내집니다.

__인자__

- __root__: 문자열; 표적 서버의 최상위 url.
- __path__: 문자열; 이벤트가 보내질 `root`를 기준으로 한 상대적 경로
- __field__: 문자열; 데이터가 저장될 JSON 필드.
    형식 내 payload가 보내지는 경우에만 필드가 사용됩니다
    (다시 말해 send_as_json이 거짓으로 설정된 경우).
- __headers__: 딕셔너리; 선택적 커스텀 HTTP 헤더
- __send_as_json__: 불리언; 요청을 application/json으로
    보내는지 여부.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L910)</span>
### LearningRateScheduler

```python
keras.callbacks.LearningRateScheduler(schedule, verbose=0)
```

학습 속도 스케쥴러.

__인자__

- __schedule__: 세대 색인(정수, 0에서 색인 시작)과 현재
    학습 속도를 인풋으로 받고, 새로운 학습 속도를
    아웃풋(부정소수점)으로 반환하는 함수.
- __verbose__: 정수. 0: 자동, 1: 최신화 메시지.


----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L946)</span>
### ReduceLROnPlateau

```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
```

측정 항목이 향상되지 않는 경우 학습 속도를 줄입니다.

모델은 종종 학습이 부진할 때 학습 속도를 2-10의 인자로 줄여
효과를 보기도 합니다. 이 콜백은 수량을 관찰하여
'patience' 수의 세대 동안 향상되는 기미가 없으면
'patience' 수의 세대 동안 향상되는 기미가 없으면

__예시__


```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(X_train, Y_train, callbacks=[reduce_lr])
```

__인자__

- __monitor__: 관찰할 수량.
- __factor__: 학습 속도를 줄일 인자.
    new_lr = lr * factor
- __patience__: number of epochs that produced the monitored
    quantity with no improvement after which training will
    be stopped.
    Validation quantities may not be produced for every
    epoch, if the validation frequency
    (`model.fit(validation_freq=5)`) is greater than one.
- __verbose__: 정수. 0: 자동, 1: 최신화 메시지.
- __mode__: {auto, min, max} 중 하나. `min` 모드에서는
    관찰하는 수량이 더 이상 감소하지
    않으면 학습이 멈춥니다. `max` 모드에서는
    관찰하는 수량이 더 이상 증가하지
    않으면 학습이 멈춥니다; `auto` 모드에서는
    관찰하는 수량의 이름에서
    방향성이 자동적으로 유추됩니다.
- __min_delta__: 유의미한 변화에만 집중해 새로운
    최적값을 계산할 한계점.
- __cooldown__: 학습 속도가 감소 된 후 몇 세대를 기다려야 평소
    가동이 재개되는지를 결정하는 세대의 수.
- __min_lr__: 학습 속도에 대한 하한선.
    

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L1071)</span>
### CSVLogger

```python
keras.callbacks.CSVLogger(filename, separator=',', append=False)
```

세대 결과를 csv 파일에 스트림하는 콜백.

np.ndarray와 같은 1D interable을 포함,
문자열로 표현 가능한 모든 값을 지원합니다.

__예시__


```python
csv_logger = CSVLogger('training.log')
model.fit(X_train, Y_train, callbacks=[csv_logger])
```

__인자__

- __filename__: csv 파일의 파일 이름, 예. 'run/log.csv'.
- __separator__: csv 파일의 성분을 분리하는데 사용할 문자열.
- __append__: 참: 파일이 존재하는 경우 뒤에 덧붙입니다 (학습을 이어나갈 때
    유용합니다). 거짓: 기존의 파일을 덧씌웁니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L1159)</span>
### LambdaCallback

```python
keras.callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
```

간단한 커스텀 콜백을 즉석에서 만드는 콜백.

이 콜백은 적절한 시점에 호출될 익명 함수로 생성됩니다.
콜백이 다음과 같이 위치적 인자를 전달 받는다는 것을
참고하십시오:

- `on_epoch_begin`과 `on_epoch_end`는 다음 두 가지 위치적 인자를 전달 받습니다:
`epoch`, `logs`
- `on_batch_begin`과 `on_batch_end`는 다음 두 가지 위치적 인자를 전달 받습니다:
`batch`, `logs`
- `on_train_begin`과 `on_train_end`는 다음 위치적 인자를 전달 받습니다:
`logs`

__인자__

- __on_epoch_begin__: 각 세대의 시작에 호출됩니다.
- __on_epoch_end__: 각 세대의 끝에 호출됩니다.
- __on_batch_begin__: 각 배치의 시작에 호출됩니다.
- __on_batch_end__: 각 배치의 끝에 호출됩니다.
- __on_train_begin__: 모델 학습의 시작에 호출됩니다.
- __on_train_end__: 모델 학습의 끝에 호출됩니다.

__예시__


```python
# 각 배치가 시작할 때 배치 번호를 프린트합니다
batch_print_callback = LambdaCallback(
    on_batch_begin=lambda batch,logs: print(batch))

# 세대 손실을 JSON 형식으로 파일에 스트림합니다
# 파일 내용은 명확히 구성된 JSON이 아니며 줄 별로 JSON 객체를 가집니다
import json
json_log = open('loss_log.json', mode='wt', buffering=1)
json_logging_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
    on_train_end=lambda logs: json_log.close()
)

# 모델 학습을 끝낸 후 몇몇 처리 과정을 종료합니다
processes = ...
cleanup_callback = LambdaCallback(
    on_train_end=lambda logs: [
        p.terminate() for p in processes if p.is_alive()])

model.fit(...,
          callbacks=[batch_print_callback,
                     json_logging_callback,
                     cleanup_callback])
```





    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks/tensorboard_v1.py#L20)</span>
### TensorBoard

```python
keras.callbacks.tensorboard_v1.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
```

TensorBoard 기초 시각화.

[TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)는
텐서플로우가 제공하는 시각화 도구입니다.

이 콜백은 TensorBoard에 로그를 기록하여
학습과 테스트 측정 항목에 대한 동적 그래프나
모델 내 다양한 레이어에 대한 활성화 히스토그램을
시각화 할 수 있도록 합니다.

pip으로 텐서플로우를 설치했다면, 다음과 같이
명령 줄에서 TensorBoard를 시작할 수 있습니다:
```sh
tensorboard --logdir=/full_path_to_your_logs
```

텐서플로우 외의 백엔드를 사용하는 경우에도
(텐서플로우가 설치만 되어있다면), TensorBoard가 동작하기는 하지만,
손실과 측정 항목 플롯을 보여주는 기능만 사용이 가능합니다.

__인자__

- __log_dir__: TensorBoard가 구문 분석할 로그 파일을
    저장할 위치 경로.
- __histogram_freq__: 모델의 레이어에 대해 활성화와 가중치 히스토그램을 계산할
    (세대 내) 빈도. 0으로 설정되어 있으면 히스토그램을 계산하지
    않습니다. 히스토그램 시각화를 하려면 검증 데이터(혹은 데이터 조각)이
    특정되어야 합니다.
- __batch_size__: 히스토그램 계산 목적으로 네트워크에 전달할
    인풋 배치의 크기.
- __write_graph__: TensorBoard에서 그래프를 시각화할지 여부.
    write_graph이 참으로 설정되어 있으면
    로그 파일이 상당히 커질 수 있습니다.
- __write_grads__: TensorBoard에서 경사 히스토그램를 시각화할지 여부.
    `histogram_freq`이 0보다 커야 합니다.
- __write_images__: TensorBoard에서 이미지로 시각화할 모델 가중치를
    작성할지 여부.
- __embeddings_freq__: 선택된 임베딩 레이어가 저장되는 (세대 내) 빈도.
    0으로 설정되어 있으면, 임베딩이 계산되지 않습니다.
    TensorBoard의 Embedding 탭에서 시각화할 데이터는
    `embeddings_data`로 전달되어야 합니다.
- __embeddings_layer_names__: 관찰할 레이어 이름의 리스트.
    None이나 빈 리스트의 경우 모든 임베딩 레이어가 관찰됩니다.
- __embeddings_metadata__: 레이어 이름을 해당 임베딩 레이어의 메타데이터가 저장되는
    파일 이름에 매핑하는 딕셔너리. 메타데이터 파일 형식에 대해서는
    [세부사항](https://www.tensorflow.org/guide/embedding#metadata)을
    참고하십시오. 모든 임베딩 레이어에 대해서 동일한 메타데이터
    파일이 사용되는 경우는 문자열을 전달할 수 있습니다.
- __embeddings_data__: `embeddings_layer_names`에 특정되어 레이어에 
    임베딩할 데이터. Numpy 배열 (모델이 하나의 인풋을 갖는 경우) 혹은
    Numpy 배열의 리스트 (모델이 여러 인풋을 갖는 경우).
    임베딩에 대해서 [더 알아보려면](
    https://www.tensorflow.org/guide/embedding).
- __update_freq__: `'batch'`, `'epoch'`, 혹은 정수. `'batch'`를 사용하는 경우
    각 배치 이후 손실과 특정항목을 TensorBoard에 기록합니다.
    `'epoch'`의 경우에도 마찬가지입니다. 정수를 사용하는 경우,
    예를 들어 `10000`이라면, 10000 샘플마다 콜백이 측정 항목과 손실을
    TensorBoard에 작성합니다. TensorBoard에 너무 자주 기록하면
    학습이 느려질 수도 있다는 점을 참고하십시오.
    

    

---


# 콜백을 만드십시오

베이스 클래스인 `keras.callbacks.Callback`를 확장해서 커스텀 콜백을 만들 수 있습니다. 콜백은 클래스 재산인 `self.model`을 통해서 관련 모델에 접근할 수 있습니다.

다음은 학습 과정 중 각 배치에 대한 손실 리스트를 저장하는 간단한 예시입니다:
```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
```

---

### 예시: 손실 역사 기록

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
# 아웃풋
'''
[0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
'''
```

---

### 예시: 모델 체크포인트

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
