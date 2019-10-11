# Sequential 모델 API

Sequential 모델을 처음 사용하는 경우, 먼저 [케라스 Sequential 모델 시작하기](/getting-started/sequential-model-guide)를 읽어보십시오.

----
## Sequential 모델 메소드

### compile
```python
compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
```
학습시킬 모델을 구성합니다.

__인자<sub>Argument</sub>__
- __optimizer__: 모형에 사용할 최적화 함수<sub>Optimizer</sub>를 지정합니다. 케라스가 제공하는 최적화 함수의 이름 문자열<sub>String</sub> 또는 개별 최적화 함수의 인스턴스를 입력합니다. 자세한 사항은 [최적화 함수](/optimizers)를 참고하십시오.
- __loss__: 모형에 사용할 손실 함수<sub>Loss function</sub>를 지정합니다. 케라스가 제공하는 손실 함수의 이름 문자열 또는 개별 손실 함수의 인스턴스를 입력합니다. 자세한 사항은 [손실 함수](/losses)을 참고 하십시오. 모델이 다중의 결과<sub>Output</sub>를 출력하는 경우, 손실 함수를 리스트 또는 딕셔너리 형태로 입력하여 결과의 종류별로 서로 다른 손실 함수를 적용할 수 있습니다. 이 경우 모델이 최소화할 손실값은 모든 손실 함수별 결과값의 합이 됩니다. 
- __metrics__: 학습 및 평가 과정에서 사용할 평가 지표<sub>Metric</sub>의 리스트입니다. 가장 빈번히 사용하는 지표는 정확도`metrics=['accuracy']`입니다. 모델이 다중의 결과를 출력하는 경우, 각 결과에 서로 다른 지표를 특정하려면 `metrics={'output_a': 'accuracy'}`와 같은 형식의 딕셔너리를 입력합니다. 또한 출력 결과 개수와 동일한 길이의 리스트 `len = len(outputs)`형식으로도 지정할 수 있습니다 (예: `metrics=[['accuracy'], ['accuracy', 'mse']]` 혹은 `metrics=['accuracy', ['accuracy', 'mse']]`).
- __loss_weights__: 모델이 다중의 결과를 출력하는 경우 각각의 손실값이 전체 손실값에 미치는 영향을 조정하기 위한 계수를 설정합니다. 모델이 최소화할 손실값은 각각 `loss_weights` 만큼의 가중치가 곱해진 개별 손실값의 합입니다. `float`형식의 스칼라 값으로 이루어진 리스트 또는 딕셔너리를 입력받습니다. 리스트일 경우 모델의 결과와 순서에 맞게 1:1로 나열되어야 하며, 딕셔너리의 경우 결과값의 문자열 이름을 `key`로, 적용할 가중치의 스칼라 값을 `value`로 지정해야 합니다.
- __sample_weight_mode__: `fit` 메소드의 `sample_weight`인자는 모델의 학습과정에서 손실을 계산할 때 입력된 훈련<sub>Train</sub> 세트의 표본<sub>Sample</sub>들 각각에 별도의 가중치를 부여하는 역할을 합니다. 이때 부여할 가중치가 2D(시계열) 형태라면 `compile`메소드의 `sample_weight_mode`를 `"temporal"`로 설정해야 합니다. 기본값은 1D 형태인 `None`입니다. 모델이 다중의 결과를 출력하는 경우 딕셔너리 혹은 리스트의 형태로 입력하여 각 결과마다 서로 다른 `sample_weight_mode`를 적용할 수 있습니다. 
- __weighted_metrics__: 학습 및 시험<sub>Test</sub> 과정에서 `sample_weight`나 `class_weight`를 적용할 평가 지표들을 리스트 형식으로 지정합니다. `sample_weight`와 `class_weight`에 대해서는 `fit`메소드의 인자 항목을 참고하십시오.
- __target_tensors__: 기본적으로 케라스는 학습 과정에서 외부로부터 목표값<sub>Traget</sub>을 입력받아 저장할 공간인 플레이스홀더<sub>Placeholder</sub>를 미리 생성합니다. 만약 플레이스홀더를 사용하는 대신 특정한 텐서를 목표값으로 사용하고자 한다면 `target_tensors`인자를 통해 직접 지정하면 됩니다. 이렇게 하면 학습과정에서 외부로부터 NumPy 데이터를 목표값으로 입력받지 않게 됩니다. 단일 결과 모델일 경우 하나의 텐서를, 다중 모델의 경우는 텐서의 리스트 또는 결과값의 문자열 이름을 `key`로, 텐서를 'value'로 지정한 딕셔너리를 `target_tensors`로 입력받습니다.
- __**kwargs__: Theano/CNTK 백엔드를 사용하는 경우 이 인자는 `K.function`에 전달됩니다. 텐서플로우 백엔드를 사용하는 경우, 인자가 `tf.Session.run`에 전달됩니다.

__오류__
- __ValueError__: `optimizer`, `loss`, `metrics` 혹은 `sample_weight_mode`의 인자가 잘못된 경우 오류가 발생합니다.


----
### fit
```python
fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
```
전체 데이터셋을 정해진 횟수만큼 반복하여 모델을 학습시킵니다.

__인자__
- __x__: 입력 데이터로 다음과 같은 형태가 가능합니다.
    - NumPy 배열<sub>array</sub> 또는 배열과 같은 형식의 데이터. 다중 입력의 경우 배열의 리스트.
    - 모델이 이름이 지정된 입력값을 받는 경우 이름과 배열/텐서가 `key`와 `value`로 연결된 딕셔너리.
    - 파이썬 제너레이터 또는 `keras.utils.Sequence` 인스턴스로 `(inputs, targets)` 혹은 `(inputs, targets, sample weights)`를 반환하는 것.
    - 프레임워크(예: TensorFlow)를 통해 이미 정의된 텐서를 입력받는 경우 `None`(기본값).
- __y__: 목표 데이터로 다음과 같은 형태가 가능합니다.
    - NumPy 배열 또는 배열과 같은 형식의 데이터. 다중의 결과를 출력하는 경우 배열의 리스트.
    - 프레임워크(예: TensorFlow)를 통해 이미 정의된 텐서를 입력받는 경우 `None`(기본값).
    - 결과값의 이름이 지정되어 있는 경우 이름과 배열/텐서가 `key`와 `value`로 연결된 딕셔너리.
    - 만약 `x`에서 파이썬 제너레이터 또는 `keras.utils.Sequence` 인스턴스를 사용할 경우 목표값이 `x`와 함께 입력되므로 별도의 `y`입력은 불필요합니다.
- __batch_size__: `int` 혹은 `None`. 역전파<sub>Backpropagation</sub> 과정에서 한 번에 그래디언트가 적용될 표본의 개수입니다. 따로 정하지 않는 경우 `batch_size`는 기본값인 32가 됩니다. 별도의 심볼릭 텐서나 제네레이터, 혹은 `Sequence` 인스턴스로 데이터를 받는 경우 인스턴스가 자동으로 배치를 생성하기 때문에 별도의 `batch_size`를 지정하지 않습니다. 
- __epochs__: `int`. 모델에 데이터 세트를 학습시킬 횟수입니다. 한 번의 에폭은 훈련 데이터로 주어진 모든 `x`와 `y`를 각 1회씩 학습시키는 것을 뜻합니다. 횟수의 인덱스가 `epochs`로 주어진 값에 도달할 때까지 학습이 반복되도록 되어있기 때문에 만약 시작 회차를 지정하는 `initial_epoch` 인자와 같이 쓰이는 경우 `epochs`는 전체 횟수가 아닌 "마지막 회차"의 순번을 뜻하게 됩니다. 
- __verbose__: `int`. `0`, `1`, 혹은 `2`. 학습 중 진행 정보의 화면 출력 여부를 설정하는 인자입니다. `0`은 표시 없음, `1`은 진행 표시줄<sub>Progress bar</sub> 출력, `2`는 에폭당 한 줄씩 출력을 뜻합니다. 기본값은 `1`입니다. 
- __callbacks__: `keras.callbacks.Callback` 인스턴스의 리스트. 학습과 검증 과정에서 적용할 콜백의 리스트입니다. 자세한 사항은 [콜백](/callbacks)을 참조하십시오.
- __validation_split__: 0과 1사이의 `float`. 입력한 `x`와 `y` 훈련 데이터의 마지막부터 지정된 비율만큼의 표본을 분리하여 검증<sub>Validation</sub> 데이터를 만듭니다. 이 과정은 데이터를 뒤섞기 전에 실행됩니다. 검증 데이터는 학습에 사용되지 않으며 각 에폭이 끝날 때마다 검증 손실과 평가 지표를 구하는데 사용됩니다. `x`가 제너레이터 혹은 `Sequence`인 경우에는 지원되지 않습니다. 
- __validation_data__: 매 에폭이 끝날 때마다 손실 및 평가지표를 측정할 검증 데이터를 지정합니다. 검증 데이터는 오직 측정에만 활용되며 학습에는 사용되지 않습니다. `validation_split`인자와 같이 지정될 경우 `validation_split`인자를 무시하고 적용됩니다. `validation_data`로는 다음과 같은 형태가 가능합니다.
    - NumPy 배열 또는 텐서로 이루어진 `(x_val, y_val)` 튜플.
    - NumPy 배열로 이루어진 `(x_val, y_val, val_sample_weights)` 튜플.
    - 위와 같이 튜플을 입력하는 경우에는 `batch_size`도 같이 명시해야 합니다.
    - 데이터 세트 또는 데이터 세트의 이터레이터<sub>Iterator</sub>. 이 경우 `validation_steps`를 같이 명시해야 합니다.
- __shuffle__: 불리언 또는 문자열 `'batch'`. 불리언 입력의 경우 각 에폭을 시작하기 전에 훈련 데이터를 뒤섞을지를 결정합니다. `'batch'`의 경우 HDF5 데이터의 제약을 해결하기 위한 설정으로 각 배치 크기 안에서 데이터를 뒤섞습니다. `steps_per_epoch`값이 `None`이 아닌 경우 `shuffle`인자는 무효화됩니다.
- __class_weight__: 필요한 경우에만 사용하는 인자로, 각 클래스 인덱스를 `key`로, 가중치를 `value`로 갖는 딕셔너리를 입력합니다. 인덱스는 정수, 가중치는 부동소수점 값을 갖습니다. 주로 훈련 데이터의 클래스 분포가 불균형할 때 이를 완화하기 위해 사용하며, 손실 계산 과정에서 표본 수가 더 적은 클래스의 영향을 가중치를 통해 끌어올립니다. 학습 과정에서만 사용됩니다.
- __sample_weight__: 특정한 훈련 표본이 손실 함수에서 더 큰 영향을 주도록 하고자 할 경우에 사용하는 인자입니다. 일반적으로 표본과 동일한 길이의 1D NumPy 배열을, 시계열 데이터의 경우 `(samples, sequence_length)`로 이루어진 2D 배열을 입력하여 손실 가중치와 표본이 1:1로 짝지어지게끔 합니다. 2D 배열을 입력하는 경우 반드시 `compile()` 단계에서 `sample_weight_mode="temporal"`로 지정해야 합니다. `x`가 제너레이터 또는 `Sequence`인스턴스인 경우는 손실 가중치를 `sample_weight`인자 대신 `x`의 세 번째 구성요소로 입력해야 적용됩니다.
- __initial_epoch__: `int`. 특정한 에폭에서 학습을 시작하도록 시작 회차를 지정합니다. 이전의 학습을 이어서 할 때 유용합니다.
- __steps_per_epoch__: `int` 혹은 `None`. 1회의 에폭을 이루는 배치의 개수를 정의하며 기본값은 `None`입니다. 프레임워크에서 이미 정의된 텐서(예: TensorFlow data tensor)를 훈련 데이터로 사용할 때 `None`을 지정하면 자동으로 데이터의 표본 수를 배치 크기로 나누어 올림한 값을 갖게 되며, 값을 구할 수 없는 경우 `1`이 됩니다.
- __validation_steps__: 다음의 두 가지 경우에 한해서 유효한 인자입니다.
    - `steps_per_epoch`값이 특정된 경우, 매 에폭의 검증에 사용될 배치의 개수를 특정합니다. 
    - `validation_data`를 사용하며 이에 제너레이터 형식의 값을 입력할 경우, 매 에폭의 검증에 사용하기 위해 제너레이터로부터 생성할 배치의 개수를 특정합니다. 
- __validation_freq__: 검증 데이터가 있을 경우에 한해 유효한 인자입니다. 정수 또는 리스트/튜플/세트 타입의 입력을 받습니다. 정수 입력의 경우 몇 회의 에폭마다 1회 검증할지를 정합니다. 예컨대 `validation_freq=2`의 경우 매 2회 에폭마다 1회 검증합니다. 만약 리스트나 튜플, 세트 형태로 입력된 경우 입력값 안에 지정된 회차에 한해 검증을 실행합니다. 예를 들어 `validation_freq=[1, 2, 10]`의 경우 1, 2, 10번째 에폭에서 검증합니다.
- __max_queue_size__: `int`. 입력 데이터가 제너레이터 또는 `keras.utils.Sequence` 인스턴스일 때만 유효합니다. 제너레이터 대기열<sub>Queue</sub>의 최대 크기를 지정하며, 미정인 경우 기본값 `10`이 적용됩니다.
- __workers__: `int`. 입력 데이터가 제너레이터 또는 `keras.utils.Sequence` 인스턴스일 때만 유효합니다. 프로세스 기반으로 다중 스레딩을 할 때 제너레이터 작동에 사용할 프로세스의 최대 개수를 설정합니다. 기본값은 `1`이며, `0`을 입력할 경우 메인 스레드에서 제너레이터를 작동시킵니다. 
- __use_multiprocessing__: 불리언. 입력 데이터가 제너레이터 또는 `keras.utils.Sequence` 인스턴스일 때만 유효합니다. `True`인 경우 프로세스 기반 다중 스레딩을 사용하며 기본값은 `False`입니다. 이 설정을 사용할 경우 제너레이터에서 객체 직렬화<sub>pickle</sub>가 불가능한 인자들은 사용하지 않도록 합니다 (멀티프로세싱 과정에서 자식 프로세스로 전달되지 않기 때문입니다). 
- __**kwargs__: 이전 버전과의 호환성을 위해 사용됩니다.

__반환값__
`History` 객체를 반환합니다. `History.history` 속성<sub>Attribute</sub>은 각 에폭마다 계산된 학습 손실 및 평가 지표가 순서대로 기록된 값입니다. 검증 데이터를 적용한 경우 해당 손실 및 지표도 함께 기록됩니다.

__오류__
- __RuntimeError__: 모델이 컴파일되지 않은 경우 발생합니다.
- __ValueError__: 모델에 정의된 입력과 실제 입력이 일치하지 않을 경우 발생합니다.

    
----
### evaluate
```python
evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
```
시험 모드에서 모델의 손실 및 평가 지표 값을 구합니다. 계산은 배치 단위로 실행됩니다.

__인자__
- __x__: 입력 데이터로 다음과 같은 형태가 가능합니다.
    - NumPy 배열 또는 배열과 같은 형식의 데이터. 다중 입력의 경우 배열의 리스트.
    - 모델이 이름이 지정된 입력값을 받는 경우 이름과 배열/텐서가 `key`와 `value`로 연결된 딕셔너리.
    - 파이썬 제너레이터 또는 `keras.utils.Sequence` 인스턴스로 `(inputs, targets)` 혹은 `(inputs, targets, sample weights)`를 반환하는 것.
    - 프레임워크(예: TensorFlow)를 통해 이미 정의된 텐서를 입력받는 경우 `None`(기본값).
- __y__: 목표 데이터로 다음과 같은 형태가 가능합니다.
    - NumPy 배열 또는 배열과 같은 형식의 데이터. 다중의 결과를 출력하는 경우 배열의 리스트.
    - 프레임워크(예: TensorFlow)를 통해 이미 정의된 텐서를 입력받는 경우 `None`(기본값).
    - 결과값의 이름이 지정되어 있는 경우 이름과 배열/텐서가 `key`와 `value`로 연결된 딕셔너리.
    - 만약 `x`에서 파이썬 제너레이터 또는 `keras.utils.Sequence` 인스턴스를 사용할 경우 목표값이 `x`와 함께 입력되므로 별도의 `y`입력은 불필요합니다.
- __batch_size__: `int` 혹은 `None`. 한 번에 평가될 표본의 개수입니다. 따로 정하지 않는 경우 `batch_size`는 기본값인 32가 됩니다. 별도의 심볼릭 텐서나 제네레이터, 혹은 `Sequence` 인스턴스로 데이터를 받는 경우 인스턴스가 자동으로 배치를 생성하기 때문에 별도의 `batch_size`를 지정하지 않습니다. 
- __verbose__: `0` 또는 `1`. 진행 정보의 화면 출력 여부를 설정하는 인자입니다. `0`은 표시 없음, `1`은 진행 표시줄 출력을 뜻합니다. 기본값은 `1`입니다. 
- __sample_weight__: 특정한 시험 표본이 손실 함수에서 더 큰 영향을 주도록 하고자 할 경우에 사용하는 인자입니다. 일반적으로 표본과 동일한 길이의 1D NumPy 배열을, 시계열 데이터의 경우 `(samples, sequence_length)`로 이루어진 2D 배열을 입력하여 손실 가중치와 표본이 1:1로 짝지어지게끔 합니다. 2D 배열을 입력하는 경우 반드시 `compile()` 단계에서 `sample_weight_mode="temporal"`로 지정해야 합니다.
- __steps__: `int` 혹은 `None`. 평가를 완료하기까지의 단계(배치) 개수를 정합니다. 기본값은 `None`으로, 이 경우 고려되지 않습니다.
- __callbacks__: `keras.callbacks.Callback` 인스턴스의 리스트. 평가 과정에서 적용할 콜백의 리스트입니다. [콜백](/callbacks)을 참조하십시오.
- __max_queue_size__: `int`. 입력 데이터가 제너레이터 또는 `keras.utils.Sequence` 인스턴스일 때만 유효합니다. 제너레이터 대기열의 최대 크기를 지정하며, 미정인 경우 기본값 `10`이 적용됩니다.
- __workers__: `int`. 입력 데이터가 제너레이터 또는 `keras.utils.Sequence` 인스턴스일 때만 유효합니다. 프로세스 기반으로 다중 스레딩을 할 때 제너레이터 작동에 사용할 프로세스의 최대 개수를 설정합니다. 기본값은 `1`이며, `0`을 입력할 경우 메인 스레드에서 제너레이터를 작동시킵니다. 
- __use_multiprocessing__: 불리언. 입력 데이터가 제너레이터 또는 `keras.utils.Sequence` 인스턴스일 때만 유효합니다. `True`인 경우 프로세스 기반 다중 스레딩을 사용하며 기본값은 `False`입니다. 이 설정을 사용할 경우 제너레이터에서 객체 직렬화가 불가능한 인자들은 사용하지 않도록 합니다 (멀티프로세싱 과정에서 자식 프로세스로 전달되지 않기 때문입니다). 

__반환값__
적용할 모델이 단일한 결과를 출력하며 별도의 평가 지표를 사용하지 않는 경우 시험 손실의 스칼라 값을 생성합니다. 다중의 결과를 출력하는 모델이거나 여러 평가 지표를 사용하는 경우 스칼라 값의 리스트를 생성합니다. `model.metrics_names` 속성은 각 스칼라 결과값에 할당된 이름을 보여줍니다. 

__오류__
- __ValueError__: 잘못된 인자 전달시에 발생합니다.


----
### predict
```python
predict(x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
```
모델에 표본을 입력하여 예측값을 생성합니다. 계산은 배치 단위로 실행됩니다.

__인자__
- __x__: 입력 데이터로 다음과 같은 형태가 가능합니다.
    - NumPy 배열 또는 배열과 같은 형식의 데이터. 다중 입력의 경우 배열의 리스트.
    - 모델이 이름이 지정된 입력값을 받는 경우 이름과 배열/텐서가 `key`와 `value`로 연결된 딕셔너리.
    - 파이썬 제너레이터 또는 `keras.utils.Sequence` 인스턴스로 `(inputs, targets)` 혹은 `(inputs, targets, sample weights)`를 반환하는 것.
    - 프레임워크(예: TensorFlow)를 통해 이미 정의된 텐서를 입력받는 경우 `None`(기본값).
- __batch_size__:  `int` 혹은 `None`. 한 번에 예측될 표본의 개수입니다. 따로 정하지 않는 경우 `batch_size`는 기본값인 32가 됩니다. 별도의 심볼릭 텐서나 제네레이터, 혹은 `Sequence` 인스턴스로 데이터를 받는 경우 인스턴스가 자동으로 배치를 생성하기 때문에 별도의 `batch_size`를 지정하지 않습니다. 
- __verbose__: `0` 또는 `1`. 진행 정보의 화면 출력 여부를 설정하는 인자입니다. `0`은 표시 없음, `1`은 진행 표시줄 출력을 뜻합니다. 기본값은 `1`입니다. 
- __steps__: `int` 혹은 `None`. 예측을 완료하기까지의 단계(배치) 개수를 정합니다. 기본값은 `None`으로, 이 경우 고려되지 않습니다.
- __callbacks__: `keras.callbacks.Callback` 인스턴스의 리스트. 예측 과정에서 적용할 콜백의 리스트입니다. [콜백](/callbacks)을 참조하십시오.
- __max_queue_size__: `int`. 입력 데이터가 제너레이터 또는 `keras.utils.Sequence` 인스턴스일 때만 유효합니다. 제너레이터 대기열의 최대 크기를 지정하며, 미정인 경우 기본값 `10`이 적용됩니다.
- __workers__: `int`. 입력 데이터가 제너레이터 또는 `keras.utils.Sequence` 인스턴스일 때만 유효합니다. 프로세스 기반으로 다중 스레딩을 할 때 제너레이터 작동에 사용할 프로세스의 최대 개수를 설정합니다. 기본값은 `1`이며, `0`을 입력할 경우 메인 스레드에서 제너레이터를 작동시킵니다. 
- __use_multiprocessing__: 불리언. 입력 데이터가 제너레이터 또는 `keras.utils.Sequence` 인스턴스일 때만 유효합니다. `True`인 경우 프로세스 기반 다중 스레딩을 사용하며 기본값은 `False`입니다. 이 설정을 사용할 경우 제너레이터에서 객체 직렬화가 불가능한 인자들은 사용하지 않도록 합니다 (멀티프로세싱 과정에서 자식 프로세스로 전달되지 않기 때문입니다). 

__반환값__
예측값의 NumPy 배열.

__오류__
- __ValueError__: 모델에 정의된 입력과 실제 입력이 일치하지 않을 경우, 또는 상태 저장 모델<sub>stateful model</sub>을 사용하여 예측할 때 입력한 표본의 개수가 배치 크기의 배수가 아닌 경우 발생합니다.


----
### train_on_batch
```python
train_on_batch(x, y, sample_weight=None, class_weight=None, reset_metrics=True)
```
하나의 데이터 배치에 대해서 그래디언트를 한 번 적용합니다.

__인자__
- __x__: 기본적으로 훈련 데이터의 NumPy 배열을, 모델이 다중 입력을 받는 경우 NumPy 배열의 리스트를 입력합니다. 모델의 모든 입력에 이름이 배정된 경우 입력값 이름을 `key`로, NumPy 배열을 `value`로 묶은 딕셔너리를 입력할 수 있습니다.
- __y__: 기본적으로 목표 데이터의 NumPy 배열을, 모델이 다중의 결과를 출력하는 경우 NumPy 배열의 리스트를 입력합니다. 모델의 모든 결과에 이름이 배정된 경우 결과값 이름을 `key`로, NumPy 배열을 `value`로 묶은 딕셔너리를 입력할 수 있습니다.
- __sample_weight__: 특정한 훈련 표본이 손실 함수에서 더 큰 영향을 주도록 하고자 할 경우에 사용하는 인자입니다. 일반적으로 표본과 동일한 길이의 1D NumPy 배열을, 시계열 데이터의 경우 `(samples, sequence_length)`로 이루어진 2D 배열을 입력하여 손실 가중치와 표본이 1:1로 짝지어지게끔 합니다. 2D 배열을 입력하는 경우 반드시 `compile()` 단계에서 `sample_weight_mode="temporal"`로 지정해야 합니다. `x`가 제너레이터 또는 `Sequence`인스턴스인 경우는 손실 가중치를 `sample_weight`인자 대신 `x`의 세 번째 구성요소로 입력해야 적용됩니다.
- __class_weight__: 필요한 경우에만 사용하는 인자로, 각 클래스 인덱스를 `key`로, 가중치를 `value`로 갖는 딕셔너리를 입력합니다. 인덱스는 정수, 가중치는 부동소수점 값을 갖습니다. 주로 훈련 데이터의 클래스 분포가 불균형할 때 이를 완화하기 위해 사용하며, 손실 계산 과정에서 표본 수가 더 적은 클래스의 영향을 가중치를 통해 끌어올립니다. 학습 과정에서만 사용됩니다.
- __reset_metrics__: `True`인 경우 오직 해당되는 하나의 배치만을 고려한 평가 지표가 생성됩니다. `False`인 경우 평가 지표는 이후에 입력될 다른 배치들까지 누적됩니다. 

__반환값__
적용할 모델이 단일한 결과를 출력하며 별도의 평가 지표를 사용하지 않는 경우 훈련 손실의 스칼라 값을 생성합니다. 다중의 결과를 출력하는 모델이거나 여러 평가 지표를 사용하는 경우 스칼라 값의 리스트를 생성합니다. `model.metrics_names` 속성은 각 스칼라 결과값에 할당된 이름을 보여줍니다. 


----
### test_on_batch
```python
test_on_batch(x, y, sample_weight=None, reset_metrics=True)
```
하나의 표본 배치에 대해서 모델을 테스트합니다.

__인자__
- __x__: 기본적으로 훈련 데이터의 NumPy 배열을, 모델이 다중 입력을 받는 경우 NumPy 배열의 리스트를 입력합니다. 모델의 모든 입력에 이름이 배정된 경우 입력값 이름을 `key`로, NumPy 배열을 `value`로 묶은 딕셔너리를 입력할 수 있습니다.
- __y__: 기본적으로 목표 데이터의 NumPy 배열을, 모델이 다중의 결과를 출력하는 경우 NumPy 배열의 리스트를 입력합니다. 모델의 모든 결과에 이름이 배정된 경우 결과값 이름을 `key`로, NumPy 배열을 `value`로 묶은 딕셔너리를 입력할 수 있습니다.
- __sample_weight__: 특정한 훈련 표본이 손실 함수에서 더 큰 영향을 주도록 하고자 할 경우에 사용하는 인자입니다. 일반적으로 표본과 동일한 길이의 1D NumPy 배열을, 시계열 데이터의 경우 `(samples, sequence_length)`로 이루어진 2D 배열을 입력하여 손실 가중치와 표본이 1:1로 짝지어지게끔 합니다. 2D 배열을 입력하는 경우 반드시 `compile()` 단계에서 `sample_weight_mode="temporal"`로 지정해야 합니다. `x`가 제너레이터 또는 `Sequence`인스턴스인 경우는 손실 가중치를 `sample_weight`인자 대신 `x`의 세 번째 구성요소로 입력해야 적용됩니다.
- __reset_metrics__: `True`인 경우 오직 해당되는 하나의 배치만을 고려한 평가 지표가 생성됩니다. `False`인 경우 평가 지표는 이후에 입력될 다른 배치들까지 누적됩니다. 

__반환값__
적용할 모델이 단일한 결과를 출력하며 별도의 평가 지표를 사용하지 않는 경우 시험 손실의 스칼라 값을 생성합니다. 다중의 결과를 출력하는 모델이거나 여러 평가 지표를 사용하는 경우 스칼라 값의 리스트를 생성합니다. `model.metrics_names` 속성은 각 스칼라 결과값에 할당된 이름을 보여줍니다. 

    
----
### predict_on_batch
```python
predict_on_batch(x)
```
하나의 표본 배치에 대한 예측값을 생성합니다.

__인자__
- __x__: NumPy 배열로 이루어진 입력 표본.

__반환값__
예측값의 NumPy 배열.


----
### fit_generator
```python
fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, validation_freq=1, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
```
파이썬 제너레이터 또는 `Sequence` 인스턴스에서 생성된 데이터의 배치별로 모델을 학습시킵니다.

연산 효율을 높이기 위해 제너레이터는 모델과 병렬로 작동합니다. 예를 들어 GPU로 모델을 학습시키는 동시에 CPU로 실시간 이미지 데이터 증강<sub>Data augmentation</sub>을 처리할 수 있습니다. 또한 `keras.utils.Sequence`를 사용할 경우 표본의 순서가 유지됨은 물론, `use_multiprocessing=True`인 경우에도 매 에폭당 입력이 한 번만 계산되도록 보장됩니다.

__인자__
- __generator__: 파이썬 제너레이터 혹은 `Sequence`(`keras.utils.Sequence`)객체<sub>Object</sub>입니다. `Sequence` 객체를 사용하는 경우 멀티프로세싱 적용시 입력 데이터가 중복 사용되는 문제를 피할 수 있습니다. `generator`는 반드시 다음 중 하나의 값을 생성해야 합니다.
    - `(inputs, targets)` 튜플
    - `(inputs, targets, sample_weights)` 튜플.  
  매번 제너레이터가 생성하는 튜플의 값들은 각각 하나의 배치가 됩니다. 따라서 튜플에 포함된 배열들은 모두 같은 길이를 가져야 합니다. 물론 배치가 다른 경우 길이도 달라질 수 있습니다. 예컨대, 전체 표본 수가 배치 크기로 딱 나누어지지 않는 경우 마지막 배치의 길이는 다른 배치들보다 짧은 것과 같습니다. 제너레이터는 주어진 데이터를 무한정 반복 추출하는 것이어야 하며, 모델이 `steps_per_epoch`로 지정된 개수의 배치를 처리할 때 1회의 에폭이 끝나는 것으로 간주됩니다. 
- __steps_per_epoch__: `int`. 1회 에폭을 마치고 다음 에폭을 시작할 때까지 `generator`로부터 생성할 배치의 개수를 지정합니다. 일반적으로 `ceil(num_samples / batch_size)`, 즉 전체 표본의 수를 배치 크기로 나누어 올림한 값과 같아야 합니다. `Sequence` 인스턴스를 입력으로 받는 경우에 한해 `steps_per_epoch`를 지정하지 않으면 자동으로 `len(generator)`값을 취합니다. 
- __epochs__: `int`. 모델에 데이터 세트를 학습시킬 횟수입니다. 한 번의 에폭은 `steps_per_epoch`로 정의된 데이터 전체를 1회 학습시키는 것을 뜻합니다. 횟수의 인덱스가 `epochs`로 주어진 값에 도달할 때까지 학습이 반복되도록 되어있기 때문에 만약 시작 회차를 지정하는 `initial_epoch` 인자와 같이 쓰이는 경우 `epochs`는 전체 횟수가 아닌 "마지막 회차"의 순번을 뜻하게 됩니다. 
- __verbose__: `int`. `0`, `1`, 혹은 `2`. 학습 중 진행 정보의 화면 출력 여부를 설정하는 인자입니다. `0`은 표시 없음, `1`은 진행 표시줄<sub>Progress bar</sub> 출력, `2`는 에폭당 한 줄씩 출력을 뜻합니다. 기본값은 `1`입니다. 
- __callbacks__: `keras.callbacks.Callback` 인스턴스의 리스트. 학습 과정에서 적용할 콜백의 리스트입니다. 자세한 사항은 [콜백](/callbacks)을 참조하십시오.
- __validation_data__: 다음 중 하나의 형태를 취합니다.
    - 제너레이터 또는 검증 데이터용 `Sequence` 객체.
    - `(x_val, y_val)` 튜플.
    - `(x_val, y_val, val_sample_weights)` 튜플.  
  매회의 학습 에폭이 끝날 때마다 입력된 데이터를 사용하여 검증 손실 및 평가 지표를 구합니다. 학습에는 사용되지 않습니다.
- __validation_steps__: `validation_data`가 제너레이터 형식의 값을 입력할 경우에만 유효한 인자입니다. 매 에폭의 검증에 사용하기 위해 제너레이터로부터 생성할 배치의 개수를 특정합니다. 일반적으로 검증 세트의 표본 개수를 배치 크기로 나누어 올림한 값과 같아야 합니다. `Sequence` 인스턴스를 입력으로 받는 경우에 한해 `steps_per_epoch`를 지정하지 않으면 자동으로 `len(validation_data)`값을 취합니다. 
- __validation_freq__: 검증 데이터가 있을 경우에 한해 유효한 인자입니다. 정수 또는 리스트/튜플/세트 타입의 입력을 받습니다. 정수 입력의 경우 몇 회의 에폭마다 1회 검증할지를 정합니다. 예컨대 `validation_freq=2`의 경우 매 2회 에폭마다 1회 검증합니다. 만약 리스트나 튜플, 세트 형태로 입력된 경우 입력값 안에 지정된 회차에 한해 검증을 실행합니다. 예를 들어 `validation_freq=[1, 2, 10]`의 경우 1, 2, 10번째 에폭에서 검증합니다.
- __class_weight__: 필요한 경우에만 사용하는 인자로, 각 클래스 인덱스를 `key`로, 가중치를 `value`로 갖는 딕셔너리를 입력합니다. 인덱스는 정수, 가중치는 부동소수점 값을 갖습니다. 주로 훈련 데이터의 클래스 분포가 불균형할 때 이를 완화하기 위해 사용하며, 손실 계산 과정에서 표본 수가 더 적은 클래스의 영향을 가중치를 통해 끌어올립니다. 학습 과정에서만 사용됩니다.
- __max_queue_size__: `int`. 제너레이터 대기열의 최대 크기를 지정하며, 미정인 경우 기본값 `10`이 적용됩니다.
- __workers__: `int`. 프로세스 기반으로 다중 스레딩을 할 때 제너레이터 작동에 사용할 프로세스의 최대 개수를 설정합니다. 기본값은 `1`이며, `0`을 입력할 경우 메인 스레드에서 제너레이터를 작동시킵니다. 
- __use_multiprocessing__: 불리언. `True`인 경우 프로세스 기반 다중 스레딩을 사용하며 기본값은 `False`입니다. 이 설정을 사용할 경우 제너레이터에서 객체 직렬화가 불가능한 인자들은 사용하지 않도록 합니다 (멀티프로세싱 과정에서 자식 프로세스로 전달되지 않기 때문입니다). 
- __shuffle__: 불리언. `Sequence`인스턴스 입력을 받을 때에만 사용됩니다. 각 에폭을 시작하기 전에 훈련 데이터를 뒤섞을지를 결정합니다. `steps_per_epoch`값이 `None`이 아닌 경우 `shuffle`인자는 무효화됩니다.
- __initial_epoch__: `int`. 특정한 에폭에서 학습을 시작하도록 시작 회차를 지정합니다. 이전의 학습을 이어서 할 때 유용합니다.

__반환값__
`History` 객체를 반환합니다. `History.history` 속성<sub>Attribute</sub>은 각 에폭마다 계산된 학습 손실 및 평가 지표가 순서대로 기록된 값입니다. 검증 데이터를 적용한 경우 해당 손실 및 지표도 함께 기록됩니다.

__오류__
- __ValueError__: 제너레이터가 유효하지 않은 형식의 데이터를 만들어 내는 경우 오류가 발생합니다.

__예시__

```python
def generate_arrays_from_file(path):
    while True:
        with open(path) as f:
            for line in f:
                # 파일의 각 라인으로부터
                # 입력 데이터와 레이블의 NumPy 배열을 만듭니다
                x1, x2, y = process_line(line)
                yield ({'input_1': x1, 'input_2': x2}, {'output': y})

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                    steps_per_epoch=10000, epochs=10)
```
    
    
----
### evaluate_generator
```python
evaluate_generator(generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
```
제너레이터에서 생성한 데이터를 이용하여 모델을 평가합니다. 제너레이터는 `test_on_batch`가 요구하는 것과 동일한 종류의 데이터를 생성하는 것이어야 합니다.

__인자__
- __generator__: 
`(inputs, targets)` 튜플 또는 `(inputs, targets, sample_weights)` 튜플을 생성하는 파이썬 제너레이터 혹은 `Sequence`(`keras.utils.Sequence`)인스턴스입니다. `Sequence`인스턴스를 사용하는 경우 멀티프로세싱 적용시 입력 데이터가 중복 사용되는 문제를 피할 수 있습니다. 
- __steps__: 평가를 마치기 전까지 `generator`로부터 생성할 배치의 개수를 지정합니다. `Sequence` 인스턴스를 입력으로 받는 경우에 한해 `steps`를 지정하지 않으면 자동으로 `len(generator)`값을 취합니다. 
- __callbacks__: `keras.callbacks.Callback` 인스턴스의 리스트. 평가 과정에서 적용할 콜백의 리스트입니다. [콜백](/callbacks)을 참조하십시오.
- __max_queue_size__: `int`. 제너레이터 대기열의 최대 크기를 지정하며, 미정인 경우 기본값 `10`이 적용됩니다.
- __workers__: `int`. 프로세스 기반으로 다중 스레딩을 할 때 제너레이터 작동에 사용할 프로세스의 최대 개수를 설정합니다. 기본값은 `1`이며, `0`을 입력할 경우 메인 스레드에서 제너레이터를 작동시킵니다. 
- __use_multiprocessing__: 불리언. 입력 데이터가 제너레이터 또는 `keras.utils.Sequence` 인스턴스일 때만 유효합니다. `True`인 경우 프로세스 기반 다중 스레딩을 사용하며 기본값은 `False`입니다. 이 설정을 사용할 경우 제너레이터에서 객체 직렬화가 불가능한 인자들은 사용하지 않도록 합니다 (멀티프로세싱 과정에서 자식 프로세스로 전달되지 않기 때문입니다). 
- __verbose__: `0` 또는 `1`. 진행 정보의 화면 출력 여부를 설정하는 인자입니다. `0`은 표시 없음, `1`은 진행 표시줄 출력을 뜻합니다. 기본값은 `1`입니다.

__반환값__
적용할 모델이 단일한 결과를 출력하며 별도의 평가 지표를 사용하지 않는 경우 시험 손실의 스칼라 값을 생성합니다. 다중의 결과를 출력하는 모델이거나 여러 평가 지표를 사용하는 경우 스칼라 값의 리스트를 생성합니다. `model.metrics_names` 속성은 각 스칼라 결과값에 할당된 이름을 보여줍니다. 

__오류__
- __ValueError__: 제너레이터가 유효하지 않은 형식의 데이터를 만들어 내는 경우 오류가 발생합니다.


----
### predict_generator
```python
predict_generator(generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
```
제너레이터에서 생성한 표본 데이터에 대한 예측값을 생성합니다. 제너레이터는 `predict_on_batch`가 요구하는 것과 동일한 종류의 데이터를 생성하는 것이어야 합니다.

__인수__
- __generator__: 표본 데이터의 배치를 생성하는 제너레이터 혹은 `Sequence`(`keras.utils.Sequence`)인스턴스입니다. `Sequence`인스턴스를 사용하는 경우 멀티프로세싱 적용시 입력 데이터가 중복 사용되는 문제를 피할 수 있습니다. 
- __steps__:  예측을 마치기 전까지 `generator`로부터 생성할 배치의 개수를 지정합니다. `Sequence` 인스턴스를 입력으로 받는 경우에 한해 `steps`를 지정하지 않으면 자동으로 `len(generator)`값을 취합니다. 
- __callbacks__: `keras.callbacks.Callback` 인스턴스의 리스트. 예측 과정에서 적용할 콜백의 리스트입니다. [콜백](/callbacks)을 참조하십시오.
- __max_queue_size__: `int`. 제너레이터 대기열의 최대 크기를 지정하며, 미정인 경우 기본값 `10`이 적용됩니다.
- __workers__: `int`. 프로세스 기반으로 다중 스레딩을 할 때 제너레이터 작동에 사용할 프로세스의 최대 개수를 설정합니다. 기본값은 `1`이며, `0`을 입력할 경우 메인 스레드에서 제너레이터를 작동시킵니다. 
- __use_multiprocessing__: 불리언. 입력 데이터가 제너레이터 또는 `keras.utils.Sequence` 인스턴스일 때만 유효합니다. `True`인 경우 프로세스 기반 다중 스레딩을 사용하며 기본값은 `False`입니다. 이 설정을 사용할 경우 제너레이터에서 객체 직렬화가 불가능한 인자들은 사용하지 않도록 합니다 (멀티프로세싱 과정에서 자식 프로세스로 전달되지 않기 때문입니다). 
- __verbose__: `0` 또는 `1`. 진행 정보의 화면 출력 여부를 설정하는 인자입니다. `0`은 표시 없음, `1`은 진행 표시줄 출력을 뜻합니다. 기본값은 `1`입니다.

__반환값__
예측 값의 NumPy 배열.

__오류__
- __ValueError__: 제너레이터가 유효하지 않은 형식의 데이터를 만들어 내는 경우 오류가 발생합니다.


----
### get_layer
```python
get_layer(name=None, index=None)
```
층<sub<Layer</sub>의 (고유한) 이름, 혹은 인덱스를 바탕으로 해당 층을 가져옵니다. `name`과 `index`가 모두 제공되는 경우, `index`가 우선 순위를 갖습니다.  
  
인덱스는 (상향식) 너비 우선 그래프 탐색<sub>(Bottom-up) horizontal graph traversal</sub> 순서를 따릅니다.

__인자__
- __name__: `str`. 층의 이름입니다.
- __index__: `int`. 층의 인덱스입니다.

__반환__
층 인스턴스.

__오류__
- __ValueError__: 층의 이름이나 인덱스가 유효하지 않은 경우 오류가 발생합니다.   
