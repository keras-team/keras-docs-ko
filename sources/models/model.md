# Model 클래스 API

기능적 API에서, 인풋 텐서와 아웃풋 텐서가 주어졌을 때, `Model`을 다음과 같이 인스턴스화 할 수 있습니다:

```python
from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)
```

이 모델은 `a`를 받아 `b`를 계산하는데 필요한 모든 레이어를 포함합니다.

다중 인풋 혹은 다중 아웃풋 모델의 경우 리스트를 사용할 수도 있습니다:

```python
model = Model(inputs=[a1, a2], outputs=[b1, b2, b3])
```

`Model`로 할 수 있는 작업에 대한 보다 자세한 설명은 [케라스 기능적 API 안내서](/getting-started/functional-api-guide)를 참조하십시오.


## Methods

### compile


```python
compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
```


학습을 목적으로 모델을 구성합니다.

__인수__

- __optimizer__: 문자열 (옵티마이저의 이름) 혹은 옵티마이저 인스턴스.
    [옵티마이저](/optimizers)를 참조하십시오.
- __loss__: 문자열 (목적 함수의 이름) 혹은 목적 함수.
    [손실](/losses)을 참조하십시오.
    모델이 다중 아웃풋을 갖는 경우, 손실의 리스트 혹은 손실의
    딕셔너리를 전달하여 각 아웃풋에 각기 다른 손실을 사용할 수 있습니다.
    따라서 모델에 의해 최소화되는 손실 값은 
    모든 개별적 손실의 합이 됩니다.
- __metrics__: 학습과 테스트 과정에서 모델이
    평가할 측정항목의 리스트.
    보통은 `metrics=['accuracy']`를 사용하면 됩니다.
    다중 아웃풋 모델의 각 아웃풋에 각기 다른
    측정항목을 특정하려면, `metrics={'output_a': 'accuracy'}`와
    같은 딕셔너리를 전달할 수도 있습니다.
- __loss_weights__: 각기 다른 모델 아웃풋의 손실 기여도에
    가중치를 부여하는 스칼라 계수(파이썬 부동소수점)를
    특정하는 선택적 리스트 혹은 딕셔너리.
    따라서 모델이 최소화할 손실 값은
    `loss_weights` 계수에  의해 *가중치가 적용된*
    모든 개별 손실의 합이 됩니다.
    리스트의 경우 모델의 아웃풋에 1:1 매핑을
    가져야 하며, 텐서의 경우 아웃풋 이름(문자열)을
    스칼라 계수에 매핑해야 합니다.
- __sample_weight_mode__: 시간 단계별로 샘플 가중치를 주어야 하는 
    경우 (2D 가중치), 이 인수를 `"temporal"`로.
    설정하십시오. 디폴트 값은 `None`으로 (1D) 샘플별 가중치를 적용합니다.
    모델이 다중 아웃풋을 갖는 경우 모디의 리스트
    혹은 모드의 딕셔너리를 전달하여 각 아웃풋에 별도의
    `sample_weight_mode`를 사용할 수 있습니다.
- __weighted_metrics__: 학습 혹은 테스트 과정에서 sample_weight 
    혹은 class_weight로 가중치를 주고 평가할 측정항목의 리스트.
- __target_tensors__: 케라스는 디폴트 설정으로 모델의 표적을 위한
    플레이스 홀더를 만들고, 학습 과정 중 이 플레이스 홀더에 표적 데이터를
    채웁니다. 이러한 디폴트 설정 대신 직접 만든 표적 텐서를
    사용하고 싶다면 (이 경우 케라스는 학습 과정 중 이러한 표적의
    외부 Numpy 데이터를 기대하지 않습니다),
    `target_tensors` 인수를 통해서 그 표적 텐서를 특정할 수 있습니다.
    표적 텐서는 (단일 아웃풋 모델의 경우) 단일 텐서, 텐서 리스트,
    혹은 표적 텐서에 아웃풋 이름을 매핑하는 딕셔너리가 될 수 있습니다.
- __**kwargs__: Theano/CNTK 백엔드를 사용하는 경우, 이 인수는
    `K.function`에 전달됩니다.
    텐서플로우 백엔드를 사용하는 경우,
    이는 `tf.Session.run`에 전달됩니다.

__오류 알림__

- __ValueError__: `optimizer`, `loss`, `metrics` 혹은
    `sample_weight_mode`의 인수가 유효하지 않은 경우 오류를 알립니다.
    
----

### fit


```python
fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
```


정해진 수의 세대에 걸쳐 모델을 학습시킵니다 (데이터셋에 대해 반복).

__인수__

- __x__: (모델이 단일 인풋을 갖는 경우) 학습 데이터의 Numpy 배열,
    혹은 (모델이 다중 인풋을 갖는 경우) Numpy 배열의 리스트.
    모델의 인풋 레이어에 이름이 명명된 경우는 인풋
    이름을 Numpy 배열에 매핑하는 딕셔너리를 전달할 수도 있습니다.
    프레임워크-네이티브 텐서(예. 텐서플로우 데이터 텐서)를
    전달받는 경우 x를 None(디폴트 값)으로 둘 수 있습니다.
- __y__: (모델이 단일 아웃풋을 갖는 경우)
    표적(라벨) 데이터의 Numpy 배열,
    혹은 (모델이 다중 아웃풋을 갖는 경우) Numpy 배열의 리스트.
    모델의 아웃풋 레이어에 이름이 명명된 경우는 아웃풋 이름을
    Numpy 배열에 매핑하는 딕셔너리를 전달할 수도 있습니다.
    프레임워크-네이티브 텐서(예. 텐서플로우 데이터 텐서)를 전달받는 경우
    `y`를 `None`(디폴트 값)으로 둘 수 있습니다.
- __batch_size__: 정수 혹은 `None`.
    경사 업데이트 별 샘플의 수.
    따로 정하지 않으면, `batch_size`는 디폴트 값인 32가 됩니다.
- __epochs__: 정수. 모델을 학습시킬 세대의 수.
    한 세대는 제공된 모든 `x`와 `y` 데이터에
    대한 반복입니다.
    `initial_epoch`이 첫 세대라면,
    `epochs`는 "마지막 세대"로 이해하면 됩니다.
    모델은 `epochs`에 주어진 반복의 수만큼
    학습되는 것이 아니라, 단지 `epochs` 색인의 세대에
    도달할 때까지 학습됩니다.
- __verbose__: 정수. 0, 1, 혹은 2. 다변 모드.
    0 = 자동, 1 = 진행 표시 줄, 2 = 세대 당 한 라인.
- __callbacks__: `keras.callbacks.Callback` 인스턴스의 리스트.
    학습과 검증과정에서 적용할 콜백의 리스트입니다.
    
    [콜백](/callbacks)을 참조하십시오.
- __validation_split__: 0과 1사이 부동소수점.
    검증 데이터로 사용할 학습 데이터의 비율.
    이 비율만큼의 학습 데이터를 따로 두어
    모델이 학습하지 않도록 하며, 각 세대의
    끝에서 이 데이터에 대한 손실과
    모델 측정항목을 평가합니다.
    제공된 `x`와 `y` 데이터를 뒤섞기 전
    끝 부분의 샘플을 골라 검증 데이터를 만듭니다.
- __validation_data__: 각 세대의 끝에서 손실과
    모델 측정항목을 평가할 `(x_val, y_val)` 튜플,
    혹은 `(x_val, y_val, val_sample_weights)` 튜플.
    이 데이터에 대해서는 모델이 학습되지 않습니다.
    `validation_data`는 `validation_split`에 대해 우선 순위를 갖습니다.
- __shuffle__: 불리언 (각 세대를 시작하기 전
    학습 데이터를 뒤섞을지 여부) 혹은 ('batch'에 넣을) 문자열.
    'batch'는 HDF5 데이터의 제한 사항으로 다루기 위한
    특별 옵션으로, 배치 크기 조각 단위로 데이터를 뒤섞습니다.
    `steps_per_epoch`이 `None`이 아닌 경우 아무 효과도 없습니다.
- __class_weight__: 가중치 값(부동소수점)에 클래스 색인(정수)을
    매핑하는 선택적 딕셔너리로, (학습 과정 중에 한해)
    손실 함수에 가중치를 적용하는데 사용됩니다.
    모델이 상대적으로 적은 수의 샘플을 가진
    클래스의 샘플에 “더 주의를 기울이도록”
    하는데 유용합니다.
- __sample_weight__: 학습 샘플에 적용할 가중치의
    선택적 Numpy 배열로, (학습 과정 중에 한해) 손실 함수에
    가중치를 적용하는데 사용됩니다. 인풋 샘플과
    동일한 길이의 플랫 (1D) Numpy 배열(가중치와 샘플간 1:1 매핑)을
    전달할 수 있고, 혹은
    시간적 데이터의 경우
    `(samples, sequence_length)`의 형태를 가진
    2D 배열을 전달하여 모든 샘플의
    모든 시간 단계에 각기 다른 가중치를 적용할 수 있습니다.
    이러한 경우 반드시 `compile()` 내에서
    `sample_weight_mode="temporal"`을 특정해야 합니다.
- __initial_epoch__: 정수.
    학습을 시작하는 세대
    (이전의 학습 과정을 재개하는데 유용합니다).
- __steps_per_epoch__: 정수 혹은 `None`.
    한 세대의 종료를 선언하고 다음 세대를
    시작하기까지 단계(샘플 배치)의 총 개수.
    텐서플로우 데이터 텐서와 같은
    인풋 텐서로 학습하는 경우, 디폴트 값인 `None`은
    데이터셋의 샘플 수를 배치 크기로 나눈 값을 갖거나,
    그런 값이 확정될 수 없는 경우 1의 값이 됩니다.
- __validation_steps__: `steps_per_epoch`가 특정된
    경우에만 유의미합니다. 정지하기 전
    검증할 단계(샘플 배치)의 총 개수.
- __validation_freq__: 검증 데이터가 제공된 경우 유의미합니다. 
    정수형 또는 리스트 / 튜플 / 집합 자료형들이 있습니다. 
    정수형의 경우, 새롭게 검증하기 전에 학습에 필요한 세대 수를 
    지정합니다. 예를들어, `validation_freq=2`는 2세대 마다 
    검증을 수행합니다. 그리고 각 자료형에 대해 검증을 하기위한 
    세대 수가 지정된 경우, `validation_freq=[1, 2, 10]`이 
    첫 번째, 두 번째 및 열 번째 세대의 끝부분에서 검증할 수 
    있습니다.
    

__반환값__

`History` 객체. `History.history` 속성은
연속된 세대에 걸친 학습 손실 값과
측정항목 값, 그리고 (적용 가능한 경우) 검증 손실 값과
검증 측정항목 값의 기록입니다.

__오류 알림__

- __RuntimeError__: 모델이 컴파일되지 않은 경우 오류를 알립니다.
- __ValueError__: 제공된 인풋 데이터가 모델의 예상과 다른 경우
    오류를 알립니다.
    
----

### evaluate


```python
evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None)
```


테스트 모드에서의 모델의 손실 값과 측정항목 값을 반환합니다.

계산은 배치 단위로 실행됩니다.

__인수__

- __x__: (모델이 단일 인풋을 갖는 경우) 테스트 데이터의 Numpy 배열,
    혹은 (모델이 다중 인풋을 갖는 경우) Numpy 배열의 리스트.
    모델의 인풋 레이어에 이름이 명명된 경우는 인풋
    이름을 Numpy 배열에 매핑하는 딕셔너리를 전달할 수도 있습니다.
    프레임워크-네이티브 텐서(예. 텐서플로우 데이터 텐서)를
    전달받는 경우 `x`를 `None`(디폴트 값)으로 둘 수 있습니다.
- __y__: (모델이 단일 아웃풋을 갖는 경우)
    표적(라벨) 데이터의 Numpy 배열,
    혹은 (모델이 다중 아웃풋을 갖는 경우) Numpy 배열의 리스트.
    모델의 아웃풋 레이어에 이름이 명명된 경우는 아웃풋
    이름을 Numpy 배열에 매핑하는 딕셔너리를 전달할 수도 있습니다.
    프레임워크-네이티브 텐서(예. 텐서플로우 데이터 텐서)를
    `y`를 `None`(디폴트 값)으로 둘 수 있습니다.
- __batch_size__: 정수 혹은 `None`.
    경사 업데이트별 샘플의 수.
    따로 정하지 않으면, `batch_size`는 디폴트 값인 32가 됩니다.
- __verbose__: 0 혹은 1. 다변 모드.
    0 = 자동, 1 = 진행 표시줄.
- __sample_weight__: 테스트 샘플에 적용할
    가중치의 선택적 Numpy 배열로, 손실 함수에
    가중치를 적용하는데 사용됩니다. You can either pass a flat (1D)
    인풋 샘플과 동일한 길이의 (1D) Numpy 배열with the same length as the input samples
    (가중치와 샘플 간 1:1 매핑)을 전달할
    수 있고, 혹은 시간적 데이터의
    경우 `(samples, sequence_length)`의
    형태를 가진 2D 배열을 전달하여 모든 샘플의
    모든 시간 단계에 각기 다른 가중치를 적용할 수 있습니다.
    이러한 경우 반드시 `compile()` 내에서
    `sample_weight_mode="temporal"`을 특정해야 합니다.
- __steps__: 정수 혹은 `None`.
    평가가 한 회 완료되었음을
    선언하기까지 단계(샘플 배치)의 총 개수.
    디폴트 값인 `None`의 경우 고려되지 않습니다.
- __callbacks__: `keras.callbacks.Callback` 인스턴스의 리스트.
    평가 과정에서 적용할 콜백의 리스트입니다.
    [콜백](/callbacks)을 참조하십시오.

__반환값__

(모델이 단일 아웃풋만 갖고 측정항목을 보유하지 않는 경우)
스칼라 테스트 손실 혹은 (모델이 다중 아웃풋, 그리고/혹은 측정항목을
갖는 경우) 스칼라 리스트. `model.metrics_names` 속성은
스칼라 아웃풋에 대한 디스플레이 라벨을 제공합니다.
    
----

### predict


```python
predict(x, batch_size=None, verbose=0, steps=None, callbacks=None)
```


인풋 샘플에 대한 아웃풋 예측을 생성합니다.

계산은 배치 단위로 실행됩니다.

__인수__

- __x__: Numpy 배열 (혹은
    (모델이 다중 인풋을 갖는 경우 Numpy 배열의 리스트) 형태의 인풋 데이터.
- __batch_size__: 정수. 따로 정하지 않으면 디폴트 값인 32가 됩니다.
- __verbose__: 다변 모드, 0 혹은 1.
- __steps__: 예측이 한 회 완료되었음을
    선언하기까지 단계(샘플 배치)의 총 개수.
    디폴트 값인 `None`의 경우 고려되지 않습니다.
- __callbacks__: `keras.callbacks.Callback` 인스턴스의 리스트.
    학습과 검증 과정에서 적용할 콜백의 리스트입니다.
    [콜백](/callbacks)을 참조하십시오.

__반환값__

예측 값의 Numpy 배열.

__오류 알림__

- __ValueError__: 제공된 인풋 데이터가 모델의
    예상과 다른 경우 혹은,
    상태 기반 모델이 배치 크기의 배수가 아닌 수의
    샘플을 받은 경우 오류를 알립니다.
    
----

### train_on_batch


```python
train_on_batch(x, y, sample_weight=None, class_weight=None)
```


하나의 데이터 배치에 대해서 경사 업데이트를 1회 실시합니다.

__인수__

- __x__: 학습 데이터의 Numpy 배열, 혹은
    모델이 다중 인풋을 갖는 경우 Numpy 배열의 리스트.
    모델의 모든 인풋에 이름이 명명된 경우
    인풋 이름을 Numpy 배열에 매핑하는
    딕셔너리를 전달할 수도 있습니다.
- __y__: 표적 데이터의 Numpy 배열,
    혹은 모델이 다중 아웃풋을 갖는 경우 Numpy 배열의 리스트.
    모델의 모든 아웃풋에 이름이 명명된 경우
    아웃풋 이름을 Numpy 배열에 매핑하는
    딕셔너리를 전달할 수도 있습니다.
- __sample_weight__: x와 동일한 길이의 선택적 배열로,
    각 샘플에 대한 모델의 손실에 적용할 가중치를 담습니다.
    시간적 데이터의 경우, (samples, sequence_length)의
    형태를 가진 2D 배열을 전달하여 모든 샘플의
    모든 시간 단계에 대해 각기 다른 가중치를 적용할 수 있습니다.
    이러한 경우 반드시 `compile()` 내에서
    `sample_weight_mode="temporal"`을 특정해야 합니다.
- __class_weight__: 가중치 값(부동소수점)에 클래스 색인(정수)을
    매핑하는 선택적 딕셔너리로, 학습 과정 중
    모델의 손실을 각 클래스의 샘플에 적용하는데 사용됩니다.
    이는 모델이 상대적으로 적은 수의 샘플을 가진
    클래스의 샘플에 "더 주의를 기울이도록"
    하는데 유용합니다.

__반환값__

(모델이 단일 아웃풋만 갖고
측정항목을 보유하지 않는 경우) 스칼라 학습
손실, 혹은 (모델이 다중 아웃풋, 그리고/혹은 측정항목을
갖는 경우) 스칼라 리스트. `model.metrics_names` 속성은
스칼라 아웃풋에 대한 디스플레이 라벨을 제공합니다.
    
----

### test_on_batch


```python
test_on_batch(x, y, sample_weight=None)
```


하나의 샘플 배치에 대해서 모델을 테스트합니다.

__인수__

- __x__: 테스트 데이터의 Numpy 배열, 혹은
    모델이 다중 인풋을 갖는 경우 Numpy 배열의 리스트.
    모델의 모든 인풋에 이름이 명명된 경우
    인풋 이름을 Numpy 배열에 매핑하는
    딕셔너리를 전달할 수도 있습니다.
- __y__: 표적 데이터의 Numpy 배열,
    혹은 모델이 다중 아웃풋을 갖는 경우 Numpy 배열의 리스트.
    모델의 모든 아웃풋에 이름이 명명된 경우
    아웃풋 이름을 Numpy 배열에 매핑하는
    딕셔너리를 전달할 수도 있습니다.
- __sample_weight__: x와 동일한 길이의 선택적 배열로,
    각 샘플에 대한 모델의 손실에 적용할 가중치를 담습니다.
    시간적 데이터의 경우, (samples, sequence_length)의
    형태를 가진 2D 배열을 전달하여 모든 샘플의
    모든 시간 단계에 대해 각기 다른 가중치를 적용할 수 있습니다.
    이러한 경우 반드시 `compile()` 내에서
    `sample_weight_mode="temporal"`을 특정해야 합니다.

__반환값__

(모델이 단일 아웃풋만 갖고 측정항목을 보유하지 않는 경우)
스칼라 테스트 손실, 혹은 (모델이 다중 아웃풋, 그리고/혹은 측정항목을
갖는 경우) 스칼라 리스트. `model.metrics_names` 속성은
스칼라 아웃풋에 대한 디스플레이 라벨을 제공합니다.
    
----

### predict_on_batch


```python
predict_on_batch(x)
```


하나의 샘플 배치에 대한 예측 값을 반환합니다.

__인수__

- __x__: 인풋 샘플의 Numpy 배열.

__반환값__

예측 값의 Numpy 배열.
    
----

### fit_generator


```python
fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
```


파이썬 생성기(혹은 `Sequence` 인스턴스)가 배치 단위로 생산한
데이터에 대해서 모델을 학습시킵니다.

효율성을 위해 생성기가 모델에 병행하여 작동합니다.
예를 들어 이는 CPU의 이미지에 대해 실시간 데이터 증강을 실행하는
동시에, GPU에서 모델을 학습할 수 있도록 합니다.

`keras.utils.Sequence`는 순서를 정렬해주고,
또한 `use_multiprocessing=True`를 같이 사용하면
모든 인풋이 세대당 한 번만 사용됩니다.

__인수__

- __generator__: 다중 처리 과정을
    사용하는 경우 데이터 중복을 피하기 위한 생성기,
    혹은 `Sequence` (`keras.utils.Sequence`) 객체.
    생성기의 아웃풋은 반드시 다음 중 하나이어야 합니다:
    - `(inputs, targets)` 튜플
    - `(inputs, targets, sample_weights)` 튜플.

    이 튜플(생성기의 단일 아웃풋)은 하나의 배치가 됩니다.
    따라서 이 튜플의 모든 배열은 동일한 (이 배치의 크기와 같은)
    길이를 가져야 합니다. 배치에 따라 크기는 다를 수 있습니다.
    예를 들어 데이터셋의 크기가 배치 크기로
    정확히 나누어 떨어지지 않으면 세대의 마지막
    배치는 다른 배치보다 작은 경우가 흔합니다.
    생성기는 데이터에 대해서 무기한으로
    루프됩니다. 모델이 `steps_per_epoch`개의 배치를
    처리하면 한 세대가 끝나게 됩니다.

- __steps_per_epoch__: 정수.
    한 세대의 종료를 선언하고 다음 세대를
    시작하기까지 `generator`에서 생성할
    단계(샘플 배치)의 총 개수. 보통은
    데이터셋의 샘플 수를 배치 크기로
    나눈 값을 갖습니다.
    `Sequence`에 대한 선택사항입니다: 따로 정하지 않으면,
    단계의 개수는 `len(generator)`의 값이 됩니다.
- __epochs__: 정수. 모델을 학습시킬 세대의 수.
    한 세대는 `steps_per_epoch`에서 정의된 대로
    제공된 모든 데이터에 대한 1회 반복입니다.
    `initial_epoch`이 첫 세대라면,
    `epochs`은 "마지막 세대"로 이해하면 됩니다.
    모델은 `epochs`에 주어진 반복의 수만큼
    학습되는 것이 아니라, 단지 `epochs` 색인의 세대에
    도달할 때까지 학습됩니다.
- __verbose__: 정수. 0, 1, 혹은 2. 다변 모드.
    0 = 자동, 1 = 진행 표시줄, 2 = 세대 당 한 라인.
- __callbacks__: `keras.callbacks.Callback` 인스턴스의 리스트.
    학습 과정에서 적용할 콜백의 리스트입니다.
    [콜백](/callbacks)을 참조하십시오.
- __validation_data__: 다음 중 하나의 형태를 취합니다:
    - 생성기 혹은 검증 데이터용 `Sequence` 객체
    - `(x_val, y_val)` 튜플
    - `(x_val, y_val, val_sample_weights)` 튜플

    이를 기준으로
    각 세대의 끝에 손실과 모델 측정항목을 평가합니다.
    모델은 이 데이터에 대해서는 학습되지 않습니다.

- __validation_steps__: `validation_data`가
    생성기인 경우에만 유의미한, 매 세대 정지하기 전
    `validation_data` 생성기에서 만들어 낼
    단계(샘플 배치)의 총 개수. 보통은 검증
    데이터셋의 샘플 수를 배치 크기로
    나눈 값을 갖습니다.
    `Sequence`에 대한 선택사항입니다: 따로 정하지 않으면,
    단계의 개수는 `len(validation_data)`의 값이 됩니다.
- __class_weight__: 가중치 값(부동소수점)에 클래스 색인(정수)을
    매핑하는 선택적 딕셔너리로, (학습 과정 중에 한해)
    손실 함수에 가중치를 적용하는데 사용됩니다.
    모델이 상대적으로 적은 수의 샘플을 가진 클래스의
    샘플에 "더 주의를 기울이도록" 하는데 유용합니다.
- __max_queue_size__: 정수. 생성기 큐의 최대 크기.
    따로 정하지 않으면, `max_queue_size`의 디폴트 값은 10입니다.
- __workers__: 정수. 스레딩 기반의 처리 과정을 사용하는 경우
    가동할 수 있는 처리 과정의 최대 개수.
    따로 정하지 않으면, `workers`의 디폴트 값은 1입니다. 0인 경우
    생성기를 메인 스레드에서 실행합니다.
- __use_multiprocessing__: 불리언.
    `True`인 경우 스레딩 기반의 처리 과정을 사용합니다.
    따로 정하지 않으면, `use_multiprocessing`의 디폴트 값은 `False`입니다.
    이러한 구현 방식은
    다중 처리 과정에 기반하기 때문에,
    자식 처리 과정에 쉽게 전달할 수 없는
    비 picklable 인수를 생성기에 전달하면 안됩니다.
- __shuffle__: 불리언. 각 세대의 시작에
    배치의 순서를 뒤섞을지 여부.
    반드시 `Sequence` (`keras.utils.Sequence`) 인스턴스와 같이 사용합니다.
    `steps_per_epoch`이 `None`이 아닌 경우 아무 효과도 없습니다.
- __initial_epoch__: 정수.
    학습을 시작하는 세대
    (이전의 학습 과정을 재개하는데 유용합니다).

__반환값__

`History` 객체. `History.history` 속성은
연속된 세대에 걸친 학습 손실 값과 측정항목 값,
그리고 (적용 가능한 경우) 검증 손실 값과
검증 측정항목 값의 기록입니다.

__오류 알림__

- __ValueError__: 생성기가 유효하지 않은 형식의 데이터를 만들어 내는 경우 오류를 알립니다.

__예시__


```python
def generate_arrays_from_file(path):
    while True:
        with open(path) as f:
            for line in f:
                # 파일의 각 라인으로부터
                # 인풋 데이터와 라벨의 numpy 배열을 만듭니다
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


데이터 생성기에 대해서 모델을 평가합니다.

생성기는 `test_on_batch`에서 수용되는 것과 동일한
종류의 데이터를 반환해야 합니다.

__인수__

- __generator__: 다중 처리 과정을 사용하는 경우
    데이터 중복을 피하기 위해
    (inputs, targets) 튜플이나
    (inputs, targets, sample_weights) 튜플을 만들어내는 생성기,
    혹은 `Sequence` (`keras.utils.Sequence`) 객체.
- __steps__: 정지 전 `generator`에서
    만들어 낼 단계(샘플 배치)의 총 개수.
    `Sequence`에 대한 선택사항입니다: 따로 정하지 않으면,
    단계의 개수는 `len(generator)`의 값이 됩니다.
- __callbacks__: `keras.callbacks.Callback` 인스턴스의 리스트.
    학습과 검증 과정에서 적용할 콜백의 리스트입니다.
    [콜백](/callbacks)을 참조하십시오.
- __max_queue_size__: 생성기 큐의 최대 크기.
- __workers__: 정수. 스레딩 기반의 처리 과정을 사용하는 경우
    가동할 수 있는 처리 과정의 최대 개수
    따로 정하지 않으면, `workers`의 디폴트 값은 1입니다. 0인 경우
    생성기를 메인 스레드에서 실행합니다.
- __use_multiprocessing__: `True`인 경우 스레딩 기반의
    처리 과정을 사용합니다.
    이러한 구현 방식은 다중 처리 과정에
    기반하기 때문에,
    자식 처리 과정에 쉽게 전달할 수 없는
    비 picklable 인수를
    생성기에 전달하면 안됩니다.
- __verbose__: 다변 모드, 0 혹은 1.

__반환값__

(모델이 단일 아웃풋만 갖고 측정항목을 보유하지 않는 경우)
스칼라 테스트 손실, 혹은 (모델이 다중 아웃풋, 그리고/혹은 측정항목을
갖는 경우) 스칼라 리스트. `model.metrics_names`속성은
스칼라 아웃풋에 대한 디스플레이 라벨을 제공합니다.

__오류 알림__

- __ValueError__: 생성기가 유효하지 않은 형식의 데이터를 만들어 내는 경우
    오류를 알립니다.
    
----

### predict_generator


```python
predict_generator(generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
```


데이터 생성기의 인풋 샘플에 대해서 예측 값을 생성합니다.

생성기는 `predict_on_batch`에서 수용되는 것과 동일한
종류의 데이터를 반환해야 합니다.

__인수__

- __generator__: 다중 처리 과정을
    사용하는 경우 데이터 중복을
    피하기 위한 생성기, 혹은
    `Sequence` (`keras.utils.Sequence`) 객체.
- __steps__: 정지 전 `generator`에서
    만들어 낼 단계(샘플 배치)의 총 개수.
    `Sequence`에 대한 선택사항입니다: 따로 정하지 않으면,
    단계의 개수는 `len(generator)`의 값이 됩니다.
- __callbacks__: `keras.callbacks.Callback` 인스턴스의 리스트.
    학습과 검증 과정에서 적용할 콜백의 리스트입니다.
    [콜백](/callbacks)을 참조하십시오.
- __max_queue_size__: 생성기 큐의 최대 크기.
- __workers__: 정수. 스레딩 기반의 처리 과정을 사용하는 경우
    가동할 수 있는 처리 과정의 최대 개수.
    따로 정하지 않으면, `workers`의 디폴트 값은 1입니다. 0인 경우
    생성기를 메인 스레드에서 실행합니다.
- __use_multiprocessing__: `True`인 경우 스레딩 기반의
    처리 과정을 사용합니다.
    이러한 구현 방식은 다중 처리 과정에
    기반하기 때문에,
    자식 처리 과정에 쉽게 전달할 수 없는
    비 picklable 인수를
    생성기에 전달하면 안됩니다.
- __verbose__: 다변 모드, 0 혹은 1.

__반환값__

예측 값의 Numpy 배열.

__오류 알림__

- __ValueError__: 생성기가 유효하지 않은 형식의 데이터를 만들어 내는 경우
    오류를 알립니다.
    
----

### get_layer


```python
get_layer(name=None, index=None)
```


레이어의 (고유한) 이름, 혹은 색인에 따라 레이어를 회수합니다.

`name`과 `index`가 모두 제공되는 경우, `index`가 우선 순위를 갖습니다.

색인은 (상향식) 가로 그래프 횡단 순서에 기반합니다.

__인수__

- __name__: 문자열, 레이어의 이름.
- __index__: 정수, 레이어의 색인.

__반환값__

레이어 인스턴스.

__오류 알림__

- __ValueError__: 레이어 이름이나 색인이 유효하지 않은 경우 오류를 알립니다.
    
