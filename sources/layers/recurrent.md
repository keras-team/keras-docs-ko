<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L237)</span>
### RNN

```python
keras.layers.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

순환 레이어의 베이스 클래스.

__인수__

- __cell__: 순환 신경망 셀 인스턴스. 순환 신경망 셀은 다음의 요소를 가진 클래스입니다:
    - `(output_at_t, states_at_t_plus_1)`을 반환하는
        `call(input_at_t, states_at_t)` 메서드. 셀의 호출
        메서드는 선택적 인수인 `constants`를 받을 수도 있습니다.
        밑의 "외부 상수 전달의 유의점" 섹션을 참고하십시오.
    - `state_size` 속성. 단일 정수(단일 상태)일 경우
        순환 상태의 크기를
        나타냅니다 (이는 셀 아웃풋의
        크기와 동일해야 합니다).
        정수의 리스트/튜플의 형태(상태당 하나의 크기)를 
        취할 수도 있습니다.
    - `output_size` 속성. 단일 정수, 혹은
        아웃풋의 형태를 나타내는 TensorShape.
        역방향 호환성 이유로 셀에서 이 속성을 사용할 수
        없는 경우, `state_size`의 첫 번째 성분에서 값을
        유추합니다.

    순환 신경망 내에서 셀을 겹쳐 쌓아 효율적인
    스택 순환 신경망을 구현하는 경우, `cell`은 순환 신경망 셀
    인스턴스의 리스트가 될 수도 있습니다.

- __return_sequences__: 불리언. 아웃풋 시퀀스의 마지막 아웃풋을 반환할지,
    혹은 시퀀스 전체를 반환할지 여부.
- __return_state__: 불리언. 아웃풋에 더해 마지막 상태도
    반환할지 여부.
- __go_backwards__: 불리언 (디폴트 값은 거짓).
    참인 경우, 인풋 시퀀스를 거꾸로 처리하여
    뒤집힌 시퀀스를 반환합니다.
- __stateful__: 불리언 (디폴트 값은 거짓). 참인 경우,
    배치 내 색인 i의 각 샘플의 마지막 상태가 다음 배치의
    색인 i 샘플의 초기 상태로 사용됩니다.
- __unroll__: 불리언 (디폴트 값은 거짓).
    참인 경우, 신경망을 펼쳐서 사용하고
    그렇지 않은 경우 심볼릭 루프가 사용됩니다.
    신경망을 펼쳐 순환 신경망의 속도를
    높일 수 있지만, 메모리 소모가 큰 경향이 있습니다.
    신경망 펼치기는 짧은 시퀀스에만 적합합니다.
- __input_dim__: 인풋의 차원 (정수).
    이 인수(혹은 키워드 인수 `input_shape`)는
    현재 레이어를 모델의
    첫 번째 레이어로 사용하는 경우 요구됩니다.
- __input_length__: 인풋 시퀀스의 길이.
    값이 상수일 경우 특정합니다.
    `Flatten`과 상위의 `Dense` 레이어를 연결하는
    경우 이 인수가 필요합니다
    (이 인수 없이는 밀집 아웃풋의 형태를 계산할 수 없습니다). 이 인수는
    순환 레이어가 모델의 첫 레이어가 아닌 경우
    첫 레이어에서 (예. `input_shape` 인수를 통해서)
    인풋 길이를 특정해야
    한다는 것을 참고하십시오.

__인풋 형태__

`(batch_size, timesteps, input_dim)` 형태의 3D 텐서.

__아웃풋 형태__

- `return_state`인 경우: 텐서의 리스트. 첫 번째 텐서는
    아웃풋 입니다. 나머지 텐서는 마지막 상태로, 각각
    `(batch_size, units)`의 형태를 갖습니다. 예를 들어 , (순환 신경망과
    회로형 순환 유닛의 경우) 1, (장단기 메모리의 경우) 2입니다.
- `return_sequences`인 경우: `(batch_size, timesteps, units)`
    형태의 3D 텐서.
- 그 외의 경우, `(batch_size, units)` 형태의 2D 텐서.

__마스킹__

이 레이어는 가변적 시간 단계의 개수를 가진 인풋 데이터에 대한
마스킹을 지원합니다. 데이터에 마스크를 도입하려면
`mask_zero` 매개변수를 `True`로 설정한 ‘[Embedding](embeddings.md) 레이어를
사용하십시오.

__순환 신경망의 상태 기반 모드 사용에 대한 주의점__

순환 신경망 레이어를 ‘stateful’로 설정할 수 있는데,
이는 한 배치 내 샘플의 상태를 계산하여, 이 값을
다음 배치 내 샘플의 초기 상태로 재사용한다는 의미입니다.
이는 연속된 배치 내 샘플간 일대일 매핑을 가정합니다.

상태 기반 모드를 사용하려면:
- 레이어 생성자에 `stateful=True`를 특정하십시오.
- 다음을 전달하여 모델의 고정 배치 크기를 특정하십시오:
시퀀스 모델의 경우
모델의 첫 레이어에 `batch_input_shape=(...)`를 전달하십시오.
그 외의 1개 이상의 인풋 레이어를 가진 기능적 모델의 경우
모델의 모든 첫 레이어에 `batch_shape=(...)`를 전달하십시오.
이는 *배치 크기를 포함한* 인풋의
예상되는 형태를 나타냅니다.
그 값은 정수 튜플이 되어야 합니다. 예. `(32, 10, 100)`
- fit()을 호출할 때 `shuffle=False`로 설정하십시오.

모델의 상태를 재설정하려면, 특정 레이어 혹은 전체 모델에 대해서
`.reset_states()`를 호출하십시오.

__순환 신경망 초기 상태 특정에 대한 안내__

키워드 인수 `initial_state`로 순환 레이어를
호출하여 순환 레이어의 초기 상태를 특정할 수 있습니다.
`initial_state`의 값은 순환 레이어의 초기 상태를 나타내는
텐서 혹은 텐서의 리스트이 되어야 합니다.

키워드 인수 `states`로 `reset_states`를 호출하여
순환 신경망 레이어의 초기 상태를 특정할 수 있습니다.
`states`의 값은 순환 신경망 레이어의 초기 상태를 나타내는
numpy 배열 혹은 numpy 배열의 리스트가 되어야 합니다.

__순환 신경망에 대한 외부 상수 전달의 유의점__

`RNN.__call__` (as well as `RNN.call`) 메서드의 `constants`
키워드 인수를 사용해 “외부” 상수를 셀에 전달할 수 있습니다.
이를 위해선 `cell.call` 메서드도 동일한 키워드 인수 `constants`를
수용해야 합니다. 그러한 상수를 사용해 (시간에 따라 변하지 않는) 추가적
고정 인풋에 대해서 셀 변형을 조정하는데, 이는
집중 메커니즘이라고도 불립니다.

__예시__


```python
# 우선 순환 신경망 셀을 레이어 하위클래스로 정의해 봅시다.

class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]

# 이 셀을 순환 신경망 레이어에 사용해 봅시다:

cell = MinimalRNNCell(32)
x = keras.Input((None, 5))
layer = RNN(cell)
y = layer(x)

# 다음은 스택 순환 신경망을 구성하기 위해 셀을 어떻게 사용하는지 보여줍니다:

cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
x = keras.Input((None, 5))
layer = RNN(cells)
y = layer(x)
```
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L945)</span>
### SimpleRNN

```python
keras.layers.SimpleRNN(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

아웃풋이 인풋으로 다시 연결되는 완전 연결 순환 신경망.

__인수__

- __units__: 양의 정수, 아웃풋 공간의 차원입니다.
- __activation__: 사용할 활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    디폴트: 쌍곡 탄젠트 (`tanh`).
    `None`을 전달하는 경우, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __use_bias__: 불리언, 레이어가 편향 벡터를 사용하는지 여부.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기값 설정기.
    인풋의 선형적 변형에 사용됩니다
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __recurrent_initializer__: `recurrent_kernel` 가중치 행렬의
    초기값 설정기.
    순환 상태의 선형적 변형에 사용됩니다
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __bias_initializer__: 편향 벡터의 초기값 설정기
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __recurrent_regularizer__: `recurrent_kernel` 
    가중치 행렬에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __activity_regularizer__: 레이어의 아웃풋(레이어의 “활성화”)에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __recurrent_constraint__: `recurrent_kernel` 
    가중치 행렬에 적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __dropout__: 0과 1사이 부동소수점.
    인풋의 선형적 변형을 실행하는데
    드롭시킬(고려하지 않을) 유닛의 비율.
- __recurrent_dropout__: 0과 1사이 부동소수점.
    순환 상태의 선형적 변형을 실행하는데
    드롭시킬(고려하지 않을) 유닛의 비율.
- __return_sequences__: 불리언. 아웃풋 시퀀스의 마지막 아웃풋을 반환할지,
    혹은 시퀀스 전체를 반환할지 여부.
- __return_state__: 불리언. 아웃풋에 더해 마지막 상태도
    반환할지 여부.
- __go_backwards__: 불리언 (디폴트 값은 거짓).
    참인 경우, 인풋 시퀀스를 거꾸로 처리하여
    뒤집힌 시퀀스를 반환합니다.
- __stateful__: 불리언 (디폴트 값은 거짓). 참인 경우,
    배치 내 색인 i의 각 샘플의 마지막 상태가 다음 배치의
    색인 i 샘플의 초기 상태로 사용됩니다.
- __unroll__: 불리언 (디폴트 값은 거짓).
    참인 경우, 신경망을 펼쳐서 사용하고
    그렇지 않은 경우 심볼릭 루프가 사용됩니다.
    신경망을 펼쳐 순환 신경망의 속도를
    높일 수 있지만, 메모리 소모가 큰 경향이 있습니다.
    신경망 펼치기는 짧은 시퀀스에만 적합합니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1491)</span>
### GRU

```python
keras.layers.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False)
```

회로형 순환 유닛 - Cho et al. 2014.

두 가지 종류가 있습니다. 디폴트인 첫 번째 종류는 1406.1078v3에 
기반하고 행렬 곱셈 전 재설정 회로가 은닉 상태에 적용됩니다. 다른 하나는
원래의 1406.1078v1에 기반하고 그 순서가 반대입니다.

두 번째 종류는 (GPU 전용) CuDNNGRU와 호환이 가능하며
CPU 상에서 유추를 실행할 수 있도록 합니다. 따라서 `kernel`과
`recurrent_kernel`에 대해 별도의 편향을 갖습니다.
`'reset_after'=True`와 `recurrent_activation='sigmoid'`를 사용하십시오.

__인수__

- __units__: 양의 정수, 아웃풋 공간의 차원입니다.
- __activation__: 사용할 활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    디폴트: 쌍곡 탄젠트 (`tanh`).
    `None`을 전달하는 경우, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __recurrent_activation__: 순환 단계에 사용할
    활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    디폴트 값: 하드 시그모이드 (`hard_sigmoid`).
    `None`을 전달하는 경우, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __use_bias__: 불리언, 레이어가 편향 벡터를 사용하는지 여부.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기값 설정기.
    인풋의 선형적 변형에 사용됩니다
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __recurrent_initializer__: `recurrent_kernel` 가중치 행렬의
    초기값 설정기.
    순환 상태의 선형적 변형에 사용됩니다
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __bias_initializer__: 편향 벡터의 초기값 설정기
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __recurrent_regularizer__: `recurrent_kernel` 
    가중치 행렬에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __activity_regularizer__: 레이어의 아웃풋(레이어의 “활성화”)에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __recurrent_constraint__: `recurrent_kernel` 
    가중치 행렬에 적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __dropout__: 0과 1사이 부동소수점.
    인풋의 선형적 변형을 실행하는데
    드롭시킬(고려하지 않을) 유닛의 비율.
- __recurrent_dropout__: 0과 1사이 부동소수점.
    순환 상태의 선형적 변형을 실행하는데
    드롭시킬(고려하지 않을) 유닛의 비율.
- __implementation__: 실행 모드, 1 혹은 2.
    모드 1은 비교적 많은 수의 소규모의 점곱과 덧셈을
    이용해 연산을 구성하는데 반해, 모드 2는 이를
    소수의 대규모 연산으로 묶습니다. 이 두 모드는,
    하드웨어나 어플리케이션에 따라서 성능의 차이를
    보입니다.
- __return_sequences__: 불리언. 아웃풋 시퀀스의 마지막 아웃풋을 반환할지,
    혹은 시퀀스 전체를 반환할지 여부.
- __return_state__: 불리언. 아웃풋에 더해 마지막 상태도
    반환할지 여부.
- __go_backwards__: 불리언 (디폴트 값은 거짓).
    참인 경우, 인풋 시퀀스를 거꾸로 처리하여
    뒤집힌 시퀀스를 반환합니다.
- __stateful__: 불리언 (디폴트 값은 거짓). 참인 경우,
    배치 내 색인 i의 각 샘플의 마지막 상태가 다음 배치의
    색인 i 샘플의 초기 상태로 사용됩니다.
- __unroll__: 불리언 (디폴트 값은 거짓).
    참인 경우, 신경망을 펼쳐서 사용하고
    그렇지 않은 경우 심볼릭 루프가 사용됩니다.
    신경망을 펼쳐 순환 신경망의 속도를
    높일 수 있지만, 메모리 소모가 큰 경향이 있습니다.
    신경망 펼치기는 짧은 시퀀스에만 적합합니다.
- __reset_after__: 회로형 순환 유닛의 상례 (행렬 곱셈 전이나 후에
    재설정 회로를 적용할지 여부). 거짓 = "before" (디폴트 값),
    참 = "after" (CuDNN과 호환)

__참조__

- [Learning Phrase Representations using RNN Encoder-Decoder for
   Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
- [On the Properties of Neural Machine Translation:
   Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
- [Empirical Evaluation of Gated Recurrent Neural Networks on
   Sequence Modeling](https://arxiv.org/abs/1412.3555v1)
- [A Theoretically Grounded Application of Dropout in
   Recurrent Neural Networks](https://arxiv.org/abs/1512.05287)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L2051)</span>
### LSTM

```python
keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

장단기 메모리 레이어 - Hochreiter 1997.

__인수__

- __units__: 양의 정수, 아웃풋 공간의 차원입니다.
- __activation__: 사용할 활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    디폴트: 쌍곡 탄젠트 (`tanh`).
    `None`을 전달하는 경우, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __recurrent_activation__: 순환 단계에 사용할
    활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    디폴트 값: 하드 시그모이드 (`hard_sigmoid`).
    `None`을 전달하는 경우, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __use_bias__: 불리언, 레이어가 편향 벡터를 사용하는지 여부.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기값 설정기.
    인풋의 선형적 변형에 사용됩니다
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __recurrent_initializer__: `recurrent_kernel` 가중치 행렬의
    초기값 설정기.
    순환 상태의 선형적 변형에 사용됩니다
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __bias_initializer__: 편향 벡터의 초기값 설정기
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __unit_forget_bias__: 불리언.
    참일 경우, 초기값 설정 시 망각 회로에 1을 더합니다.
    참으로 설정 시 강제적으로 `bias_initializer="zeros"`가 됩니다.
    이는 [Jozefowicz et al. (2015)] 에서 권장됩니다.
    (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __recurrent_regularizer__: `recurrent_kernel` 
    가중치 행렬에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __activity_regularizer__: 레이어의 아웃풋(레이어의 “활성화”)에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __recurrent_constraint__: `recurrent_kernel` 
    가중치 행렬에 적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __dropout__: 0과 1사이 부동소수점.
    인풋의 선형적 변형을 실행하는데
    드롭시킬(고려하지 않을) 유닛의 비율.
- __recurrent_dropout__: 0과 1사이 부동소수점.
    순환 상태의 선형적 변형을 실행하는데
    드롭시킬(고려하지 않을) 유닛의 비율.
- __implementation__: 실행 모드, 1 혹은 2.
    모드 1은 비교적 많은 수의 소규모의 점곱과 덧셈을
    이용해 연산을 구성하는데 반해, 모드 2는 이를
    소수의 대규모 연산으로 묶습니다. 이 두 모드는,
    하드웨어나 어플리케이션에 따라서 성능의 차이를
    보입니다.
- __return_sequences__: 불리언. 아웃풋 시퀀스의 마지막 아웃풋을 반환할지,
    혹은 시퀀스 전체를 반환할지 여부.
- __return_state__: 불리언. 아웃풋에 더해 마지막 상태도
    반환할지 여부. 상태 리스트의 반환된 성분은
    각각 은닉 성분과 셀 상태입니다.
- __go_backwards__: 불리언 (디폴트 값은 거짓).
    참인 경우, 인풋 시퀀스를 거꾸로 처리하여
    뒤집힌 시퀀스를 반환합니다.
- __stateful__: 불리언 (디폴트 값은 거짓). 참인 경우,
    배치 내 색인 i의 각 샘플의 마지막 상태가 다음 배치의
    색인 i 샘플의 초기 상태로 사용됩니다.
- __unroll__: 불리언 (디폴트 값은 거짓).
    참인 경우, 신경망을 펼쳐서 사용하고
    그렇지 않은 경우 심볼릭 루프가 사용됩니다.
    신경망을 펼쳐 순환 신경망의 속도를
    높일 수 있지만, 메모리 소모가 큰 경향이 있습니다.
    신경망 펼치기는 짧은 시퀀스에만 적합합니다.

__참조__

- [Long short-term memory](
  http://www.bioinf.jku.at/publications/older/2604.pdf)
- [Learning to forget: Continual prediction with LSTM](
  http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
- [Supervised sequence labeling with recurrent neural networks](
  http://www.cs.toronto.edu/~graves/preprint.pdf)
- [A Theoretically Grounded Application of Dropout in
   Recurrent Neural Networks](https://arxiv.org/abs/1512.05287)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional_recurrent.py#L788)</span>
### ConvLSTM2D

```python
keras.layers.ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0)
```

컨볼루션 장단기 메모리.

장단기 메모리 레이어와 비슷하나, 인풋 변형과
순환 변형 둘 모두 컨볼루션이 적용됩니다.

__인수__

- __filters__: 정수, 아웃풋 공간의 차원
    (다시 말해, 컨볼루션 내 필터의 아웃풋의 개수).
- __kernel_size__: 정수 혹은 n개 정주의 튜플/리스트.
    컨볼루션 윈도우의 차원을 특정합니다.
- __strides__: 정수 혹은 n개의 정수의 튜플/리스트.
    컨볼루션의 보폭을 특정합니다.
    보폭 값 != 1이면 `dilation_rate` 값 != 1의
    어떤 경우와도 호환이 불가합니다.
- __padding__: `"valid"` 혹은 `"same"`(대소문자 무시) 중 하나.
- __data_format__: 문자열,
    `"channels_last"` (디폴트 값) 혹은 "channels_first"` 중 하나.
    인풋 내 차원의 정렬을 나타냅니다.
    `"channels_last"`는 `(batch, time, ..., channels)`
    형태의 인풋에 호응하고
    `"channels_first"`는 `(batch, time, channels, ...)`
    형태의 인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.
- __dilation_rate__: 정수 혹은 n개의 정수의 튜플/리스트.
    팽창 컨볼루션의 팽창 속도를 특정합니다.
    현재는 `dilation_rate` 값 != 1이면
    `strides` 값 != 1의 어떤 경우와도 호환이 불가합니다.
- __activation__: 사용할 활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    `None`을 전달하는 경우, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __recurrent_activation__: 순환 단계에 사용할
    활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
- __use_bias__: 불리언, 레이어가 편향 벡터를 사용하는지 여부.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기값 설정기.
    인풋의 선형적 변형에 사용됩니다
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __recurrent_initializer__: `recurrent_kernel` 가중치 행렬의
    초기값 설정기.
    순환 상태의 선형적 변형에 사용됩니다
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __bias_initializer__: 편향 벡터의 초기값 설정기
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __unit_forget_bias__: 불리언.
    참일 경우, 초기값 설정 시 망각 회로에 1을 더합니다.
    참으로 설정 시 강제적으로 `bias_initializer="zeros"`가 됩니다.
    이는 [Jozefowicz et al. (2015)] 에서 권장됩니다.
    (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __recurrent_regularizer__: `recurrent_kernel` 
    가중치 행렬에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __activity_regularizer__: 레이어의 아웃풋(레이어의 “활성화”)에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __recurrent_constraint__: `recurrent_kernel` 
    가중치 행렬에 적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __return_sequences__: 불리언. 아웃풋 시퀀스의 마지막 아웃풋을 반환할지,
    혹은 시퀀스 전체를 반환할지 여부.
- __go_backwards__: 불리언 (디폴트 값은 거짓).
    참인 경우, 인풋 프로세스를 거꾸로 처리합니다.
- __stateful__: 불리언 (디폴트 값은 거짓). 참인 경우,
    배치 내 색인 i의 각 샘플의 마지막 상태가 다음 배치의
    색인 i 샘플의 초기 상태로 사용됩니다.
- __dropout__: 0과 1사이 부동소수점.
    인풋의 선형적 변형을 실행하는데
    드롭시킬(고려하지 않을) 유닛의 비율.
- __recurrent_dropout__: 0과 1사이 부동소수점.
    순환 상태의 선형적 변형을 실행하는데
    드롭시킬(고려하지 않을) 유닛의 비율.

__인풋 형태__

- data_format='channels_first'의 경우
    다음과 같은 형태의 5D 텐서:
    `(samples, time, channels, rows, cols)`
- data_format='channels_last'의 경우
    다음과 같은 형태의 5D 텐서:
    `(samples, time, rows, cols, channels)`

__아웃풋 형태__

- `return_sequences`의 경우
     - data_format='channels_first'이면
        다음과 같은 형태의 5D 텐서:
        `(samples, time, filters, output_row, output_col)`
     - data_format='channels_last'이면
        다음과 같은 형태의 5D 텐서:
        `(samples, time, output_row, output_col, filters)`
- 그 외의 경우
    - data_format='channels_first'이면
        다음과 같은 형태의 4D 텐서:
        `(samples, filters, output_row, output_col)`
    - data_format='channels_last'이면
        다음과 같은 형태의 4D 텐서:
        `(samples, output_row, output_col, filters)`

    o_row와 o_col의 값은 필터의 형태와 패딩에 따라
    달라집니다.

__오류 알림__

- __ValueError__: 유효하지 않은 생성자 인수를 전달받는 경우 오류를 알립니다.

__참조__

- [Convolutional LSTM Network: A Machine Learning Approach for
  Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)
  현재 구현방식은 셀의 아웃풋에 대한 피드백 루프를
  포함하지 않습니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L780)</span>
### SimpleRNNCell

```python
keras.layers.SimpleRNNCell(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```

SimpleRNN의 셀 클래스.

__인수__

- __units__: 양의 정수, 아웃풋 공간의 차원입니다.
- __activation__: 사용할 활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    디폴트: 쌍곡 탄젠트 (`tanh`).
    `None`을 전달하는 경우, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __use_bias__: 불리언, 레이어가 편향 벡터를 사용하는지 여부.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기값 설정기.
    인풋의 선형적 변형에 사용됩니다
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __recurrent_initializer__: `recurrent_kernel` 가중치 행렬의
    초기값 설정기.
    순환 상태의 선형적 변형에 사용됩니다
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __bias_initializer__: 편향 벡터의 초기값 설정기
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __recurrent_regularizer__: `recurrent_kernel` 
    가중치 행렬에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __recurrent_constraint__: `recurrent_kernel` 
    가중치 행렬에 적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __dropout__: 0과 1사이 부동소수점.
    인풋의 선형적 변형을 실행하는데
    드롭시킬(고려하지 않을) 유닛의 비율.
- __recurrent_dropout__: 0과 1사이 부동소수점.
    순환 상태의 선형적 변형을 실행하는데
    드롭시킬(고려하지 않을) 유닛의 비율.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1164)</span>
### GRUCell

```python
keras.layers.GRUCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, reset_after=False)
```

회로형 순환 유닛 레이어의 셀 클래스.

__인수__

- __units__: 양의 정수, 아웃풋 공간의 차원입니다.
- __activation__: 사용할 활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    디폴트: 쌍곡 탄젠트 (`tanh`).
    `None`을 전달하는 경우, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __recurrent_activation__: 순환 단계에 사용할
    활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    디폴트 값: 하드 시그모이드 (`hard_sigmoid`).
    `None`을 전달하는 경우, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __use_bias__: 불리언, 레이어가 편향 벡터를 사용하는지 여부.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기값 설정기.
    인풋의 선형적 변형에 사용됩니다
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __recurrent_initializer__: `recurrent_kernel` 가중치 행렬의
    초기값 설정기.
    순환 상태의 선형적 변형에 사용됩니다
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __bias_initializer__: 편향 벡터의 초기값 설정기
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __recurrent_regularizer__: `recurrent_kernel` 
    가중치 행렬에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __recurrent_constraint__: `recurrent_kernel` 
    가중치 행렬에 적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __dropout__: 0과 1사이 부동소수점.
    인풋의 선형적 변형을 실행하는데
    드롭시킬(고려하지 않을) 유닛의 비율.
- __recurrent_dropout__: 0과 1사이 부동소수점.
    순환 상태의 선형적 변형을 실행하는데
    드롭시킬(고려하지 않을) 유닛의 비율.
- __implementation__: 실행 모드, 1 혹은 2.
    모드 1은 비교적 많은 수의 소규모의 점곱과 덧셈을
    이용해 연산을 구성하는데 반해, 모드 2는 이를
    소수의 대규모 연산으로 묶습니다. 이 두 모드는,
    하드웨어나 어플리케이션에 따라서 성능의 차이를
    보입니다.
- __reset_after__: 회로형 순환 유닛의 상례 (행렬 승법 전이나 후에
    재설정 회로를 적용할지 여부). 거짓 = "before" (디폴트 값),
    참 =  "after" (CuDNN과 호환).
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1765)</span>
### LSTMCell

```python
keras.layers.LSTMCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1)
```

장단기 메모리의 셀 클래스.

__인수__

- __units__: 양의 정수, 아웃풋 공간의 차원입니다.
- __activation__: 사용할 활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    디폴트: 쌍곡 탄젠트 (`tanh`).
    `None`을 전달하는 경우, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __recurrent_activation__: 순환 단계에 사용할
    활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    디폴트 값: 하드 시그모이드 (`hard_sigmoid`).
    `None`을 전달하는 경우, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __use_bias__: 불리언, 레이어가 편향 벡터를 사용하는지 여부.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기값 설정기.
    인풋의 선형적 변형에 사용됩니다
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __recurrent_initializer__: `recurrent_kernel` 가중치 행렬의
    초기값 설정기.
    순환 상태의 선형적 변형에 사용됩니다
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __bias_initializer__: 편향 벡터의 초기값 설정기
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __unit_forget_bias__: 불리언.
    참일 경우, 초기값 설정 시 망각 회로에 1을 더합니다.
    참으로 설정 시 강제적으로 `bias_initializer="zeros"`가 됩니다.
    이는 [Jozefowicz et al. (2015)] 에서 권장됩니다.
    (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __recurrent_regularizer__: `recurrent_kernel` 
    가중치 행렬에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __recurrent_constraint__: `recurrent_kernel` 
    가중치 행렬에 적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __dropout__: 0과 1사이 부동소수점.
    인풋의 선형적 변형을 실행하는데
    드롭시킬(고려하지 않을) 유닛의 비율.
- __recurrent_dropout__: 0과 1사이 부동소수점.
    순환 상태의 선형적 변형을 실행하는데
    드롭시킬(고려하지 않을) 유닛의 비율.
- __implementation__: 실행 모드, 1 혹은 2.
    모드 1은 비교적 많은 수의 소규모의 점곱과 덧셈을
    이용해 연산을 구성하는데 반해, 모드 2는 이를
    소수의 대규모 연산으로 묶습니다. 이 두 모드는,
    하드웨어나 어플리케이션에 따라서 성능의 차이를
    보입니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/cudnn_recurrent.py#L135)</span>
### CuDNNGRU

```python
keras.layers.CuDNNGRU(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
```

[CuDNN](https://developer.nvidia.com/cudnn)을 사용한 빠른 회로형 순환 유닛 실행.

텐서플로우 백엔드로 GPU에서만 실행할 수 있습니다.

__인수__

- __units__: 양의 정수, 아웃풋 공간의 차원입니다.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기값 설정기.
    인풋의 선형적 변형에 사용됩니다
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __recurrent_initializer__: `recurrent_kernel` 가중치 행렬의
    초기값 설정기.
    순환 상태의 선형적 변형에 사용됩니다
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __bias_initializer__: 편향 벡터의 초기값 설정기
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __recurrent_regularizer__: `recurrent_kernel` 
    가중치 행렬에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __activity_regularizer__: 레이어의 아웃풋(레이어의 “활성화”)에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __recurrent_constraint__: `recurrent_kernel` 
    가중치 행렬에 적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __return_sequences__: 불리언. 아웃풋 시퀀스의 마지막 아웃풋을 반환할지,
    혹은 시퀀스 전체를 반환할지 여부.
- __return_state__: 불리언. 아웃풋에 더해 마지막 상태도
    반환할지 여부.
- __stateful__: 불리언 (디폴트 값은 거짓). 참인 경우,
    배치 내 색인 i의 각 샘플의 마지막 상태가 다음 배치의
    색인 i 샘플의 초기 상태로 사용됩니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/cudnn_recurrent.py#L328)</span>
### CuDNNLSTM

```python
keras.layers.CuDNNLSTM(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
```

[CuDNN](https://developer.nvidia.com/cudnn)을 사용한 빠른 장단기 메모리 실행.

텐서플로우 백엔드로 GPU에서만 실행할 수 있습니다.

__인수__

- __units__: 양의 정수, 아웃풋 공간의 차원입니다.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기값 설정기.
    인풋의 선형적 변형에 사용됩니다
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __unit_forget_bias__: 불리언.
    참일 경우, 초기값 설정 시 망각 회로에 1을 더합니다.
    참으로 설정 시 강제적으로 `bias_initializer="zeros"`가 됩니다.
    이는 [Jozefowicz et al. (2015)] 에서 권장됩니다.
    (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
- __recurrent_initializer__: `recurrent_kernel` 가중치 행렬의
    초기값 설정기.
    순환 상태의 선형적 변형에 사용됩니다
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __bias_initializer__: 편향 벡터의 초기값 설정기
    ( [초기값 설정기](../initializers.md)를 참조하십시오).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __recurrent_regularizer__: `recurrent_kernel` 
    가중치 행렬에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __activity_regularizer__: 레이어의 아웃풋(레이어의 “활성화”)에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __recurrent_constraint__: `recurrent_kernel` 
    가중치 행렬에 적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __return_sequences__: 불리언. 아웃풋 시퀀스의 마지막 아웃풋을 반환할지,
    혹은 시퀀스 전체를 반환할지 여부.
- __return_state__: 불리언. 아웃풋에 더해 마지막 상태도
    반환할지 여부.
- __stateful__: 불리언 (디폴트 값은 거짓). 참인 경우,
    배치 내 색인 i의 각 샘플의 마지막 상태가 다음 배치의
    색인 i 샘플의 초기 상태로 사용됩니다.
    
