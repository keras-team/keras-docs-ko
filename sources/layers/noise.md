<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/noise.py#L14)</span>
### GaussianNoise

```python
keras.layers.GaussianNoise(stddev)
```

평균 0의 가산적 가우시안 잡음을 적용합니다.

이는 과적합을 완화하는데 유용합니다
(무작위 데이터 증강의 하나로 볼 수 있습니다).
가우시안 잡음(GS)은 실수 값의 인풋을 변질 처리하는 목적에 있어
자연스러운 선택입니다.

정규화 레이어이므로, 학습 과정 중에만 활성화됩니다.

__인수__

- __stddev__: 부동소수점, 소음 분포의 표준 편차.

__인풋 형태__

임의의 형태를 취합니다. 이 레이어를 모델의 첫 번째 레이어로
사용하려면 키워드 인수 `input_shape`(정수 튜플로 샘플 축은 포함하지 않습니다)을
사용하십시오.

__아웃풋 형태__

인풋 형태와 동일.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/noise.py#L58)</span>
### GaussianDropout

```python
keras.layers.GaussianDropout(rate)
```

평균 1의 승법적 가우시안 잡음을 적용합니다.

정규화 레이어이므로, 학습 과정 중에만 활성화됩니다.

__인수__

- __rate__: 부동소수점, (`Dropout`에서처럼) 드롭 확률.
    이 승법적 잡음은
    `sqrt(rate / (1 - rate))`의 표준편차를 갖습니다.

__인풋 형태__

임의의 형태를 취합니다. 이 레이어를 모델의 첫 번째 레이어로
사용하려면 키워드 인수 `input_shape`(정수 튜플로 샘플 축은 포함하지 않습니다)을
사용하십시오.

__아웃풋 형태__

인풋 형태와 동일.

__참조__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
   http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/noise.py#L106)</span>
### AlphaDropout

```python
keras.layers.AlphaDropout(rate, noise_shape=None, seed=None)
```

인풋에 알파 드롭아웃을 적용합니다.

알파 드롭아웃은 드롭아웃 이후에도 자기-정규화 특성이
보장되도록 인풋의 평균과 분산을
원래 값으로 유지하는 `Dropout`입니다.
알파 드롭아웃은 활성화를 무작위로 음수 포화 값에 지정하여,
조정 지수 선형 유닛(Scaled Exponential Linear Units)에 대한 학습에 탁월합니다.

__인수__

- __rate__: 부동소수점, (`Dropout`에서처럼) 드롭 확률.
    이 승법적 잡음은
    `sqrt(rate / (1 - rate))`의 표준편차를 갖습니다.
- __seed__: 난수 시드로 사용할 파이썬 정수.

__인풋 형태__

임의의 형태를 취합니다. 이 레이어를 모델의 첫 번째 레이어로
사용하려면 키워드 인수 `input_shape`(정수 튜플로 샘플 축은 포함하지 않습니다)을
사용하십시오.

__아웃풋 형태__

인풋 형태와 동일.

__참조__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    
