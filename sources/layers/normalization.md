<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/normalization.py#L16)</span>
### BatchNormalization

```python
keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
```

배치 정규화 층<sub>batch normalization layer</sub>입니다. (Ioffe and Szegedy, 2014)

각 배치<sub>batch</sub>에서 이전 층<sub>layer</sub>의 activations를 정규화<sub>normalize</sub>합니다. 즉, activation의 평균은 0에 가깝도록 하고 표준편차는 1에 가깝게 유지하는 변환을 적용합니다. 

__인자 설명__

- __axis__: `int`, 정규화되어야 하는 축을 의미합니다.(일반적으로 요인 축<sub>feature axis</sub>입니다.) 예를 들어, `data_format="channels_first"`가 있는 `Conv2D`층의 다음에 `axis=1`인 `BatchNormalization`을 설정할 수 있습니다.
- __momentum__: 이동 평균<sub>moving mean</sub> 및 이동 분산<sub>moving variance</sub>에 대한 모멘텀을 의미합니다. 
- __epsilon__: 0으로 나누기를 방지하기 위해 분산에 추가되는 작은 `float`값 입니다.
- __center__: `True`일 경우, 정규화된 텐서에 `beta`만큼의 거리(offset)를 추가합니다. `False`인 경우 `beta`는 무시됩니다. 
- __scale__: `True`일 경우, `gamma`를 곱합니다. `False`인 경우 `gamma`는 사용되지 않습니다. 다음 층이 선형일 때(예: `nn.relu`), Scaling이 다음 층에서 수행될 것이기 때문에 사용되지 않을 수 있습니다. 
- __beta_initializer__: beta 가중치를 위한 초기화 함수<sub>initializer</sub>입니다. 
- __gamma_initializer__: gamma 가중치를 위한 초기화 함수입니다.
- __moving_mean_initializer__: 이동 평균<sub>moving mean</sub>을 위한 초기화 함수입니다.
- __moving_variance_initializer__: 이동 분산<sub>moving variance</sub>을 위한 초기화 함수입니다. 
- __beta_regularizer__: beta 가중치를 위해 선택적으로 사용 가능한 규제 함수<sub>regularizer</sub>입니다. 
- __gamma_regularizer__: gamma 가중치를 위해 선택적으로 사용 가능한 규제 함수입니다. 
- __beta_constraint__: beta 가중치를 위해 선택적으로 적용 가능한 제약입니다. 
- __gamma_constraint__: gamma 가중치를 위해 선택적으로 적용 가능한 제약입니다. 

__입력 크기__

임의입니다. 이 층을 모델의 첫 번째 층으로 사용할 때, 키워드 인자 `input_shape` (정수 튜플, 샘플 축 미포함)를 사용하십시오.

__출력 크기__

입력 크기와 동일합니다. 

__참고 자료__

- [Batch Normalization: Accelerating Deep Network Training by
   Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
