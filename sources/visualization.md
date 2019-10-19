## 모델 시각화

케라스는(`graphviz`를 사용해서) 케라스 모델을 그래프로 그리기 위한 유틸리티 함수를 제공합니다.  

아래 예시는 모델의 그래프를 그려주고 그 결과를 파일로 저장합니다.  
```python
from keras.utils import plot_model
plot_model(model, to_file='model.png')
```

`plot_model`은 네 가지 인자를 전달받습니다.

- `show_shapes`(기본값은 `False`)는 결과의 형태을 그래프에 나타낼 것인지 여부를 설정합니다.
- `show_layer_names`(기본값은 `True`)는 층의 이름을 그래프에 나타낼 것인지 여부를 설정합니다.
- `expand_nested`(기본값은 `False`)는 중첩된 모델을 그래프상에서 클러스터로 확장할 것인지 여부를 설정합니다.
- `dpi`(기본값은 96)는 이미지 dpi를 설정합니다.

또한 직접 `pydot.Graph`오브젝트를 만들어 사용할 수도 있습니다.  
아래는 IPython Notebook에서 나타낸 예시입니다.
```python
from IPython.display import SVG
from keras.utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
```

## 학습 히스토리 시각화

케라스 `Model`의 `fit()` 메소드는 `History` 오브젝트를 반환합니다. `History.history` 속성<sub>attribute</sub>은 각 에폭마다 계산된 학습 손실 및 평가 지표가 순서대로 기록된 딕셔너리입니다. 검증 데이터를 적용한 경우에는 해당 손실 및 지표도 함께 기록됩니다. 아래는 `matplotlib`을 사용하여 학습 및 검증의 손실과 정확도 그래프를 그리는 예시입니다.

```python
import matplotlib.pyplot as plt

history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

# 학습 정확도와 검증 정확도를 그래프로 나타냅니다. 
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 학습 손실값과 검증 손실값을 그래프로 나타냅니다.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```
