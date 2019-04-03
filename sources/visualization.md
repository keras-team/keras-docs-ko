
## 모델 시각화

Keras는 모델을 시각화할 수 있게 유틸리티 함수를 제공합니다(내부적으로 graphviz를 사용합니다).

다음 코드로 모델을 그래프화하고 파일로 저장할 수 있습니다:
```python
from keras.utils import plot_model
plot_model(model, to_file='model.png')
```

`plot_model` 함수에는 선택적으로 입력할 수 있는 4개의 인자가 있습니다:

- `show_shapes` (기본값 False) 출력 형태를 그래프로 보여줄지 여부
- `show_layer_names` (기본값 True) 레이어 이름을 보여줄지 여부
- `expand_nested` (기본값 False) 인접 모델을 클러스터로 확장할지 여부
- `dpi` (기본값 96) DPI 조절

`pydot.Graph` 객체로 여러분이 직접 그릴 수도 있습니다. 노트북 예제를 보이겠습니다 :
```python
from IPython.display import SVG
from keras.utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
```

## 훈련 히스토리 시각화


`Model`의 `fit()` 메서드는 `History` 객체를 리턴합니다. `History.history` 속성은 파이썬 딕셔너리로 형태인데 에포크에 따른 검증 손실값과 검증 메트릭값(획득가능할시)은 물론 훈련 손실값과 메트릭값을 알 수 있습니다. `matplotlib`을 사용하여 훈련과 검증시 손실과 정확도를 그려보겠습니다 :

```python
import matplotlib.pyplot as plt

history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

# 훈련과 검증 정확도 그리기
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 훈련과 검증의 손실값 그리기
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```
