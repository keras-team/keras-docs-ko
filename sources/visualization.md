
## Model visualization

Keras provides utility functions to plot a Keras model (using `graphviz`).

This will plot a graph of the model and save it to a file:
```python
from keras.utils import plot_model
plot_model(model, to_file='model.png')
```

`plot_model` takes four optional arguments:

- `show_shapes` (defaults to False) controls whether output shapes are shown in the graph.
- `show_layer_names` (defaults to True) controls whether layer names are shown in the graph.
- `expand_nested` (defaults to False) controls whether to expand nested models into clusters in the graph.
- `dpi` (defaults to 96) controls image dpi.

You can also directly obtain the `pydot.Graph` object and render it yourself,
for example to show it in an ipython notebook :
```python
from IPython.display import SVG
from keras.utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
```

## 학습 히스토리 시각화

케라스 `Model`의 `fit()` 메소드는 `History` 오브젝트를 반환합니다. `History.history` 속성은 연속된 에폭에 걸쳐 학습 손실 값과 학습 측정항목값을 기록하는 딕셔너리로, 또한 (적용 가능한 경우에는) 검증 손실 값과 검증 측정항목 값도 기록합니다. 아래 간단한 예시에서는 `matplotlib`을 사용하여 학습과 검증에 대한 손실과 정확성 플롯을 만듭니다:

```python
import matplotlib.pyplot as plt

history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

# 학습 정확성 값과 검증 정확성 값을 플롯팅 합니다. 
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 학습 손실 값과 검증 손실 값을 플롯팅 합니다.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```
