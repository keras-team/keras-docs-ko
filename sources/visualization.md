
## 모델 시각화

케라스는 (`graphviz`를 사용해서) 케라스 모델을 플롯팅하기 위한 유틸리티 함수를 제공합니다.

아래 예시는 모델의 그래프를 플롯팅하고 그 결과를 파일로 저장합니다:
```python
from keras.utils import plot_model
plot_model(model, to_file='model.png')
```

`plot_model`는 네가지 인수를 전달받습니다:

- `show_shapes`(디폴트값은 False)는 결과의 모양을 그래프에 나타낼 것인지 조정합니다.
- `show_layer_names`(디폴트값은 True)는 레이어의 이름을 그래프에 나타낼 것인지 조정합니다.
- `expand_nested`(디폴트값은 False)는 중첩된 모델을 그래프상에서 클러스터로 확장할 것인지 조정합니다.
- `dpi`(디폴트값은 96)는 이미지 dpi를 조정합니다.

또한 직접 `pydot.Graph`오브젝트를 만들어 사용할 수도 있습니다, 
예를들어 ipython notebook에 나타내자면 :
```python
from IPython.display import SVG
from keras.utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
```

## Training history visualization

The `fit()` method on a Keras `Model` returns a `History` object. The `History.history` attribute is a dictionary recording training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable). Here is a simple example using `matplotlib` to generate loss & accuracy plots for training & validation:

```python
import matplotlib.pyplot as plt

history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```
