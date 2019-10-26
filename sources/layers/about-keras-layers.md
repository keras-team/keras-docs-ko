# 케라스 층에 관하여

케라스의 모든 층<sub>layer</sub>들은 공통적으로 아래의 메소드들을 가지고 있습니다.

- `layer.get_weights()`: NumPy형 배열<sub>array</sub>의 리스트 형태로 해당 층의 가중치<sub>weight</sub>들을 반환합니다.
- `layer.set_weights(weights)`: NumPy형 배열의 리스트를 가져와 해당 층의 가중치로 설정합니다 (`get_weights`의 출력과 동일한 크기를 가져야 합니다).
- `layer.get_config()`: 해당 층의 설정이 저장된 딕셔너리를 반환합니다. 모든 층은 다음과 같이 설정 내용을 가져오는 방식으로 다시 인스턴스화 될 수 있습니다.

```python
layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.from_config(config)
```

또는

```python
from keras import layers

config = layer.get_config()
layer = layers.deserialize({'class_name': layer.__class__.__name__,
                            'config': config})
```

만약 어떤 층이 단 하나의 노드만을 가지고 있다면 (즉, 다중 입력에 여러 번 호출되어 공유된 층이 아닌 경우), 입력 텐서, 출력 텐서, 입력 크기 및 출력 크기를 다음과 같이 얻어올 수 있습니다.

- `layer.input`
- `layer.output`
- `layer.input_shape`
- `layer.output_shape`

만약 해당 층이 여러 개의 노드들을 가지고 있다면 ([노드와 공유된 층의 개념](/getting-started/functional-api-guide/#"노드"의-개념) 참조), 다음의 메소드들을 사용할 수 있습니다.

- `layer.get_input_at(node_index)`
- `layer.get_output_at(node_index)`
- `layer.get_input_shape_at(node_index)`
- `layer.get_output_shape_at(node_index)`
