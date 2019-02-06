# Keras 계층에 관하여

모든 Keras 계층들은 많은 공통적인 메소드를 가지고 있습니다:

- `layer.get_weights()`: Numpy형 배열의 리스트 형태로 계층의 가중치들을 반환 합니다.
- `layer.set_weights(weights)`: Numpy형 배열의 리스트로 계층의 가중치들을 설정 합니다. (`get_weights`의 출력과 동일한 shape 이어야 합니다)
- `layer.get_config()`: 계층의 설정이 저장된 딕셔너리를 반환 합니다. 계층은 다음과 같이 설정내용으로 부터 다시 인스턴스화 될 수 있습니다:

```python
layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.from_config(config)
```

또는:

```python
from keras import layers

config = layer.get_config()
layer = layers.deserialize({'class_name': layer.__class__.__name__,
                            'config': config})
```

계층이 단 하나의 노드만을 가지고 있다면 (예를 들어서, 공유된 계층이 아닌 경우), 입력 tensor, 출력 tensor, 입력 shape, 출력 shape을 다음과 같이 얻어올 수 있습니다:

- `layer.input`
- `layer.output`
- `layer.input_shape`
- `layer.output_shape`

계층이 여러개의 노드를 가지고 있다면 (참조: [계층 노드와 공유된 계층의 개념](/getting-started/functional-api-guide/#the-concept-of-layer-node)), 다음의 메소드들을 사용할 수 있습니다:

- `layer.get_input_at(node_index)`
- `layer.get_output_at(node_index)`
- `layer.get_input_shape_at(node_index)`
- `layer.get_output_shape_at(node_index)`
