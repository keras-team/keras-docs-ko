# Scikit-Learn API의 래퍼

`keras.wrappers.scikit_learn.py`의 래퍼를 통해 `Sequential` 케라스 모델을 (단일 입력에 한정하여) Scikit-Learn 작업의 일부로 사용할 수 있습니다.

두 가지 래퍼가 이용가능합니다

`keras.wrappers.scikit_learn.KerasClassifier(build_fn=None, **sk_params)`는 Scikit-Learn 분류 인터페이스를 시행하고,

`keras.wrappers.scikit_learn.KerasRegressor(build_fn=None, **sk_params)`는 Scikit-Learn 회귀 인터페이스를 시행합니다.

### 인자

- __build_fn__: 호출가능한 함수 혹은 클래스 인스턴스
- __sk_params__: 모델 생성 및 학습에 사용되는 매개변수(아래 `sk_params`설명 참조) 

`build_fn`은 케라스 모델을 생성하고, 컴파일하고, 반환하여, 
모델이 학습/예측할 수 있도록 합니다.
`build_fn`은 다음의 세 가지 값 중 하나를 전달받습니다.

1. 함수
2. `__call__` 메소드를 시행하는 클래스의 인스턴스
3. 비워두기. 이는 `KerasClassifier` 혹은 `KerasRegressor`를 상속받는 클래스를
만들어야 함을 뜻합니다. 이 경우 현재 클래스의 `__call__` 메소드가
기본 `build_fn`이 됩니다.

`sk_params`는 모델 매개변수와 조정 매개변수 둘 모두 전달받습니다.
유효한 모델 매개변수는 `build_fn`의 인자입니다.
`build_fn`은 scikit-learn의 다른 에스티메이터처럼 의무적으로 인자에 대한
기본값을 넣도록 하여, `sk_params`에 따로 값을 전달하지 않고 에스티메이터를 만들 수 있도록 한다는 점을
참고하십시오.

`sk_params`는 또한 `fit`, `predict`, `predict_proba`, 그리고 `score` 메소드를
호출하는데 필요한 매개변수를 전달받습니다(예시: `epochs`, `batch_size`).
조정(예측) 매개변수는 다음과 같은 순서로 선택됩니다.

1. `fit`, `predict`, `predict_proba`, and `score` 메소드에 
등록된 값들
2. `sk_params`에 전달되는 값
3. `keras.models.Sequential`, `fit`, `predict`, `predict_proba`, `score` 메소드의
 기본값

scikit-learn의 `grid_search` API를 사용하는 경우, `sk_params`에 전달되는 모델 생성 및 학습에 사용되는 매개변수는 
조정 가능한 매개변수 입니다.
다시 말해, `grid_search`를 사용하여
최적의 `batch_size`나 `epochs`, 그리고 모델 매개변수를 찾아낼 수 있습니다.
