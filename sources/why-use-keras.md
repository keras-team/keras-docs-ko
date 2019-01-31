# 왜 Keras일까요?

오늘날 존재하는 수많은 딥러닝 프레임워크 중에서, 왜 굳이 Keras일까요? 다른 대안들에 비해 Keras를 선호하는 이유는 다음과 같습니다.

---

## Keras는 사용자 친화적입니다
    
- Keras는 기계가 아닌 사람을 위한 도구입니다. Keras는 [사용자의 부담을 덜기 위해](https://blog.keras.io/user-experience-design-for-apis.html) 일관되고 간결한 API를 제공하며, 일반적인 유스케이스에 필요한 사용자의 조작을 최소화 하고, 오작동에 대한 명확하고 실용적인 피드백을 제공합니다.

- Keras의 이런 개발 철학 덕분에 Keras는 배우기도, 사용하기에도 쉽습니다. Keras를 통해서 더 많은 아이디어를 빠르게 시도해 볼 수 있고, 이는 [머신러닝 대회에서 좋은 성적을 거둘 수 있도록 도와줍니다](https://www.quora.com/Why-has-Keras-been-so-successful-lately-at-Kaggle-competitions).

- Keras는 쉬운 고수준의 API를 제공하면서도, TensorFlow와 같은 저수준의 API와도 호환이 잘 되어 어떠한 네트워크 구조도 만들 수 있게 합니다. 특히, `tf.keras`를 사용하면 TensorFlow 기반의 작업 흐름에도 매끄럽게 통합시킬 수 있습니다.

---

## Keras는 업계와 학계 양쪽에서 모두 폭넓게 사용되고 있습니다

<a href='https://towardsdatascience.com/deep-learning-framework-power-scores-2018-23607ddf297a'>
    <img style='width: 80%; margin-left: 10%;' src='https://s3.amazonaws.com/keras.io/img/dl_frameworks_power_scores.png'/>
</a>
<p style='font-style: italic; font-size: 10pt; text-align: center;'>
    7개의 분류에 걸친 11개의 데이터 소스를 기반으로 계산된 딥러닝 프레임워크 순위, Jeff Hale.
</i>

Keras는 250,000명 이상의 개인 사용자(2018년 기준)를 기반으로 TensorFlow를 제외한 그 어떤 딥러닝 프레임워크보다 업계와 학계 모두에 깊게 배어있습니다. 또한 Keras API는 `tf.keras` 모듈을 통해 TensorFlow의 공식 프론트엔드로 사용되고 있습니다.

Keras를 통해 개발된 기능들은 Netflix, Uber, Yelp, Instacart, Zocdoc, Square사 등의 서비스에서 쉽게 찾아볼 수 있습니다. 이는 특히 딥러닝을 서비스의 핵심으로 삼는 스타트업 기업들 사이에서 인기가 많습니다.

Keras는 [arXiv.org](https://arxiv.org/archive/cs)에 업로드 된 과학 논문들 중에서 두 번째로 많이 언급 될 정도로 딥러닝 연구자들에게 사랑받고 있습니다. Keras는 또한 CERN과 NASA와 같은 대형 연구소에서도 채택된 도구입니다.

---

## Keras는 모델의 제품화를 쉽게 해줍니다

Keras는 다른 어떤 딥러닝 프레임워크보다도 다양한 방면의 플랫폼에 쉽게 배치할 수 있습니다. 이에 해당하는 플랫폼들은 다음과 같습니다.

- iOS에서는 [Apple’s CoreML](https://developer.apple.com/documentation/coreml)을 통해서 가능합니다. Apple사는 공식적으로 Keras를 지원합니다 ([튜토리얼](https://www.pyimagesearch.com/2018/04/23/running-keras-models-on-ios-with-coreml/)). 
- Android에서는 TensorFlow Android 런타임을 통해서 가능합니다.
- 웹 브라우저에서는 [Keras.js](https://transcranial.github.io/keras-js/#/)와 같은 GPU 가속된 JavaScript 런타임과 [WebDNN](https://mil-tokyo.github.io/webdnn/)을 통해서 가능합니다.
- Google Cloud에서는 [TensorFlow-Serving](https://www.tensorflow.org/serving/)을 통해서 가능합니다.
- [Flask app과 같은 Python 웹 백엔드](https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html)에서도 가능합니다.
- JVM에서 [SkyMind가 제공하는 DL4J model import](https://deeplearning4j.org/model-import-keras)를 통해서 가능합니다.
- Raspberry Pi에서도 가능합니다.

---

## Keras는 여러 백엔드 엔진을 지원하여 하나의 생태계에 속박되지 않습니다

Keras 모델은 여러 [딥러닝 백엔드](https://keras.io/backend/)를 지원합니다. 눈여겨볼 만한 점은, 내장 레이어로만 구성된 Keras 모델들은 지원하는 모든 백엔드들과 호환이 되어 학습에 사용되는 백엔드와 배포 등을 위한 로드에 사용되는 백엔드가 서로 달라도 된다는 것입니다. 사용 가능한 백엔드들은 다음과 같습니다.

- TensorFlow 백엔드 (Google사 제공)
- CNTK 백엔드 (Microsoft사 제공)
- Theano 백엔드

Amazon사는 MXNet을 백엔드로 사용하는 [Keras의 분기 버전](https://github.com/awslabs/keras-apache-mxnet)을 제공합니다.

결과적으로 Keras 모델들은 CPU뿐만이 아닌 다른 여러 하드웨어 플랫폼들에서도 학습이 가능합니다.

- [NVIDIA GPUs](https://developer.nvidia.com/deep-learning)
- [Google TPUs](https://cloud.google.com/tpu/) (TensorFlow 백엔드와 Google Cloud를 통해서)
- AMD사의 OpenCL과 호환되는 GPU ([PlaidML Keras 백엔드](https://github.com/plaidml/plaidml)를 통해서)

---

## Keras는 다중 GPU와 학습의 분산처리를 지원합니다

- Keras는 [다중 GPU 데이터 병렬성에 대한 지원이 내장되어있습니다](/utils/#multi_gpu_model).
- Uber사의 [Horovod](https://github.com/uber/horovod)는 케라스 모델을 일차적으로 지원합니다.
- Keras 모델을 [TensorFlow 추정자로 변환](https://www.tensorflow.org/versions/master/api_docs/python/tf/keras/estimator/model_to_estimator)이 가능하며, [Google Cloud를 통한 GPU 다발](https://cloud.google.com/solutions/running-distributed-tensorflow-on-compute-engine)에서 학습시킬 수 있습니다.
- [Dist-Keras](https://github.com/cerndb/dist-keras)와 [Elephas](https://github.com/maxpumperla/elephas)를 통해 Spark에서 Keras를 실행할 수 있습니다.

---

## Keras의 개발은 딥러닝 생태계의 주요 기업들의 지원을 받습니다

Keras는 Google사의 지원을 중심으로 개발되고 있으며, Keras API는 `tf.keras`로 TensorFlow의 패키지로 제공됩니다. CNTK Keras 백엔드의 유지보수 또한 Microsoft사의 책임하에 이루어집니다. Amazon AWS는 MXNet과 함께 Keras를 관리합니다. NVIDIA, Uber, CoreML을 포함한 Apple사 또한 Keras의 개발에 공헌하였습니다.

<img src='https://keras.io/img/google-logo.png' style='width:200px; margin-right:15px;'/>
<img src='https://keras.io/img/microsoft-logo.png' style='width:200px; margin-right:15px;'/>
<img src='https://keras.io/img/nvidia-logo.png' style='width:200px; margin-right:15px;'/>
<img src='https://keras.io/img/aws-logo.png' style='width:110px; margin-right:15px;'/>
