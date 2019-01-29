# 왜 케라스일까요?

수많은 심층학습 프레임워크 중에서 왜 굳이 케라스일까요? 다른 대안들에 비하여 케라스를 선호하는 이유는 다음과 같습니다.

---

## 케라스는 사용자 친화적입니다.
    
- 케라스는 기계가 아닌 사람을 위한 도구입니다. 케라스는 [사용자의 부담을 덜기 위하여](https://blog.keras.io/user-experience-design-for-apis.html) 일관되고 간결한 API를 제공하고 일상적인 경우에 대한 사용자의 조작을 최소화하며 오작동에 대한 명확하고 실용적인 피드백을 주는 등 사용자 친화적인 인터페이스를 제공합니다.

- 케라스의 이런 개발 철학 덕분에 케라스는 배우기도, 사용하기에도 쉽습니다. 케라스를 통해서 더욱 효율적이고 생산적인 아이디어들을 구현해낼 수 있고, 이는 [머신러닝 대회에서 좋은 성적을 거둘 수 있도록 도와줍니다](https://www.quora.com/Why-has-Keras-been-so-successful-lately-at-Kaggle-competitions).

- 사용의 편리함은 유용함을 희생시킨 결과물이 아닙니다. 케라스는 텐서플로우와 같은 저계층 심층학습 언어를 기반으로 하기 때문에 기반 언어로 구현할 수 있는 모든 것 사용할 수 있도록 합니다. 특히, `tf.keras`와 같은 케라스 API는 텐서플로우 기반 작업의 흐름에 매끄럽게 통합시킬 수 있습니다.

---

## 케라스는 업계와 학계에 포괄적으로 적용됩니다.

<a href='https://towardsdatascience.com/deep-learning-framework-power-scores-2018-23607ddf297a'>
    <img style='width: 80%; margin-left: 10%;' src='https://s3.amazonaws.com/keras.io/img/dl_frameworks_power_scores.png'/>
</a>
<p style='font-style: italic; font-size: 10pt; text-align: center;'>
    7개의 분류에 걸친 11개의 데이터 소스를 기반으로 계산된 심층학습 프레임워크 순위, Jeff Hale.
</i>

케라스는 250,000명 이상의 개인 사용자(2018년 기준)를 기반으로 텐서플로우를 제외한 그 어떤 심층학습 프레임워크보다 업계와 학계 모두에 깊게 배어있습니다. 또한 케라스 API는 `tf.keras` 모듈을 통해 텐서플로우의 공식 프론트엔드로 사용됩니다.

케라스를 통해 개발된 기능들은 Netflix, Uebr, Yelp, Instacart, Zocdoc, Square사 등의 서비스에서 쉽게 찾아볼 수 있습니다. 이는 특히 심층학습을 서비스의 핵심으로 삼는 스타트업 기업들 사이에서 인기가 많습니다.

케라스는 [arXiv.org](https://arxiv.org/archive/cs)에 업로드 된 과학 논문 중에서 두번째로 많이 언급 될 정도로 심층학습 연구자들에게 사랑받습니다. 케라스는 또한 CERN과 NASA와 같은 대형 연구소에서 채택된 도구입니다.

---

## 케라스는 모델의 제품화를 쉽게 해줍니다.

케라스는 다른 어떤 심층학습 프레임워크보다도 다양한 방면의 플랫폼에 쉽게 배치할 수 있습니다. 이에 해당하는 플랫폼들은 다음과 같습니다.

- iOS에서는 [Apple’s CoreML](https://developer.apple.com/documentation/coreml)을 통해서 가능합니다. Apple사는 공식적으로 케라스를 지원합니다([튜토리얼](https://www.pyimagesearch.com/2018/04/23/running-keras-models-on-ios-with-coreml/)). 
- Android에서는 Tensorflow Android runtime 을 통해서 가능합니다.
- 웹 브라우저에서는 [Keras.js](https://transcranial.github.io/keras-js/#/)와 같은 GPU 가속된 JavaScript runtime 과 [WebDNN](https://mil-tokyo.github.io/webdnn/)을 통해서 가능합니다.
- Google Cloud에서는 [TensorFlow-Serving]()https://www.tensorflow.org/serving/을 통해서 가능합니다.
- [Flask app과 같은 파이썬 웹 백엔드](https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html)에서도 가능합니다.
- JVM에서 [SkyMind가 제공하는 DL4J model import](https://deeplearning4j.org/model-import-keras)를 통해서 가능합니다.
- Raspberry Pi에서도 가능합니다.

---

## 케라스는 여러 백엔드 엔진을 지원하여 하나의 생태계에 속박되지 않습니다.

케라스 모델은 여러 [심층학습 백엔드](https://keras.io/backend/)들로 개발할 수 있습니다. 여기서 눈여겨 볼만한 점은 일부 내장 레이어의 보조가 되는 케라스 모델이라도 여러 백엔드에 적용이 가능합니다. 특정 모델을 훈련시키는 백엔드와 모델을 는 백엔드가 서로 달라도 됩니다. 사용 가능한 백엔드들은 다음과 같습니다.

- 텐서플로우 백엔드 (Google사 제공)
- CNTK 백엔드 (Microsoft사 제공)
- 테아노 백엔드

Amazon사는 MXNet을 백엔드로 사용하는 [케라스의 분기 버전](https://github.com/awslabs/keras-apache-mxnet)을 제공합니다.

결과적으로 케라스 모델들은 CPU뿐만이 아닌 다른 여러 하드웨어 플랫폼에서 학습이 가능합니다.

- [NVIDIA GPUs](https://developer.nvidia.com/deep-learning)
- [Google TPUs](https://cloud.google.com/tpu/) (텐서플로우 백엔드와 Google 클라우드를 통해서)
- AMD사의 OpenCL과 호환되는 GPU ([PlaidML 케라스 백엔드](https://github.com/plaidml/plaidml)를 통해서)

---

## 케라스는 다중 GPU와 학습의 분산처리를 지원합니다.

- 케라스는 [다중 GPU 데이터 병렬성에 대한 지원이 내장되어있습니다](/utils/#multi_gpu_model).
- Uber사의 [Horovod](https://github.com/uber/horovod)는 케라스 모델을 일차적으로 지원합니다.
- 케라스 모델을 [텐서플로우 추정자로 변환](https://www.tensorflow.org/versions/master/api_docs/python/tf/keras/estimator/model_to_estimator)이 가능하며, [Google 클라우드를 통한 GPU 다발](https://cloud.google.com/solutions/running-distributed-tensorflow-on-compute-engine)에서 학습시킬 수 있습니다.
- [Dist-Keras](https://github.com/cerndb/dist-keras)와 [Elephas](https://github.com/maxpumperla/elephas)를 통해 Spark에서 케라스를 실행할 수 있습니다.)

---

## 케라스의 개발은 심층학습 생태계의 주요 기업들의 지원을 받습니다.

케라스는 Google사의 지원을 중심으로 개발되고 있으며, 케라스 API는 `tf.keras`로 텐서플로우의 패키지로 제공됩니다. CNTK Keras 백엔드의 유지보수 또한 Microsoft사의 책임하에 이루어집니다. Amazon AWS는 MXNet과 함께 케라스의 를 관리합니다. NVIDIA, Uber, CoreML을 포함한 Apple사 또한 케라스의 개발에 공헌하였습니다.

<img src='https://keras.io/img/google-logo.png' style='width:200px; margin-right:15px;'/>
<img src='https://keras.io/img/microsoft-logo.png' style='width:200px; margin-right:15px;'/>
<img src='https://keras.io/img/nvidia-logo.png' style='width:200px; margin-right:15px;'/>
<img src='https://keras.io/img/aws-logo.png' style='width:110px; margin-right:15px;'/>
