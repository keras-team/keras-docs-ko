# 깃헙 이슈와 풀 리퀘스트에 관해서

버그를 찾으셨나요? 새로이 제안할 기능이 있습니까? 코드베이스에 기여를 하고 싶으신가요? 이 안내문을 먼저 읽어주십시오.

## 버그 리포트

작성하신 코드가 작동하지 않는게 케라스쪽의 문제라고 확신하시나요? 버그 리포트를 하려면 다음의 절차를 따라주십시오.

1. 버그가 벌써 수정되었을 수도 있습니다. 먼저 가장 최신의 케라스 마스터 브랜치와 Theano/TensorFlow/CNTK 마스터 브랜치로 업데이트하십시오.
Theano를 쉽게 업데이트 하려면: `pip install git+git://github.com/Theano/Theano.git --upgrade`

2. 비슷한 이슈를 찾아보십시오. 이슈를 찾을 때 `is:open`을 지워서 해결된 문제도 찾아보십시오. 다른 누군가 이미 이 버그를 해결했을 수도 있습니다. 케라스의 [FAQ](http://keras.io/faq/)를 확인하는 것도 잊지 마십시오. 아직도 문제가 해결되지 않았나요? 깃헙에 이슈를 열어 저희에게 알려주십시오.

3. 반드시 구성에 대해서 자세히 말씀해주십시오: 어떤 OS를 사용하십니까? 케라스 백엔드는 어떤 것을 사용하십니까? GPU에 돌리고 계시나요? 그렇다면 Cuda와 cuDNN의 버전은 어떻게 됩니까? GPU는 어느 제품을 사용하십니까?

4. 이슈를 재현할 스크립트를 제공해 주십시오. 스크립트는 그 자체로 작동해야 하며 외부 데이터 다운로드를 필요로 해서는 안됩니다(모델을 어느 테스트 데이터에 작동시켜야 한다면 임의로 만들어진 데이터를 사용해주십시오). 코드를 게시할 때는 Github Gists를 사용할 것을 추천드립니다. 재현 불가능한 이슈는 닫히게 됩니다.

5. 가능하다면 버그를 스스로 고쳐보십시오 --가능하다면요!

더 많은 정보를 제공해 주실수록, 저희가 버그의 유무를 확인하기 쉬워지며 더 빨리 조치를 취할 수 있습니다. 이슈가 빨리 해결되길 바라신다면, 위의 절차를 따라주시는 것이 중요합니다.

---

## 새 기능을 신청하려면

깃헙 이슈를 사용해서 케라스에 추가됐으면 하는 기능이나 케라스 API에 대한 변경사항을 요구할 수 있습니다.

1. 원하시는 기능과 왜 그 기능이 중요한지에 대해서 분명하고 자세하게 설명해주십시오. 저희는 소수가 아닌 다수의 사용자에게 유용한 기능을 고려한다는 점을 명심해 주십시오. 만약 특정 사용자만을 고려한다면, 애드온 라이브러리를 작성하시는 것을 추천드립니다. 케라스는 API와 코드베이스가 지나치게 방대해지는 것을 피하려고 합니다.

2. 염두하시는 API를 보여주고 새 기능의 사용처를 입증하는 코드를 제공해주십시오. 물론 이 시점에서 진짜 코드를 작성하실 필요까지는 없습니다!

3. 기능에 대해서 논의 후 풀 리퀘스트를 넣을 수도 있습니다. 가능하면 코드를 작성해 주십시오. 시간은 적고 일은 많기에, 코드를 작성해 주시면 프로세스를 좀 더 빠르게 진행할 수 있습니다.


---

## 케라스에 기여하려면

[케라스 게시판입니다](https://github.com/keras-team/keras/projects/1). 여기에 현재 주요한 이슈와 추가할 기능을 게시합니다. 케라스에 기여하려면 이 게시판에서 시작하시면 됩니다.


---

## 풀 리퀘스트

**풀 리퀘스트는 어디에 제출합니까?**

1. **케라스 개선과 버그해결** [케라스 `master` 브랜치](https://github.com/keras-team/keras/tree/master)로 가시면 됩니다.
2. **새로운 기능** [Requests for Contributions](https://github.com/keras-team/keras/projects/1)에 게시된 케라스의 코어에 관련한 새로운 기능이 아니라면, 레이어나 데이터셋에 관한 새로운 기능에 관련해서는 [keras-contrib](https://github.com/farizrahman4u/keras-contrib)로 가시면 됩니다.

(버그 해결, 설명서 개선, 혹은 새로운 기능 추가가 아닌) **코딩 스타일**에 관련된 풀 리퀘스트는 거부될 가능성이 높습니다.

개선사항을 제출하는 빠른 가이드라인입니다:

1. 기능상 변경사항에 관련된 PR의 경우, 디자인 설명서를 먼저 작성하고 케라스 메일 리스트에 전송해서 변경이 필요한지 여부와, 어떻게 변경할지를 논의해야합니다. 이 작업은 PR을 진행하고 나서 PR이 거절될 확율을 줄여줍니다! 물론 간단한 버그 수정에 관한 PR이라면 그럴 필요까지는 없습니다. 디자인 설명서를 작성하고 제출하는 절차는 다음과 같습니다:
    - [Google Doc template](https://docs.google.com/document/d/1ZXNfce77LDW9tFAj6U5ctaJmI5mT7CQXOFMEAZo-mAA/edit#)을 복사해서 새로운 Google doc에 옮겨주세요.
    - 내용을 채워 주십시오. 반드시 코드 예시를 넣어 주시기 바랍니다. 코드를 삽입하려면 [CodePretty](https://chrome.google.com/webstore/detail/code-pretty/igjbncgfgnfpbnifnnlcmjfbnidkndnh?hl=en)와 같은 Google Doc 익스텐션을 사용하시면 됩니다 (다른 비슷한 익스텐션도 사용가능합니다).
    - 공유 셋팅을 "링크가 있는 모든 사용자는 댓글을 달 수 있습니다"로 설정해주십시오.
    - 저희가 알기 쉽도록 `[API DESIGN REVIEW]`(전부 대문자)로 시작하는 제목을 달아 문서를 `keras-users@googlegroups.com`으로 보내주십시오.
    - 댓글을 기다렸다가 댓글이 달리면 대답해 주십시오. 필요한대로 제안안을 수정해주시면 됩니다.
    - 제안안은 궁극적으로 거부되거나 승인됩니다. 승인되면 PR을 보내시거나 다른 이들에게 PR을 작성할 것을 부탁하시면 됩니다.


2. 코드를 작성하십시오 (아니면 다른 이들이 작성토록 해주십시오). 여기가 어려운 부분입니다!

3. Make sure any new function or class you introduce has proper docstrings. Make sure any code you touch still has up-to-date docstrings and documentation. **Docstring style should be respected.** In particular, they should be formatted in MarkDown, and there should be sections for `Arguments`, `Returns`, `Raises` (if applicable). Look at other docstrings in the codebase for examples.

4. Write tests. Your code should have full unit test coverage. If you want to see your PR merged promptly, this is crucial.

5. Run our test suite locally. It's easy: from the Keras folder, simply run: `py.test tests/`.
    - You will need to install the test requirements as well: `pip install -e .[tests]`.

6. Make sure all tests are passing:
    - with the Theano backend, on Python 2.7 and Python 3.6. Make sure you have the development version of Theano.
    - with the TensorFlow backend, on Python 2.7 and Python 3.6. Make sure you have the development version of TensorFlow.
    - with the CNTK backend, on Python 2.7 and Python 3.6. Make sure you have the development version of CNTK.

7. We use PEP8 syntax conventions, but we aren't dogmatic when it comes to line length. Make sure your lines stay reasonably sized, though. To make your life easier, we recommend running a PEP8 linter:
    - Install PEP8 packages: `pip install pep8 pytest-pep8 autopep8`
    - Run a standalone PEP8 check: `py.test --pep8 -m pep8`
    - You can automatically fix some PEP8 error by running: `autopep8 -i --select <errors> <FILENAME>` for example: `autopep8 -i --select E128 tests/keras/backend/test_backends.py`

8. When committing, use appropriate, descriptive commit messages.

9. Update the documentation. If introducing new functionality, make sure you include code snippets demonstrating the usage of your new feature.

10. Submit your PR. If your changes have been approved in a previous discussion, and if you have complete (and passing) unit tests as well as proper docstrings/documentation, your PR is likely to be merged promptly.

---

## 예시를 새로 더하려면

케라스 소스코드에 기여하지 않더라도 명료하고 강력한 케라스 어플리케이션을 만드셨다면, 범례 컬렉션에 추가하시는 것을 고려해주십시오. [기존의 범례](https://github.com/keras-team/keras/tree/master/examples)에서 케라스 문법에 부합하는 코드를 보시고 동일한 스타일로 스크립트를 작성해 주시기 바랍니다.
