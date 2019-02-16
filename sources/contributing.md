# On Github Issues and Pull Requests

Found a bug? Have a new feature to suggest? Want to contribute changes to the codebase? Make sure to read this first.

## Bug reporting

Your code doesn't work, and you have determined that the issue lies with Keras? Follow these steps to report a bug.

1. Your bug may already be fixed. Make sure to update to the current Keras master branch, as well as the latest Theano/TensorFlow/CNTK master branch.
To easily update Theano: `pip install git+git://github.com/Theano/Theano.git --upgrade`

2. Search for similar issues. Make sure to delete `is:open` on the issue search to find solved tickets as well. It's possible somebody has encountered this bug already. Also remember to check out Keras' [FAQ](http://keras.io/faq/). Still having a problem? Open an issue on Github to let us know.

3. Make sure you provide us with useful information about your configuration: what OS are you using? What Keras backend are you using? Are you running on GPU? If so, what is your version of Cuda, of cuDNN? What is your GPU?

4. Provide us with a script to reproduce the issue. This script should be runnable as-is and should not require external data download (use randomly generated data if you need to run a model on some test data). We recommend that you use Github Gists to post your code. Any issue that cannot be reproduced is likely to be closed.

5. If possible, take a stab at fixing the bug yourself --if you can!

The more information you provide, the easier it is for us to validate that there is a bug and the faster we'll be able to take action. If you want your issue to be resolved quickly, following the steps above is crucial.

---

## Requesting a Feature

You can also use Github issues to request features you would like to see in Keras, or changes in the Keras API.

1. Provide a clear and detailed explanation of the feature you want and why it's important to add. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing an add-on library for Keras. It is crucial for Keras to avoid bloating the API and codebase.

2. Provide code snippets demonstrating the API you have in mind and illustrating the use cases of your feature. Of course, you don't need to write any real code at this point!

3. After discussing the feature you may choose to attempt a Pull Request. If you're at all able, start writing some code. We always have more work to do than time to do it. If you can write some code then that will speed the process along.


---

## Requests for Contributions

[This is the board](https://github.com/keras-team/keras/projects/1) where we list current outstanding issues and features to be added. If you want to start contributing to Keras, this is the place to start.


---

## Pull Requests

**Where should I submit my pull request?**

1. **Keras improvements and bugfixes** go to the [Keras `master` branch](https://github.com/keras-team/keras/tree/master).
2. **Experimental new features** such as layers and datasets go to [keras-contrib](https://github.com/farizrahman4u/keras-contrib). Unless it is a new feature listed in [Requests for Contributions](https://github.com/keras-team/keras/projects/1), in which case it belongs in core Keras. If you think your feature belongs in core Keras, you can submit a design doc to explain your feature and argue for it (see explanations below).

Please note that PRs that are primarily about **code style** (as opposed to fixing bugs, improving docs, or adding new functionality) will likely be rejected.

Here's a quick guide to submitting your improvements:

1. If your PR introduces a change in functionality, make sure you start by writing a design doc and sending it to the Keras mailing list to discuss whether the change should be made, and how to handle it. This will save you from having your PR closed down the road! Of course, if your PR is a simple bug fix, you don't need to do that. The process for writing and submitting design docs is as follow:
    - Start from [this Google Doc template](https://docs.google.com/document/d/1ZXNfce77LDW9tFAj6U5ctaJmI5mT7CQXOFMEAZo-mAA/edit#), and copy it to new Google doc.
    - Fill in the content. Note that you will need to insert code examples. To insert code, use a Google Doc extension such as [CodePretty](https://chrome.google.com/webstore/detail/code-pretty/igjbncgfgnfpbnifnnlcmjfbnidkndnh?hl=en) (there are several such extensions available).
    - Set sharing settings to "everyone with the link is allowed to comment"
    - Send the document to `keras-users@googlegroups.com` with a subject that starts with `[API DESIGN REVIEW]` (all caps) so that we notice it.
    - Wait for comments, and answer them as they come. Edit the proposal as necessary.
    - The proposal will finally be approved or rejected. Once approved, you can send out Pull Requests or ask others to write Pull Requests.


2. Write the code (or get others to write it). This is the hard part!

3. 새로 만드신 함수나 클래스는 제대로 된 독스트링을 갖춰야 합니다. 작업하신 모든 코드에는 최신화된 독스트링과 사용설명서가 있어야 합니다. **독스트링 스타일을 지켜주십시오.** 특히 MarkDown 포맷으로 작업해야 하고, (가능한 경우) `Arguments`, `Returns`, `Raises`의 섹션이 있어야 합니다. 코드베이스에 있는 다른 독스트링을 참고해 주십시오.

4. 테스트를 작성해 주십시오. 작성하신 코드는 전 부분에 걸쳐 유닛 테스트를 갖춰야 합니다. 이는 PR을 신속하게 병합하려면 꼭 필요합니다.

5. 저희 테스트슈트를 로컬컴퓨터에 작동시키십시오. 어렵지 않습니다: 케라스 폴더에서, 다음을 실행하면 됩니다: `py.test tests/`.
    - 또한 테스트 필요요소도 설치하셔야 합니다: `pip install -e .[tests]`.

6. 다음 조건에서 테스트가 전부 통과되는지 확인해 주십시오:
    - Theano 백엔드, Python 2.7과 Python 3.6. Theano가 개발 버전인지 확인하십시오.
    - TensorFlow 백엔드, Python 2.7과 Python 3.6. TensorFlow가 개발 버전인지 확인하십시오.
    - CNTK 백엔드, Python 2.7과 Python 3.6. CNTK가 개발 버전인지 확인하십시오.

7. 저희는 PEP8 문법 규칙을 사용합니다만, 각 라인의 길이에 대해서는 크게 신경쓰지 않습니다. 그래도 각 라인의 길이가 적당한지는 확인해 주십시오. 보다 손쉬운 방법으로, PEP8 linter를 사용하시는 것을 추천드립니다:
    - PEP8 packages를 설치합니다: `pip install pep8 pytest-pep8 autopep8`
    - 독립형 PEP8 check을 실행합니다: `py.test --pep8 -m pep8`
    - 몇몇 PEP8 에러는 다음을 실행해서 자동적으로 수정할 수 있습니다: `autopep8 -i --select <errors> <FILENAME>` 예를 들면: `autopep8 -i --select E128 tests/keras/backend/test_backends.py`

8. 커밋 시 적절하고 자세한 커밋 메시지를 첨부해 주십시오.

9. 사용설명서를 최신화 시켜주십시오. 새로운 기능을 도입한다면, 그 새로운 기능의 사용처를 보여주는 코드를 포함시켜주시기 바랍니다.

10. PR을 제출하십시오. 이전 논의에서 요구하신 변경사항이 승인되었거나 적절한 독스트링/사용설명서와 함께 완전한 유닛테스트를 갖추셨다면, PR이 신속하게 병합될 가능성이 높습니다.

---

## Adding new examples

Even if you don't contribute to the Keras source code, if you have an application of Keras that is concise and powerful, please consider adding it to our collection of examples. [Existing examples](https://github.com/keras-team/keras/tree/master/examples) show idiomatic Keras code: make sure to keep your own script in the same spirit.
