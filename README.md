# real-time-yolo-lite
A home grown real-time object detection for embedded devices using YOLOv7

### Requirements:

- 🐍️ Python 3.8
- 📦️ See requirements.txt, there are a lot.
- 📷️ It is recommend to [build OpenCV from source](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) for local testing (or just in general).  Use version 4.7.0.68.
- 🌊️ It is recommend to use [Google coral](https://coral.ai/). They have low wattage USB and M.2 TPUs.  This is needed for embedded devices.

### Common Problems
Different setups create unique situations.  Here are some common ones.

- **My CPU doesn't have AVX and TensorFlow does not start.**

    You can checkout some [community builds](https://github.com/yaroslavvb/tensorflow-community-wheels/issues) download and `pip install path/to/wheel`.  I used the "Linux: no AVX, no GPU, with SSE4, Ubuntu 22.04, Python 3.8" and got lucky.

- **My pip install complains about Protobuf versions.**

    So far, this conflict is unavoidable.  It has not prevented the ONNX -> TF-lite conversion.

- **My `export.py` process fails.**

    Some packages are still not listed in the requirements.txt, and you must manually install them.  Some common suspects are:
    - pandas
    - torchvision
    - seaborn
    - tensorflow_probability


