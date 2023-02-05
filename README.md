# YOLOv7 in TensorFlow lite

What: real-time object detection using YOLOv7 in TensorFlow Lite.  (Bonus face detection and recognition).  Ideal for embedded devices with Google Coral.

** insert cool GIF here **

## Requirements:
### Software
- 🖥️ Ubuntu 20.04
- 🐍️ Python 3.8
- 📦️ See requirements.txt, there are a lot.
- 📷️ It is recommend to [build OpenCV from source](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) for local testing (or just in general).  Use version 4.7.0.68.
### Hardware
- 🌊️ [Google Coral](https://coral.ai/). They have low wattage USB and M.2 TPUs.  This is ideal for embedded devices unless you have GPU support.


### Some TODO's
- I need a TPU to test on embedded evices.  (chip shortage).
- Face recognition relies on `dlib'.
- Color bounding boxes

### Common Problems
Different setups create unique situations.  Here are some common ones.

- **My CPU doesn't have AVX and TensorFlow does not start.**
    You can checkout some [community builds](https://github.com/yaroslavvb/tensorflow-community-wheels/issues) and download them.  You can then use `pip install path/to/wheel` to finish the install.  My [device](https://www.hardkernel.com/shop/odroid-h3-plus/) needed [this](https://github.com/yaroslavvb/tensorflow-community-wheels/issues/217) TF Nightly build. (it has no AVX, no GPU, with SSE4, Ubuntu 22.04, and Python 3.8." I got lucky, YMMV.

- **My pip install complains about Protobuf versions.**

    So far, this conflict is unavoidable.  It has not prevented the ONNX -> TF-lite conversion however.  In the future I hope resolve this. 

- **My `export.py` process fails.**

    Some packages are still not listed in the requirements.txt, and you must manually install them.  Some common suspects are:
    - pandas
    - torchvision
    - seaborn
    - tensorflow_probability
