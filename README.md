# real-time-yolo-lite

What: A home grown real-time object detection, face detection, and person recognition for embedded devices using YOLOv7 among other things.

** insert cool GIF here **

## Requirements:
### Software
- 🖥️ Ubuntu 20.04
- 🐍️ Python 3.8
- 📦️ See requirements.txt, there are a lot.
- 📷️ It is recommend to [build OpenCV from source](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) for local testing (or just in general).  Use version 4.7.0.68.
### Hardware
- 🌊️ [Google Coral](https://coral.ai/). They have low wattage USB and M.2 TPUs.  This is needed. ~for embedded devices~.


### Some TODO's
- I need a TPU.  The chip shortage is real.
- Early research suggests building `dlib` from source.  This will definatly need to happen as face recognition relies on `dlib`.
- This project needs better organization, and I will begin to use a `develop` branch for a testing ground.

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
