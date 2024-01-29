# eyeball-sam

What: Real-time object detection, person detection, and face recognition using YOLOv7 in TensorFlow Lite targeted for embedded devices with Google Coral hardware.

![](https://github.com/MichaelSchmidt82/eyeball-sam/blob/main/content/cool_demo.gif)

## Requirements:
### Software
- 🖥️ Ubuntu 20.04
- 🐍️ Python 3.8
- 📦️ See requirements.txt, there are a lot.
- 📷️ It is recommend to [build OpenCV from source](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) for local testing (or just in general).  Use version 4.7.0.68.
### Hardware
- 🌊️ [Google Coral](https://coral.ai/). They have low-wattage USB and M.2 TPUs. A must for real-time video processing.

### Some TODOs
- I need a TPU to test on embedded devices.  (chip shortage).
- Face recognition relies on `dlib`.
- Color selection for bounding boxes, especially for face recognition is random and could use some polish.

### Common Problems
Different setups create unique situations.  Here are some common ones.

- **My CPU doesn't have AVX and TensorFlow does not start.**
    You can check out some [community builds](https://github.com/yaroslavvb/tensorflow-community-wheels/issues) and download them.  You can then use `pip install path/to/wheel` to finish the installation.  My personal [device](https://www.hardkernel.com/shop/odroid-h3-plus/) needed [this](https://github.com/yaroslavvb/tensorflow-community-wheels/issues/217) TF Nightly build. (it has no AVX, no GPU, with SSE4, Ubuntu 22.04, and Python 3.8." I got lucky, YMMV.

- **My pip install complains about Protobuf versions.**

    So far, this conflict is unavoidable.  It has not prevented the ONNX -> TF-lite conversion, however.  In the future, I hope to resolve this. 

- **My `export.py` process fails.**

    Some packages are still not listed in the requirements.txt, and you must manually install them.  Some common suspects are:
    - pandas
    - torch-vision
    - seaborn
    - tensorflow_probability
