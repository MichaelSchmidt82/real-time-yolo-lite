# eyeball-sam
![](https://github.com/MichaelSchmidt82/eyeball-sam/blob/main/content/cool_demo.gif)

What: **Real-time object detection, person detection, and face recognition using YOLOv7 in TensorFlow Lite targeted for embedded devices with Google Coral hardware.**

## Requirements:
### Software
- 🖥️ Ubuntu 20.04
- 🐍️ Python 3.8
- 📦️ See requirements.txt, there are a lot.
- 📷️ It is recommend to [build OpenCV from source](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) for local testing (or just in general).  Use version 4.7.0.68.
### Hardware
- 🌊️ [Google Coral](https://coral.ai/). They have low-wattage USB and M.2 TPUs. A must for real-time video processing.

Usage:
1. Create a virtual environment and `pip install -r requirements.txt`.
2. Run the `create_tf_lite.ipynb` notebook to download use the model weights. This notebook will convert ONNX format to tf-lite.
3. Run `tfl_yolov7_main.py`.

Note: by default, openCV will use your wedcam (`cv2.VideoCapture(0)`)

This project was updated on 01/29/2024
