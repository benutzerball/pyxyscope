# pyxyscope
Python module for drawing images on an oscilloscope in XY-mode using a PC sound card. Converts an image or text specified in a yaml file to a dithered black-and-white raster whose resolution is limited by the sample rate of your sound card. For a 96 kHz audio output you can expect 80x80 pixels with 50% black. That is just enough to render a face which is why pyxyscope can automatically find faces in images and crop out the rest.
## Getting Started
```
python3 pyxyscope.py default_config.yaml
```

Running this should show the following sequence on your scope when tuned right

<img src="deathstarpizzaexample.gif"><img>

### Prerequisites
* numpy
* cv2
* cvlib
* sounddevice
