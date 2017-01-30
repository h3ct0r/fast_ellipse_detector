# fast_ellipse_detector
This is the implementation in C++ of the paper: 'A fast and effective ellipse detector for embedded vision applications'.
It runs on Ubuntu 16.04.  

Original author: mikispace (https://sourceforge.net/projects/yaed/)

### How to compile:

```sh
g++ Main.cpp EllipseDetectorYaed.cpp common.cpp -o ellipse_det -std=c++11 `pkg-config --cflags --libs opencv`
```

### How to run:

```sh
./ellipse_det
```