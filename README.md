# Description

SmoothUV2 is a spatial derainbow filter.

The luma is returned unchanged.

This is [a port of the VapourSynth plugin SmoothUV](https://github.com/dubhater/vapoursynth-smoothuv).

# Requirement

CPU with minimum supported instructions SSE4.1.

# Usage

```
SmoothUV2 (clip, int "radius", int "threshold", int "interlaced")
```

## Parameters:

- clip\
    A clip to process. It must be in YUV 8..16-bit planar format and must have at least three planes.
    
- radius\
    Must be between 1 and 7 for 4:2:0 subsampling and between 1 and 3 for 4:2:2/4:4:4 subsampling.\
    Larger values smooth more.\
    Default: 3.
    
- threshold\
    Must be between 0 and 255.\
    Larger values smooth more.\
    Default: 150.
    
- interlaced\
    Whether the frame is interlaced.\
    -1: If frame properties are supported and frame property "_FieldBased" exists - "_FieldBased" value is used.\
    If frame properties aren't supported or there is no property "_FieldBased" - 0.\
    0: Progressive frame.\
    1: Interlaced frame.\
    Default: -1.

# Building

## Windows

Use solution files.

## Linux

### Requirements

- Git
- C++17 compiler
- CMake >= 3.16

```
git clone https://github.com/Asd-g/AviSynth-SmoothUV2 && \
cd AviSynth-SmoothUV2 && \
mkdir build && \
cd build && \
cmake .. && \
make -j$(nproc) && \
sudo make install
```
