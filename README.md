# Description

SmoothUV is a spatial derainbow filter.

The luma is returned unchanged.

This is [a port of the VapourSynth plugin SmoothUV](https://github.com/dubhater/vapoursynth-smoothuv).

The file name is SmoothUV2 but the function name is SmoothUV. The reason for changing the file name is that the original SmoothUV plugin has one more function SSHiQ that is not ported here.

# Usage

```
SmoothUV(clip, int "radius", int "threshold", bool "interlaced")
```

## Parameters:

- clip\
    A clip to process. It must have constant format and it must be 8 bit YUV(A).
    
- radius\
    Must be between 1 and 7.\
    Larger values smooth more.\
    Default: 3.
    
- threshold\
    Must be between 0 and 450.\
    Larger values smooth more.\
    Default: 270.
    
- interlaced\
    Determine if the frame is interlaced.\
    Default: auto detect based on Is.FiledBased().
