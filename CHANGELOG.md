##### 3.0.1:
    Another attemp to fix last columns processing.
    Removed redunant interlaced value 2.

##### 3.0.0:
    Fixed not processed edges.
    Added support for 10..16-bit.
    Set MT mode: MT_NICE_FILTER.
    Changed the range of parameter threshold from 0..450 to 0..255.
    Changed the type of parameter interlaced from bool to int.
    Changed the minimum CPU instructions support from SSE2 to SSE4.1.
    Added Linux building option.

##### 2.1.2:
    Throw error for non-planar formats.

##### 2.1.1:
    Fixed memory misalignment for AviSynth 2.6.
    Fixed processing when interlaced and h % 2.

##### 2.1.0:
    Added support for v8 interface.
    
##### 2.0.0:
    Port of the VapourSynth plugin SmoothUV v2.