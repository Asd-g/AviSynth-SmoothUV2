##### 4.3.1:
    Added SSSE3 code.

##### 4.3.0:
    Added SSE2 code.
    Parameter str replaced by strY and strC.

##### 4.2.1:
    Improved speed for opt=0.

##### 4.2.0:
    Added C++ code.
    Added paramters opt and dither.
    Changed internal calculation to 16-bit.
    Dropped support for AviSynth 2.x.

##### 4.1.0:
    SmoothUV2 threshold changed back to 0..450 range.
    A bit improved calculation for 8-bit.
    Added SSHiQ2 function: updated version of SSHiQ from SmoothUV v1.4.0:
    - replaced MMX asm code with SSE4.1 intrinsics;
    - fixed planes shift;
    - added support for 10..16-bit;
    - added support for 422/444 chroma subsampling;
    - more precise 8-bit calculation.

##### 4.0.0:
    Function name changed to SmoothUV2.
    Fixed horizontal planes shift.
    Fixed vertical planes shift for interlaced frames.
    Fixed output for 422/444 subsampling.
    Additional limit radius for 422/444.

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
