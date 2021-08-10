#pragma once

#include <algorithm>
#include <cmath>
#include <memory>

#include "avisynth.h"
#include "VCL2/vectorclass.h"

class SmoothUV2 : public GenericVideoFilter
{
    int radiusy, radiuscw, radiusch;
    int _thresholdY, _thresholdC;
    int strengthY, strengthC;    
    bool hqy, hqc;
    int _interlaced;
    bool has_at_least_v8;
    std::unique_ptr<uint16_t[]> divin;
    int64_t field_based;
    bool sse2, ssse3, sse41, avx2,avx512;

    template <bool interlaced, bool hqy, bool hqc>
    void smoothN_c(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
    void sum_pixels_c(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int);
    void sshiq_sum_pixels_c(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int strength);

    template <bool interlaced, bool hqy, bool hqc>
    void smoothN_SSE2(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
    void sum_pixels_SSE2(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int);
    void sshiq_sum_pixels_SSE2(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int strength);

    template <bool interlaced, bool hqy, bool hqc>
    void smoothN_SSSE3(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
    void sum_pixels_SSSE3(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int);
    void sshiq_sum_pixels_SSSE3(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int strength);

    template <bool interlaced, bool hqy, bool hqc>
    void smoothN_SSE41(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
    void sum_pixels_SSE41(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int);
    void sshiq_sum_pixels_SSE41(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int strength);

    template <bool interlaced, bool hqy, bool hqc>
    void smoothN_AVX2(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
    void sum_pixels_AVX2(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int);
    void sshiq_sum_pixels_AVX2(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int strength);

    template <bool interlaced, bool hqy, bool hqc>
    void smoothN_AVX512(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
    void sum_pixels_AVX512(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int);
    void sshiq_sum_pixels_AVX512(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int strength);

    void (SmoothUV2::* smooth)(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);

public:
    SmoothUV2(PClip _child, int radiusY, int radiusC, int thresholdY, int thresholdC, int strY, int strC, bool HQY, bool HQC, int interlaced, int opt, IScriptEnvironment* env);

    int __stdcall SetCacheHints(int cachehints, int frame_range) override
    {
        return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;
};
