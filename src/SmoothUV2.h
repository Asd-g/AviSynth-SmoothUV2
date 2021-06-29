#pragma once

#include <algorithm>
#include <cmath>
#include <memory>

#include "avisynth.h"

class SmoothUV2 : public GenericVideoFilter
{
    int _thresholdY, _thresholdC;
    int _strength;
    int radiusy, radiuscw, radiusch;
    bool hqy, hqc;
    int _interlaced;
    bool has_at_least_v8;
    std::unique_ptr<uint16_t[]> divin;
    int64_t field_based;
    bool sse41;

    template <bool interlaced, bool hqy, bool hqc>
    void smoothN_c(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
    void sum_pixels_c(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold);
    void sshiq_sum_pixels_c(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold);
    template <bool interlaced, bool hqy, bool hqc>
    void smoothN_SSE41(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
    void sum_pixels_SSE41(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold);
    void sshiq_sum_pixels_SSE41(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold);

    void (SmoothUV2::* smooth)(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);

public:
    SmoothUV2(PClip _child, int radiusY, int radiusC, int thresholdY, int thresholdC, int strength, bool HQY, bool HQC, int interlaced, int opt, IScriptEnvironment* env);

    int __stdcall SetCacheHints(int cachehints, int frame_range) override
    {
        return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;
};
