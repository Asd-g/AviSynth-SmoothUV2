#include <algorithm>
#include <smmintrin.h>
#include <string>
#include <cmath>

#include "avisynth.h"
#include "avs/minmax.h"
#include "avs/config.h"

class SmoothUV2 : public GenericVideoFilter
{
    int _thresholdY, _thresholdC;
    int _strength;
    int radiusy, radiuscw, radiusch;
    bool hqy, hqc;
    int _interlaced;
    bool has_at_least_v8;
    uint16_t divin[256];
    int64_t field_based;

    template <typename T, bool interlaced, int bits, bool hqy, bool hqc>
    void smoothN_SSE41(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);

    void (SmoothUV2::* smooth)(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);

public:
    SmoothUV2(PClip _child, int radiusY, int radiusC, int thresholdY, int thresholdC, int strength, bool HQY, bool HQC, int interlaced, IScriptEnvironment* env);

    int __stdcall SetCacheHints(int cachehints, int frame_range) override
    {
        return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;
};

AVS_FORCEINLINE __m128i scale_u16(const __m128i a, const float peak, const float peak1)
{
    return _mm_packus_epi32(_mm_cvtps_epi32(_mm_div_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(a, _mm_set1_epi16(0))), _mm_set_ps1(peak)), _mm_set_ps1(peak1))),
        _mm_cvtps_epi32(_mm_div_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(a, _mm_set1_epi16(0))), _mm_set_ps1(peak)), _mm_set_ps1(peak1))));
}

AVS_FORCEINLINE __m128i mul_u16(const __m128i a, const __m128i b, const float peak)
{
    return _mm_packus_epi32(_mm_cvtps_epi32(_mm_div_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(a, _mm_set1_epi16(0))), _mm_cvtepi32_ps(_mm_unpacklo_epi16(b, _mm_set1_epi16(0)))), _mm_set_ps1(peak))),
        _mm_cvtps_epi32(_mm_div_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(a, _mm_set1_epi16(0))), _mm_cvtepi32_ps(_mm_unpackhi_epi16(b, _mm_set1_epi16(0)))), _mm_set_ps1(peak))));
}

AVS_FORCEINLINE __m128i mul_u32(const __m128i a, const __m128i b, const float peak)
{
    return _mm_cvtps_epi32(_mm_div_ps(_mm_mul_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)), _mm_set_ps1(peak)));
}

template <typename T, int bits>
AVS_FORCEINLINE void sum_pixels_SSE41(const T* srcp, T* dstp, const int stride,
    const int diff, const int width, const int height,
    const int threshold,
    const uint16_t* divinp, const int strength)
{
    const __m128i zeroes = _mm_setzero_si128();
    __m128i sum = zeroes;
    __m128i count = zeroes;

    if constexpr (std::is_same_v<T, uint8_t>)
    {
        const __m128i thres = _mm_set1_epi16(static_cast<uint16_t>(llrint(sqrt((static_cast<double>(threshold) * threshold) / 3.0))));
        const __m128i center_pixel = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp)), zeroes);

        srcp = srcp - diff;

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const __m128i neighbour_pixel = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + x)), zeroes);

                const __m128i abs_diff = _mm_abs_epi16(_mm_sub_epi16(center_pixel, neighbour_pixel));

                // Absolute difference less than thres
                const __m128i mask = _mm_cmpgt_epi16(thres, abs_diff);

                // Sum up the pixels that meet the criteria
                sum = _mm_adds_epu16(sum,
                    _mm_and_si128(neighbour_pixel, mask));

                // Keep track of how many pixels are in the sum
                count = _mm_sub_epi16(count, mask);
            }

            srcp += stride;
        }

        sum = _mm_adds_epu16(sum, _mm_srli_epi16(count, 1));

        __m128i divres = [&]() {
            divres = _mm_insert_epi16(divres, divinp[_mm_extract_epi16(count, 0)], 0);
            divres = _mm_insert_epi16(divres, divinp[_mm_extract_epi16(count, 1)], 1);
            divres = _mm_insert_epi16(divres, divinp[_mm_extract_epi16(count, 2)], 2);
            divres = _mm_insert_epi16(divres, divinp[_mm_extract_epi16(count, 3)], 3);
            divres = _mm_insert_epi16(divres, divinp[_mm_extract_epi16(count, 4)], 4);
            divres = _mm_insert_epi16(divres, divinp[_mm_extract_epi16(count, 5)], 5);
            divres = _mm_insert_epi16(divres, divinp[_mm_extract_epi16(count, 6)], 6);
            return _mm_insert_epi16(divres, divinp[_mm_extract_epi16(count, 7)], 7);
        }();        

        const __m128i mul = scale_u16(mul_u16(sum, divres, 4095.0f), 255.0f, 4095.0f);

        // Store
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp), _mm_packus_epi16(mul, mul));
    }
    else
    {
        const __m128i thres = _mm_set1_epi32(static_cast<uint32_t>(llrint(sqrt((static_cast<double>(threshold) * threshold) / 3))));
        const __m128i center_pixel = _mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp)), zeroes);

        srcp = srcp - diff;

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const __m128i neighbour_pixel = _mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + x)), zeroes);

                const __m128i abs_diff = _mm_abs_epi32(_mm_sub_epi32(center_pixel, neighbour_pixel));

                // Absolute difference less than thres
                const __m128i mask = _mm_cmpgt_epi32(thres, abs_diff);

                // Sum up the pixels that meet the criteria
                sum = _mm_add_epi32(sum,
                    _mm_and_si128(neighbour_pixel, mask));

                // Keep track of how many pixels are in the sum
                count = _mm_sub_epi32(count, mask);
            }

            srcp += stride;
        }
        
        sum = _mm_add_epi32(sum, _mm_srli_epi32(_mm_slli_epi32(count, bits - 8), 1));

        __m128i divres_hi = [&]() {
            divres_hi = _mm_insert_epi32(divres_hi, divinp[_mm_extract_epi32(count, 0)], 0);
            return _mm_insert_epi32(divres_hi, divinp[_mm_extract_epi32(count, 1)], 1);
        }();

        __m128i divres_lo = [&]() {
            divres_lo = _mm_insert_epi32(divres_lo, divinp[_mm_extract_epi32(count, 2)], 0);
            return _mm_insert_epi32(divres_lo, divinp[_mm_extract_epi32(count, 3)], 1);
        }();

        __m128i sum_hi = [&]() {
            sum_hi = _mm_insert_epi32(sum_hi, _mm_extract_epi32(sum, 0), 0);
            return _mm_insert_epi32(sum_hi, _mm_extract_epi32(sum, 1), 1);
        }();

        __m128i sum_lo = [&]() {
            sum_lo = _mm_insert_epi32(sum_lo, _mm_extract_epi32(sum, 2), 0);
            return _mm_insert_epi32(sum_lo, _mm_extract_epi32(sum, 3), 1);
        }(); 
        
        const __m128i mul_hi = _mm_mul_epu32(_mm_shuffle_epi32(sum_hi, _MM_SHUFFLE(3, 1, 2, 0)), _mm_shuffle_epi32(divres_hi, _MM_SHUFFLE(3, 1, 2, 0)));
        const __m128i mul_lo = _mm_mul_epu32(_mm_shuffle_epi32(sum_lo, _MM_SHUFFLE(3, 1, 2, 0)), _mm_shuffle_epi32(divres_lo, _MM_SHUFFLE(3, 1, 2, 0)));

        const __m128i hi = _mm_unpacklo_epi16(mul_hi, zeroes);
        const __m128i hi1 = _mm_unpackhi_epi16(mul_hi, zeroes);
        const __m128i lo = _mm_unpacklo_epi16(mul_lo, zeroes);
        const __m128i lo1 = _mm_unpackhi_epi16(mul_lo, zeroes);

        __m128i result = [&]() {
            result = _mm_insert_epi32(result, _mm_extract_epi32(hi, 1), 0);
            result = _mm_insert_epi32(result, _mm_extract_epi32(hi1, 1), 1);
            result = _mm_insert_epi32(result, _mm_extract_epi32(lo, 1), 2);
            return _mm_insert_epi32(result, _mm_extract_epi32(lo1, 1), 3);
        }();
        
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp), _mm_packus_epi32(result, result));
    }
}

template <typename T, int bits>
AVS_FORCEINLINE void sshiq_sum_pixels_SSE41(const T* srcp, T* dstp, const int stride,
    const int diff, const int width, const int height,
    const int threshold,
    const uint16_t* divinp, const int strength)
{
    const __m128i zeroes = _mm_setzero_si128();
    __m128i sum = zeroes;
    __m128i count = zeroes;

    if constexpr (std::is_same_v<T, uint8_t>)
    {
        const __m128i thres = _mm_set1_epi16(static_cast<uint16_t>(llrint(sqrt((static_cast<double>(threshold) * threshold) / 3.0))));

        // Build edge values
        const __m128i center_pixel = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp)), zeroes);
        const __m128i add = _mm_adds_epu16(_mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + stride)), zeroes), _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + 1)), zeroes));
        const __m128i sllq = _mm_slli_epi16(center_pixel, 1);

        // Store weight with edge bias
        const __m128i str = _mm_subs_epu16(_mm_set1_epi16(strength), _mm_or_si128(_mm_subs_epu16(sllq, add), _mm_subs_epu16(add, sllq)));

        srcp = srcp - diff;

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const __m128i neighbour_pixel = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + x)), zeroes);
                const __m128i mask = _mm_cmpgt_epi16(thres, _mm_abs_epi16(_mm_sub_epi16(center_pixel, neighbour_pixel)));
                sum = _mm_adds_epu16(sum, _mm_and_si128(neighbour_pixel, mask));
                count = _mm_sub_epi16(count, mask);
            }

            srcp += stride;
        }

        sum = _mm_adds_epu16(sum, _mm_srli_epi16(count, 1));

        __m128i divres = [&]() {
            divres = _mm_insert_epi16(divres, divinp[_mm_extract_epi16(count, 0)], 0);
            divres = _mm_insert_epi16(divres, divinp[_mm_extract_epi16(count, 1)], 1);
            divres = _mm_insert_epi16(divres, divinp[_mm_extract_epi16(count, 2)], 2);
            divres = _mm_insert_epi16(divres, divinp[_mm_extract_epi16(count, 3)], 3);
            divres = _mm_insert_epi16(divres, divinp[_mm_extract_epi16(count, 4)], 4);
            divres = _mm_insert_epi16(divres, divinp[_mm_extract_epi16(count, 5)], 5);
            divres = _mm_insert_epi16(divres, divinp[_mm_extract_epi16(count, 6)], 6);
            return _mm_insert_epi16(divres, divinp[_mm_extract_epi16(count, 7)], 7);
        }();
        
        // Weight with original depending on edge value
        const __m128i result = _mm_adds_epu16(mul_u16(center_pixel, _mm_subs_epu16(_mm_set1_epi16(255), str), 255.0f),
            mul_u16(str, scale_u16(mul_u16(sum, divres, 4095.0f), 255.0f, 4095.0f), 255.0f));

        _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp), _mm_packus_epi16(result, result));
    }
    else
    {
        const int peak = (1 << bits) - 1;
        const __m128i thres = _mm_set1_epi32(static_cast<uint32_t>(llrint(sqrt((static_cast<double>(threshold) * threshold) / 3))));

        // Build edge values
        const __m128i center_pixel = _mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp)), zeroes);
        const __m128i add = _mm_add_epi32(_mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + stride)), zeroes), _mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + 1)), zeroes));
        const __m128i sllq = _mm_slli_epi32(center_pixel, 1);

        // Store weight with edge bias
        const __m128i str = _mm_min_epu32(_mm_sub_epi32(_mm_set1_epi32(strength), _mm_or_si128(_mm_sub_epi32(sllq, add), _mm_sub_epi32(add, sllq))), _mm_set1_epi32(peak));

        srcp = srcp - diff;

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const __m128i neighbour_pixel = _mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + x)), zeroes);
                const __m128i mask = _mm_cmpgt_epi32(thres, _mm_abs_epi32(_mm_sub_epi32(center_pixel, neighbour_pixel)));
                sum = _mm_add_epi32(sum, _mm_and_si128(neighbour_pixel, mask));
                count = _mm_sub_epi32(count, mask);
            }

            srcp += stride;
        }

        sum = _mm_add_epi32(sum, _mm_srli_epi32(_mm_slli_epi32(count, bits - 8), 1));

        __m128i divres_hi = [&]() {
            divres_hi = _mm_insert_epi32(divres_hi, divinp[_mm_extract_epi32(count, 0)], 0);
            return _mm_insert_epi32(divres_hi, divinp[_mm_extract_epi32(count, 1)], 1);
        }();

        __m128i divres_lo = [&]() {
            divres_lo = _mm_insert_epi32(divres_lo, divinp[_mm_extract_epi32(count, 2)], 0);
            return _mm_insert_epi32(divres_lo, divinp[_mm_extract_epi32(count, 3)], 1);
        }();

        __m128i sum_hi = [&]() {
            sum_hi = _mm_insert_epi32(sum_hi, _mm_extract_epi32(sum, 0), 0);
            return _mm_insert_epi32(sum_hi, _mm_extract_epi32(sum, 1), 1);
        }();

        __m128i sum_lo = [&]() {
            sum_lo = _mm_insert_epi32(sum_lo, _mm_extract_epi32(sum, 2), 0);
            return _mm_insert_epi32(sum_lo, _mm_extract_epi32(sum, 3), 1);
        }();

        __m128i mul_hi = _mm_mul_epu32(_mm_shuffle_epi32(sum_hi, _MM_SHUFFLE(3, 1, 2, 0)), _mm_shuffle_epi32(divres_hi, _MM_SHUFFLE(3, 1, 2, 0)));
        __m128i mul_lo = _mm_mul_epu32(_mm_shuffle_epi32(sum_lo, _MM_SHUFFLE(3, 1, 2, 0)), _mm_shuffle_epi32(divres_lo, _MM_SHUFFLE(3, 1, 2, 0)));

        __m128i hi = _mm_unpacklo_epi16(mul_hi, zeroes);
        __m128i hi1 = _mm_unpackhi_epi16(mul_hi, zeroes);
        __m128i lo = _mm_unpacklo_epi16(mul_lo, zeroes);
        __m128i lo1 = _mm_unpackhi_epi16(mul_lo, zeroes);

        __m128i result = [&]() {
            result = _mm_insert_epi32(result, _mm_extract_epi32(hi, 1), 0);
            result = _mm_insert_epi32(result, _mm_extract_epi32(hi1, 1), 1);
            result = _mm_insert_epi32(result, _mm_extract_epi32(lo, 1), 2);
            return _mm_insert_epi32(result, _mm_extract_epi32(lo1, 1), 3);
        }();

        // Weight with original depending on edge value
        result = _mm_add_epi32(mul_u32(center_pixel, _mm_sub_epi32(_mm_set1_epi32(peak), str), static_cast<float>(peak)),
            mul_u32(str, result, static_cast<float>(peak)));

        _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp), _mm_packus_epi32(result, result));
    }
}

template <typename T, bool interlaced, int bits, bool hqy, bool hqc>
void SmoothUV2::smoothN_SSE41(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env)
{
    const int vsize = (sizeof(T) == 2) ? 4 : 8;
    const int offs = (interlaced) ? 2 : 1;
    int planes_y[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    int planecount = min(vi.NumComponents(), 3);
    for (int i = 0; i < planecount; ++i)
    {
        const int h = dst->GetHeight(planes_y[i]);

        if constexpr (!hqy)
        {
            if (i == 0 && _thresholdY == -1337)
            {
                env->BitBlt(dst->GetWritePtr(planes_y[i]), dst->GetPitch(planes_y[i]), src->GetReadPtr(planes_y[i]), src->GetPitch(planes_y[i]), src->GetRowSize(planes_y[i]), h);
                continue;
            }
        }

        int stride = src->GetPitch(planes_y[i]) / sizeof(T);
        int dst_stride = dst->GetPitch(planes_y[i]) / sizeof(T);
        const int w = src->GetRowSize(planes_y[i]) / sizeof(T);
        const T* srcp = reinterpret_cast<const T*>(src->GetReadPtr(planes_y[i]));
        T* dstp = reinterpret_cast<T*>(dst->GetWritePtr(planes_y[i]));        

        const T* srcp2 = srcp + stride;
        T* dstp2 = dstp + dst_stride;
        int h2 = h;
        int radius_w, radius_h;
        int thr;

        if (interlaced)
        {
            stride *= 2;
            dst_stride *= 2;
            h2 >>= 1;
        }

        void (*sum_pixels)(const T * srcp, T * dstp, const int stride,
            const int diff, const int width, const int height,
            const int threshold,
            const uint16_t * divinp, const int strength);

        if (i == 0)
        {
            radius_w = radiusy;
            radius_h = radiusy;
            thr = _thresholdY;

            if constexpr (hqy)
                sum_pixels = sshiq_sum_pixels_SSE41<T, bits>;
            else
                sum_pixels = sum_pixels_SSE41<T, bits>;

        }
        else
        {
            radius_w = radiuscw;
            radius_h = radiusch;
            thr = _thresholdC;

            if constexpr (hqc)
                sum_pixels = sshiq_sum_pixels_SSE41<T, bits>;
            else
                sum_pixels = sum_pixels_SSE41<T, bits>;
        }

        for (int y = 0; y < h2; ++y)
        {
            int y0 = (y < radius_h) ? y : radius_h;

            int yn = (y < h2 - radius_h) ? y0 + radius_h + 1 * offs
                : y0 + (h2 - y);

            if (interlaced)
                yn--;

            int offset = y0 * stride;

            for (int x = 0; x < w; x += vsize)
            {
                int x0 = (x < radius_w) ? x : radius_w;

                int xn = (x + vsize - 1 + radius_w < w - 1) ? x0 + radius_w + 1
                    : max(x0 + (w - x) - (vsize - 1), 1);

                sum_pixels(srcp + x, dstp + x,
                    stride,
                    offset + x0,
                    xn, yn,
                    thr,
                    divin, _strength);

                if (interlaced)
                {
                    sum_pixels(srcp2 + x, dstp2 + x,
                        stride,
                        offset + x0,
                        xn, yn,
                        thr,
                        divin, _strength);
                }
            }

            dstp += dst_stride;
            srcp += stride;
            dstp2 += dst_stride;
            srcp2 += stride;
        }

        if (interlaced && h % 2)
        {
            int yn = radius_h;

            int offset = radius_h * stride;

            for (int x = 0; x < w; x += vsize)
            {
                int x0 = (x < radius_w) ? x : radius_w;

                int xn = (x + vsize - 1 + radius_w < w - 1) ? x0 + radius_w + 1
                    : max(x0 + (w - x) - (vsize - 1), 1);

                sum_pixels(srcp + x, dstp + x,
                    stride,
                    offset + x0,
                    xn, yn,
                    thr,
                    divin, _strength);
            }
        }
    }
}

SmoothUV2::SmoothUV2(PClip _child, int radiusY, int radiusC, int thresholdY, int thresholdC, int strength, bool HQY, bool HQC, int interlaced, IScriptEnvironment* env)
    : GenericVideoFilter(_child), radiusy(radiusY), hqy(HQY), hqc(HQC), _interlaced(interlaced)
{
    if (thresholdY != -1337)
    {
        if (vi.IsRGB() || vi.BitsPerComponent() == 32 || vi.NumComponents() < 3 || !vi.IsPlanar())
            env->ThrowError("SSHiQ2: only 8..16-bit YUV planar format supported with minimum three planes.");
        if (radiusY < 1 || radiusY > 7)
            env->ThrowError("SSHiQ2: radiusY must be between 1 and 7 (inclusive).");
        if (vi.Is420() && (radiusC < 1 || radiusC > 7))
            env->ThrowError("SSHiQ2: radiusC must be between 1 and 7 (inclusive).");
        if (!vi.Is420() && (radiusC < 1 || radiusC > 3))
            env->ThrowError("SSHiQ2: radiusC must be between 1 and 3 (inclusive) for subsampling other than 4:2:0.");
        if (thresholdY < 1 || thresholdY > 450)
            env->ThrowError("SSHiQ2: thresholdY must be between 1 and 450 (inclusive).");
        if (thresholdC < 1 || thresholdC > 450)
            env->ThrowError("SSHiQ2: thresholdC must be between 1 and 450 (inclusive).");
        if (strength < 0 || strength > 255)
            env->ThrowError("SSHiQ2: strength must be between 0 and 255 (inclusive).");
        if (_interlaced < -1 || _interlaced > 1)
            env->ThrowError("SSHiQ2: interlaced must be between -1 and 1 (inclusive).");
    }
    else
    {
        if (vi.IsRGB() || vi.BitsPerComponent() == 32 || vi.NumComponents() < 3 || !vi.IsPlanar())
            env->ThrowError("SmoothUV2: only 8..16-bit YUV planar format supported with minimum three planes.");
        if (vi.Is420() && (radiusC < 1 || radiusC > 7))
            env->ThrowError("SmoothUV2: radius must be between 1 and 7 (inclusive).");
        if (!vi.Is420() && (radiusC < 1 || radiusC > 3))
            env->ThrowError("SmoothUV2: radius must be between 1 and 3 (inclusive) for subsampling other than 4:2:0.");
        if (thresholdC < 1 || thresholdC > 450)
            env->ThrowError("SmoothUV2: threshold must be between 1 and 450 (inclusive).");
        if (_interlaced < -1 || _interlaced > 1)
            env->ThrowError("SmoothUV2: interlaced must be between -1 and 1 (inclusive).");
    }

    if (vi.ComponentSize() == 2)
    {
        const int peak = (1 << vi.BitsPerComponent()) - 1;

        _thresholdY = (thresholdY != -1337) ? (thresholdY * peak / 255) : thresholdY;
        _thresholdC = thresholdC * peak / 255;
        _strength = strength * peak / 255;
    }
    else
    {
        _thresholdY = thresholdY;
        _thresholdC = thresholdC;
        _strength = strength;
    }

    radiuscw = radiusC << (1 - vi.GetPlaneWidthSubsampling(PLANAR_U));
    radiusch = radiusC << (1 - vi.GetPlaneHeightSubsampling(PLANAR_U));

    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; }

    for (int i = 1; i < 256; ++i)
        divin[i] = static_cast<uint16_t>(min(static_cast<int>(65536.0 / i + 0.5), 65535));

    if (_interlaced > -1)
        field_based = _interlaced;
    else
    {
        if (!has_at_least_v8)
            field_based = 0;
    }

    if (!has_at_least_v8)
    {
        if (vi.ComponentSize() == 1)
        {
            if (HQY)
            {
                if (HQC)
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint8_t, true, 8, true, true> : &SmoothUV2::smoothN_SSE41<uint8_t, false, 8, true, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint8_t, true, 8, true, false> : &SmoothUV2::smoothN_SSE41<uint8_t, false, 8, true, false>;
            }
            else
            {
                if (HQC)
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint8_t, true, 8, false, true> : &SmoothUV2::smoothN_SSE41<uint8_t, false, 8, false, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint8_t, true, 8, false, false> : &SmoothUV2::smoothN_SSE41<uint8_t, false, 8, false, false>;
            }
        }
        else
        {
            if (HQY)
            {
                if (HQC)
                {
                    switch (vi.BitsPerComponent())
                    {
                        case 10: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 10, true, true> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 10, true, true>; break;
                        case 12: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 12, true, true> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 12, true, true>; break;
                        case 14: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 14, true, true> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 14, true, true>; break;
                        case 16: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 16, true, true> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 16, true, true>; break;
                    }
                }
                else
                {
                    switch (vi.BitsPerComponent())
                    {
                        case 10: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 10, true, false> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 10, true, false>; break;
                        case 12: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 12, true, false> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 12, true, false>; break;
                        case 14: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 14, true, false> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 14, true, false>; break;
                        case 16: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 16, true, false> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 16, true, false>; break;
                    }
                }
            }
            else
            {
                if (HQC)
                {
                    switch (vi.BitsPerComponent())
                    {
                        case 10: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 10, false, true> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 10, false, true>; break;
                        case 12: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 12, false, true> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 12, false, true>; break;
                        case 14: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 14, false, true> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 14, false, true>; break;
                        case 16: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 16, false, true> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 16, false, true>; break;
                    }
                }
                else
                {
                    switch (vi.BitsPerComponent())
                    {
                        case 10: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 10, false, false> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 10, false, false>; break;
                        case 12: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 12, false, false> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 12, false, false>; break;
                        case 14: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 14, false, false> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 14, false, false>; break;
                        case 16: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 16, false, false> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 16, false, false>; break;
                    }
                }
            }
        }
    }
}

PVideoFrame __stdcall SmoothUV2::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = (has_at_least_v8) ? env->NewVideoFrameP(vi, &src) : env->NewVideoFrame(vi);

    if (has_at_least_v8)
    {
        if (_interlaced == -1)
        {
            const AVSMap* props = env->getFramePropsRO(src);
            field_based = (env->propNumElements(props, "_FieldBased") > 0) ? env->propGetInt(props, "_FieldBased", 0, nullptr) : 0;
        }

        if (vi.ComponentSize() == 1)
        {
            if (hqy)
            {
                if (hqc)
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint8_t, true, 8, true, true> : &SmoothUV2::smoothN_SSE41<uint8_t, false, 8, true, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint8_t, true, 8, true, false> : &SmoothUV2::smoothN_SSE41<uint8_t, false, 8, true, false>;
            }
            else
            {
                if (hqc)
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint8_t, true, 8, false, true> : &SmoothUV2::smoothN_SSE41<uint8_t, false, 8, false, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint8_t, true, 8, false, false> : &SmoothUV2::smoothN_SSE41<uint8_t, false, 8, false, false>;
            }
        }
        else
        {
            if (hqy)
            {
                if (hqc)
                {
                    switch (vi.BitsPerComponent())
                    {
                        case 10: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 10, true, true> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 10, true, true>; break;
                        case 12: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 12, true, true> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 12, true, true>; break;
                        case 14: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 14, true, true> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 14, true, true>; break;
                        case 16: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 16, true, true> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 16, true, true>; break;
                    }
                }
                else
                {
                    switch (vi.BitsPerComponent())
                    {
                        case 10: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 10, true, false> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 10, true, false>; break;
                        case 12: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 12, true, false> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 12, true, false>; break;
                        case 14: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 14, true, false> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 14, true, false>; break;
                        case 16: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 16, true, false> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 16, true, false>; break;
                    }
                }
            }
            else
            {
                if (hqc)
                {
                    switch (vi.BitsPerComponent())
                    {
                        case 10: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 10, false, true> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 10, false, true>; break;
                        case 12: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 12, false, true> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 12, false, true>; break;
                        case 14: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 14, false, true> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 14, false, true>; break;
                        case 16: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 16, false, true> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 16, false, true>; break;
                    }
                }
                else
                {
                    switch (vi.BitsPerComponent())
                    {
                        case 10: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 10, false, false> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 10, false, false>; break;
                        case 12: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 12, false, false> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 12, false, false>; break;
                        case 14: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 14, false, false> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 14, false, false>; break;
                        case 16: smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<uint16_t, true, 16, false, false> : &SmoothUV2::smoothN_SSE41<uint16_t, false, 16, false, false>; break;
                    }
                }
            }
        }
    }

    (this->*smooth)(dst, src, env);

    return dst;
}

AVSValue __cdecl Create_SmoothUV2(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    return new SmoothUV2(
        args[0].AsClip(),
        1,
        args[1].AsInt(3),
        -1337,
        args[2].AsInt(150),
        255,
        false,
        false,
        args[3].AsInt(-1),
        env);
}

AVSValue __cdecl Create_SSHiQ2(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    return new SmoothUV2(
        args[0].AsClip(),
        args[1].AsInt(5),
        args[2].AsInt(3),
        args[3].AsInt(20) ,
        args[4].AsInt(30),
        args[5].AsInt(240),
        args[6].AsBool(true),
        args[7].AsBool(true),
        args[8].AsInt(-1),
        env);
}

const AVS_Linkage* AVS_linkage;

extern "C" __declspec(dllexport)
const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    env->AddFunction("SmoothUV2", "c[radius]i[threshold]i[interlaced]i", Create_SmoothUV2, 0);
    env->AddFunction("SSHiQ2", "c[rY]i[rC]i[tY]i[tC]i[str]i[HQY]b[HQC]b[interlaced]i", Create_SSHiQ2, 0);

    return "SmoothUV2";
}
