#include <algorithm>
#include <smmintrin.h>

#include "avisynth.h"
#include "avs/minmax.h"
#include "avs/config.h"

template <typename T, int bits>
static AVS_FORCEINLINE void sum_pixels_SSE41(const T* srcp, T* dstp, const int stride,
    const int diff, const int width, const int height,
    const int threshold,
    const uint16_t* divinp)
{
    __m128i zeroes = _mm_setzero_si128();

    __m128i sum = zeroes;
    __m128i count = zeroes;

    if constexpr (std::is_same_v<T, uint8_t>)
    {
        const __m128i thres = _mm_set1_epi16(threshold);
        __m128i center_pixel = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp)), zeroes);

        srcp = srcp - diff;

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x <= width; ++x)
            {
                __m128i neighbour_pixel = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + x)),
                    zeroes);

                __m128i abs_diff = _mm_abs_epi16(_mm_sub_epi16(center_pixel, neighbour_pixel));

                // Absolute difference less than thres
                __m128i mask = _mm_cmpgt_epi16(thres, abs_diff);

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

        // Now multiply (divres/65536)
        _mm_storel_epi64((__m128i*)dstp, _mm_packus_epi16(_mm_mulhi_epu16(sum, divres), _mm_mulhi_epu16(sum, divres)));
    }
    else
    {
        const __m128i thres = _mm_set1_epi32(threshold);
        __m128i center_pixel = _mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp)), zeroes);

        srcp = srcp - diff;

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x <= width; ++x)
            {
                __m128i neighbour_pixel = _mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + x)),
                    zeroes);

                __m128i abs_diff = _mm_abs_epi32(_mm_sub_epi32(center_pixel, neighbour_pixel));

                // Absolute difference less than thres
                __m128i mask = _mm_cmpgt_epi32(thres, abs_diff);

                // Sum up the pixels that meet the criteria
                sum = _mm_add_epi32(sum,
                    _mm_and_si128(neighbour_pixel, mask));

                // Keep track of how many pixels are in the sum
                count = _mm_sub_epi32(count, mask);
            }

            srcp += stride;
        }
        
        const int shift = bits - 8;

        sum = _mm_add_epi32(sum, _mm_srli_epi32(_mm_slli_epi32(count, shift), 1));

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
            result = _mm_insert_epi32(result, _mm_extract_epi32(lo1, 1), 3);

            return _mm_packus_epi32(result, result);
        }();
        
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp), result);
    }
}

template <typename T, bool interlaced, int bits>
static void smoothN_SSE41(int radius, const uint8_t* origsrc, uint8_t* origdst, int stride, int dst_stride, int w, int h, const int threshold, const uint16_t* divin)
{
    stride /= sizeof(T);
    dst_stride /= sizeof(T);
    w /= sizeof(T);
    const T* srcp = reinterpret_cast<const T*>(origsrc);
    const T* srcp2 = srcp + stride;
    T* dstp = reinterpret_cast<T*>(origdst);
    T* dstp2 = dstp + dst_stride;

    const int vsize = (sizeof(T) == 2) ? 4 : 8;
    int h2 = h;

    if (interlaced)
    {
        stride *= 2;
        dst_stride *= 2;
        h2 >>= 1;
    }

    for (int y = 0; y < h2; ++y)
    {
        int y0 = (y < radius) ? y : radius;

        int yn = (y < h2 - radius) ? y0 + radius + 1
            : y0 + (h2 - y);

        if (interlaced)
            yn--;

        int offset = y0 * stride;

        for (int x = 0; x < w; x += vsize)
        {
            int x0 = (x < radius) ? x : radius;

            int xn = (x + vsize - 1 + radius < w - 1) ? x0 + radius + 1
                : max(x0 + w - x - vsize - 1, 0);

            sum_pixels_SSE41<T, bits>(srcp + x, dstp + x,
                stride,
                offset + x0,
                xn, yn,
                threshold,
                divin);

            if (interlaced)
            {
                sum_pixels_SSE41<T, bits>(srcp2 + x, dstp2 + x,
                    stride,
                    offset + x0,
                    xn, yn,
                    threshold,
                    divin);
            }
        }

        dstp += dst_stride;
        srcp += stride;
        dstp2 += dst_stride;
        srcp2 += stride;
    }

    if (interlaced && h % 2)
    {
        int yn = radius;

        int offset = radius * stride;

        for (int x = 0; x < w; x += vsize)
        {
            int x0 = (x < radius) ? x : radius;

            int xn = (x + vsize - 1 + radius < w - 1) ? x0 + radius + 1
                : max(x0 + w - x - vsize - 1, 0);

            sum_pixels_SSE41<T, bits>(srcp + x, dstp + x,
                stride,
                offset + x0,
                xn, yn,
                threshold,
                divin);
        }
    }
}

class SmoothUV : public GenericVideoFilter
{
    int _radius, _threshold;
    int _interlaced;
    bool has_at_least_v8;

    uint16_t divin[256];

public:
    SmoothUV(PClip _child, int radius, int threshold, int interlaced, IScriptEnvironment* env)
        : GenericVideoFilter(_child), _radius(radius), _threshold(threshold), _interlaced(interlaced)
    {
        if (vi.IsRGB() || vi.BitsPerComponent() == 32 || vi.NumComponents() < 3 || !vi.IsPlanar())
            env->ThrowError("SmoothUV: only 8..16-bit YUV planar format supported with minimum three planes.");
        if (_radius < 1 || _radius > 7)
            env->ThrowError("SmoothUV: radius must be between 1 and 7 (inclusive).");
        if (_threshold < 0 || _threshold > 255)
            env->ThrowError("SmoothUV: threshold must be between 0 and 255 (inclusive).");
        if (_interlaced < -1 || _interlaced > 1)
            env->ThrowError("SmoothUV: interlaced must be between -1 and 1 (inclusive).");

        if (vi.ComponentSize() == 2)
            _threshold *= ((1 << vi.BitsPerComponent()) - 1) / 255;

        has_at_least_v8 = true;
        try { env->CheckVersion(8); }
        catch (const AvisynthError&) { has_at_least_v8 = false; }

        for (int i = 1; i < 256; ++i)
            divin[i] = static_cast<uint16_t>(min(static_cast<int>(65536.0 / i + 0.5), 65535));
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) override
    {
        return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
    {
        PVideoFrame src = child->GetFrame(n, env);
        PVideoFrame dst = (has_at_least_v8) ? env->NewVideoFrameP(vi, &src) : env->NewVideoFrame(vi);

        const int64_t field_based = [&]()
        {
            if (has_at_least_v8)
            {
                if (_interlaced == -1)
                {
                    const AVSMap* props = env->getFramePropsRO(src);

                    if (env->propNumElements(props, "_FieldBased") > 0)
                        return env->propGetInt(props, "_FieldBased", 0, nullptr);
                    else
                        return static_cast<int64_t>(0);
                }
                else
                    return static_cast<int64_t>(_interlaced);
            }
            else
            {
                if (_interlaced == -1)
                    return static_cast<int64_t>(0);
                else
                    return static_cast<int64_t>(_interlaced);
            }
        }();

        void (*smooth)(int radius, const uint8_t * origsrc, uint8_t * origdst, int stride, int dst_stride, int w, int h, const int threshold, const uint16_t * divin) = [&]() {
            if (vi.ComponentSize() == 1)
                return (field_based) ? smoothN_SSE41<uint8_t, true, 8> : smoothN_SSE41<uint8_t, false, 8>;
            else
            {
                switch (vi.BitsPerComponent())
                {
                    case 10: return (field_based) ? smoothN_SSE41<uint16_t, true, 10> : smoothN_SSE41<uint16_t, false, 10>;
                    case 12: return (field_based) ? smoothN_SSE41<uint16_t, true, 12> : smoothN_SSE41<uint16_t, false, 12>;
                    case 14: return (field_based) ? smoothN_SSE41<uint16_t, true, 14> : smoothN_SSE41<uint16_t, false, 14>;
                    case 16: return (field_based) ? smoothN_SSE41<uint16_t, true, 16> : smoothN_SSE41<uint16_t, false, 16>;
                }                
            }
        }();

        int planes_y[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
        int planecount = min(vi.NumComponents(), 3);
        for (int i = 0; i < planecount; ++i)
        {
            int src_stride = src->GetPitch(planes_y[i]);
            int dst_stride = dst->GetPitch(planes_y[i]);
            int width = src->GetRowSize(planes_y[i]);
            int height = dst->GetHeight(planes_y[i]);
            const uint8_t* srcp = src->GetReadPtr(planes_y[i]);
            uint8_t* dstp = dst->GetWritePtr(planes_y[i]);

            if (i == 0)
                env->BitBlt(dstp, dst_stride, srcp, src_stride, width, height);
            else
                smooth(_radius, srcp, dstp, src_stride, dst_stride, width, height, _threshold, divin);
        }

        return dst;
    }
};

AVSValue __cdecl Create_SmoothUV(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    return new SmoothUV(
        args[0].AsClip(),
        args[1].AsInt(3),
        args[2].AsInt(150),
        args[3].AsInt(-1),
        env);
}

const AVS_Linkage* AVS_linkage;

extern "C" __declspec(dllexport)
const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    env->AddFunction("SmoothUV", "c[radius]i[threshold]i[interlaced]i", Create_SmoothUV, 0);
    return "SmoothUV";
}
