#include <smmintrin.h>

#include "SmoothUV2.h"

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

AVS_FORCEINLINE void SmoothUV2::sum_pixels_SSE41(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold)
{
    const __m128i zeroes = _mm_setzero_si128();
    __m128i sum = zeroes;
    __m128i count = zeroes;

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

    sum = _mm_add_epi32(sum, _mm_srli_epi32(_mm_slli_epi32(count, 8), 1));

    __m128i divres_hi = [&]() {
        divres_hi = _mm_insert_epi32(divres_hi, divin[_mm_extract_epi32(count, 0)], 0);
        return _mm_insert_epi32(divres_hi, divin[_mm_extract_epi32(count, 1)], 1);
    }();

    __m128i divres_lo = [&]() {
        divres_lo = _mm_insert_epi32(divres_lo, divin[_mm_extract_epi32(count, 2)], 0);
        return _mm_insert_epi32(divres_lo, divin[_mm_extract_epi32(count, 3)], 1);
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

AVS_FORCEINLINE void SmoothUV2::sshiq_sum_pixels_SSE41(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold)
{
    const __m128i zeroes = _mm_setzero_si128();
    __m128i sum = zeroes;
    __m128i count = zeroes;

    const __m128i thres = _mm_set1_epi32(static_cast<uint32_t>(llrint(sqrt((static_cast<double>(threshold) * threshold) / 3))));

    // Build edge values
    const __m128i center_pixel = _mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp)), zeroes);
    const __m128i add = _mm_add_epi32(_mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(origsp + (stride << 1))), zeroes), _mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(origsp + 1)), zeroes));
    const __m128i sllq = _mm_slli_epi32(center_pixel, 1);

    // Store weight with edge bias
    const __m128i str = _mm_min_epu32(_mm_sub_epi32(_mm_set1_epi32(_strength), _mm_or_si128(_mm_sub_epi32(sllq, add), _mm_sub_epi32(add, sllq))), _mm_set1_epi32(65535));

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

    sum = _mm_add_epi32(sum, _mm_srli_epi32(_mm_slli_epi32(count, 8), 1));

    __m128i divres_hi = [&]() {
        divres_hi = _mm_insert_epi32(divres_hi, divin[_mm_extract_epi32(count, 0)], 0);
        return _mm_insert_epi32(divres_hi, divin[_mm_extract_epi32(count, 1)], 1);
    }();

    __m128i divres_lo = [&]() {
        divres_lo = _mm_insert_epi32(divres_lo, divin[_mm_extract_epi32(count, 2)], 0);
        return _mm_insert_epi32(divres_lo, divin[_mm_extract_epi32(count, 3)], 1);
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
    result = _mm_add_epi32(mul_u32(center_pixel, _mm_sub_epi32(_mm_set1_epi32(65535), str), 65535.0f),
        mul_u32(str, result, 65535.0f));

    _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp), _mm_packus_epi32(result, result));
}

template <bool interlaced, bool hqy, bool hqc>
void SmoothUV2::smoothN_SSE41(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env)
{
    void (SmoothUV2:: * sum_pixels)(const uint8_t * origsp, const uint16_t * srcp, uint16_t * dstp, const int stride, const int diff, const int width, const int height, const int threshold);
    void (SmoothUV2:: * sum_pixels_c)(const uint8_t * origsp, const uint16_t * srcp, uint16_t * dstp, const int stride, const int diff, const int width, const int height, const int threshold);

    const int offs = (interlaced) ? 2 : 1;
    int planes_y[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    int planecount = std::min(vi.NumComponents(), 3);
    for (int i = 0; i < planecount; ++i)
    {
        int radius_w, radius_h;
        int thr;
        const int h = dst->GetHeight(planes_y[i]);

        if constexpr (!hqy)
        {
            if (i == 0 && _thresholdY == -1337)
            {
                env->BitBlt(dst->GetWritePtr(planes_y[i]), dst->GetPitch(planes_y[i]), src->GetReadPtr(planes_y[i]), src->GetPitch(planes_y[i]), src->GetRowSize(planes_y[i]), h);
                continue;
            }
        }

        if (i == 0)
        {
            if (_thresholdY == 0)
            {
                env->BitBlt(dst->GetWritePtr(planes_y[i]), dst->GetPitch(planes_y[i]), src->GetReadPtr(planes_y[i]), src->GetPitch(planes_y[i]), src->GetRowSize(planes_y[i]), h);
                continue;
            }

            radius_w = radiusy;
            radius_h = radiusy;
            thr = _thresholdY;

            if constexpr (hqy)
            {
                sum_pixels = &SmoothUV2::sshiq_sum_pixels_SSE41;
                sum_pixels_c = &SmoothUV2::sshiq_sum_pixels_c;
            }
            else
            {
                sum_pixels = &SmoothUV2::sum_pixels_SSE41;
                sum_pixels_c = &SmoothUV2::sum_pixels_c;
            }
        }
        else
        {
            if (_thresholdC == 0)
            {
                env->BitBlt(dst->GetWritePtr(planes_y[i]), dst->GetPitch(planes_y[i]), src->GetReadPtr(planes_y[i]), src->GetPitch(planes_y[i]), src->GetRowSize(planes_y[i]), h);
                continue;
            }

            radius_w = radiuscw;
            radius_h = radiusch;
            thr = _thresholdC;

            if constexpr (hqc)
            {
                sum_pixels = &SmoothUV2::sshiq_sum_pixels_SSE41;
                sum_pixels_c = &SmoothUV2::sshiq_sum_pixels_c;
            }
            else
            {
                sum_pixels = &SmoothUV2::sum_pixels_SSE41;
                sum_pixels_c = &SmoothUV2::sum_pixels_c;
            }
        }

        int stride = src->GetPitch(planes_y[i]) / 2;
        int dst_stride = dst->GetPitch(planes_y[i]) / 2;
        const int w = src->GetRowSize(planes_y[i]) / 2;
        const uint8_t* origsp = src->GetReadPtr(planes_y[i]);
        const uint16_t* srcp = reinterpret_cast<const uint16_t*>(origsp);
        uint16_t* dstp = reinterpret_cast<uint16_t*>(dst->GetWritePtr(planes_y[i]));

        const uint16_t* srcp2 = srcp + stride;
        uint16_t* dstp2 = dstp + dst_stride;
        int h2 = h;
        const int col = (w - 16) - ((w - 16) % 4);

        if (interlaced)
        {
            stride *= 2;
            dst_stride *= 2;
            h2 >>= 1;
        }

        for (int y = 0; y < h2; ++y)
        {
            const int y0 = (y < radius_h) ? y : radius_h;

            int yn = (y < h2 - radius_h) ? y0 + radius_h + 1 * offs
                : y0 + (h2 - y);

            if (interlaced)
                yn--;

            int offset = y0 * stride;

            for (int x = 0; x < 16; ++x)
            {
                const int x0 = (x < radius_w) ? x : radius_w;
                const int xn = x0 + radius_w + 1;

                (this->*sum_pixels_c)(origsp + (x << 1), srcp + x, dstp + x, stride, offset + x0, xn, yn, thr);

                if (interlaced)
                    (this->*sum_pixels_c)(origsp + (x << 1), srcp2 + x, dstp2 + x, stride, offset + x0, xn, yn, thr);
            }

            for (int x = 16; x < col; x += 4)
            {
                const int xn = (radius_w << 1) + 1;

                (this->*sum_pixels)(origsp + (x << 1), srcp + x, dstp + x, stride, offset + radius_w, xn, yn, thr);

                if (interlaced)
                    (this->*sum_pixels)(origsp + (x << 1), srcp2 + x, dstp2 + x, stride, offset + radius_w, xn, yn, thr);
            }

            for (int x = col; x < w; ++x)
            {
                const int xn = (x + radius_w < w - 1) ? (radius_w << 1) + 1 : std::max(radius_w + (w - x), 1);

                (this->*sum_pixels_c)(origsp + (x << 1), srcp + x, dstp + x, stride, offset + radius_w, xn, yn, thr);

                if (interlaced)
                    (this->*sum_pixels_c)(origsp + (x << 1), srcp2 + x, dstp2 + x, stride, offset + radius_w, xn, yn, thr);
            }

            dstp += dst_stride;
            srcp += stride;
            dstp2 += dst_stride;
            srcp2 += stride;
        }

        if (interlaced && h % 2)
        {
            const int yn = radius_h;

            const int offset = radius_h * stride;

            for (int x = 0; x < 16; ++x)
            {
                const int x0 = (x < radius_w) ? x : radius_w;

                (this->*sum_pixels_c)(origsp + (x << 1LL), srcp + x, dstp + x, stride, offset + x0, x0 + radius_w + 1, yn, thr);
            }

            for (int x = 16; x < col; x += 4)
                (this->*sum_pixels)(origsp + (x << 1), srcp + x, dstp + x, stride, offset + radius_w, (radius_w << 1) + 1, yn, thr);

            for (int x = col; x < w; ++x)
                (this->*sum_pixels_c)(origsp + (x << 1LL), srcp + x, dstp + x, stride, offset + radius_w, (x + radius_w < w - 1) ? (radius_w << 1) + 1 : std::max(radius_w + (w - x), 1), yn, thr);
        }
    }
}

template void SmoothUV2::smoothN_SSE41<true, true, true>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
template void SmoothUV2::smoothN_SSE41<true, true, false>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
template void SmoothUV2::smoothN_SSE41<true, false, true>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
template void SmoothUV2::smoothN_SSE41<true, false, false>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);

template void SmoothUV2::smoothN_SSE41<false, true, true>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
template void SmoothUV2::smoothN_SSE41<false, true, false>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
template void SmoothUV2::smoothN_SSE41<false, false, true>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
template void SmoothUV2::smoothN_SSE41<false, false, false>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
