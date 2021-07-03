#include <emmintrin.h>

#include "SmoothUV2.h"

static AVS_FORCEINLINE __m128i mul_u32_(const __m128i a, const __m128i b, const float peak)
{
    return _mm_cvtps_epi32(_mm_div_ps(_mm_mul_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)), _mm_set_ps1(peak)));
}

static AVS_FORCEINLINE __m128i packus(const __m128i a, const __m128i b)
{
    const static __m128i val_32 = _mm_set_epi32(0x8000, 0x8000, 0x8000, 0x8000);

    return _mm_add_epi16(_mm_packs_epi32(_mm_sub_epi32(a, val_32), _mm_sub_epi32(b, val_32)), _mm_set_epi16(0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000));
}

static AVS_FORCEINLINE __m128i abs(const __m128i a)
{
    const __m128i mask = _mm_cmplt_epi32(a, _mm_setzero_si128());
    return _mm_add_epi32(_mm_xor_si128(a, mask), _mm_srli_epi32(mask, 31));
}

static AVS_FORCEINLINE __m128i insert(const __m128i a, int i, const int imm8)
{
    switch (imm8)
    {
        case 0: return _mm_insert_epi16(_mm_insert_epi16(a, i, 0), i << 16, 1);
        case 1: return _mm_insert_epi16(_mm_insert_epi16(a, i, 2), i << 16, 3);
        case 2: return _mm_insert_epi16(_mm_insert_epi16(a, i, 4), i << 16, 5);
        default: return _mm_insert_epi16(_mm_insert_epi16(a, i, 6), i << 16, 7);
    }
}

static AVS_FORCEINLINE __m128i insert_hi(const __m128i a, int i, const int imm8)
{
    switch (imm8)
    {
        case 0: return _mm_insert_epi16(_mm_insert_epi16(a, i, 0), i >> 16, 1);
        default: return _mm_insert_epi16(_mm_insert_epi16(a, i, 2), i >> 16, 3);
    }
}

static AVS_FORCEINLINE int extract(const __m128i a, const int imm)
{
    switch (imm)
    {
        case 0: return _mm_cvtsi128_si32(_mm_srli_si128((a), 0));
        case 1: return _mm_cvtsi128_si32(_mm_srli_si128((a), 4));
        case 2: return _mm_cvtsi128_si32(_mm_srli_si128((a), 8));
        default: return _mm_cvtsi128_si32(_mm_srli_si128((a), 12));
    }
}

AVS_FORCEINLINE void SmoothUV2::sum_pixels_SSE2(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int)
{
    const __m128i zeroes = _mm_setzero_si128();
    __m128i sum = zeroes;
    __m128i count = zeroes;

    const __m128i thres = _mm_set1_epi32(sqrt(static_cast<int64_t>(threshold) * threshold / 3));
    const __m128i center_pixel = _mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp)), zeroes);

    srcp = srcp - diff;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const __m128i neighbour_pixel = _mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + x)), zeroes);

            const __m128i abs_diff = abs(_mm_sub_epi32(center_pixel, neighbour_pixel));

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
        divres_hi = insert(divres_hi, divin[extract(count, 0)], 0);
        return insert(divres_hi, divin[extract(count, 1)], 1);
    }();

    __m128i divres_lo = [&]() {
        divres_lo = insert(divres_lo, divin[extract(count, 2)], 0);
        return insert(divres_lo, divin[extract(count, 3)], 1);
    }();

    __m128i sum_hi = [&]() {
        sum_hi = insert_hi(sum_hi, extract(sum, 0), 0);
        return insert_hi(sum_hi, extract(sum, 1), 1);
    }();

    __m128i sum_lo = [&]() {
        sum_lo = insert_hi(sum_lo, extract(sum, 2), 0);
        return insert_hi(sum_lo, extract(sum, 3), 1);
    }();

    const __m128i mul_hi = _mm_mul_epu32(_mm_shuffle_epi32(sum_hi, _MM_SHUFFLE(3, 1, 2, 0)), _mm_shuffle_epi32(divres_hi, _MM_SHUFFLE(3, 1, 2, 0)));
    const __m128i mul_lo = _mm_mul_epu32(_mm_shuffle_epi32(sum_lo, _MM_SHUFFLE(3, 1, 2, 0)), _mm_shuffle_epi32(divres_lo, _MM_SHUFFLE(3, 1, 2, 0)));

    const __m128i hi = _mm_unpacklo_epi16(mul_hi, zeroes);
    const __m128i hi1 = _mm_unpackhi_epi16(mul_hi, zeroes);
    const __m128i lo = _mm_unpacklo_epi16(mul_lo, zeroes);
    const __m128i lo1 = _mm_unpackhi_epi16(mul_lo, zeroes);

    __m128i result = [&]() {
        result = insert(result, extract(hi, 1), 0);
        result = insert(result, extract(hi1, 1), 1);
        result = insert(result, extract(lo, 1), 2);
        return insert(result, extract(lo1, 1), 3);
    }();

    _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp), packus(result, result));
}

AVS_FORCEINLINE void SmoothUV2::sshiq_sum_pixels_SSE2(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int strength)
{
    const __m128i zeroes = _mm_setzero_si128();
    __m128i sum = zeroes;
    __m128i count = zeroes;

    const __m128i thres = _mm_set1_epi32(sqrt(static_cast<int64_t>(threshold) * threshold / 3));

    // Build edge values
    const __m128i center_pixel = _mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp)), zeroes);
    const __m128i add = _mm_add_epi32(_mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(origsp + (static_cast<int64_t>(stride) << 1))), zeroes), _mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(origsp + 1)), zeroes));
    const __m128i sllq = _mm_slli_epi32(center_pixel, 1);

    // Store weight with edge bias
    const __m128i sub = _mm_sub_epi32(_mm_set1_epi32(strength), _mm_or_si128(_mm_sub_epi32(sllq, add), _mm_sub_epi32(add, sllq)));
    const __m128i peak = _mm_set1_epi32(65535);
    const __m128i greater = _mm_cmpgt_epi32(sub, peak);
    const __m128i str = _mm_or_si128(_mm_and_si128(greater, peak), _mm_andnot_si128(greater, sub));

    srcp = srcp - diff;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const __m128i neighbour_pixel = _mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + x)), zeroes);
            const __m128i mask = _mm_cmpgt_epi32(thres, abs(_mm_sub_epi32(center_pixel, neighbour_pixel)));
            sum = _mm_add_epi32(sum, _mm_and_si128(neighbour_pixel, mask));
            count = _mm_sub_epi32(count, mask);
        }

        srcp += stride;
    }

    sum = _mm_add_epi32(sum, _mm_srli_epi32(_mm_slli_epi32(count, 8), 1));

    __m128i divres_hi = [&]() {
        divres_hi = insert(divres_hi, divin[extract(count, 0)], 0);
        return insert(divres_hi, divin[extract(count, 1)], 1);
    }();

    __m128i divres_lo = [&]() {
        divres_lo = insert(divres_lo, divin[extract(count, 2)], 0);
        return insert(divres_lo, divin[extract(count, 3)], 1);
    }();

    __m128i sum_hi = [&]() {
        sum_hi = insert_hi(sum_hi, extract(sum, 0), 0);
        return insert_hi(sum_hi, extract(sum, 1), 1);
    }();

    __m128i sum_lo = [&]() {
        sum_lo = insert_hi(sum_lo, extract(sum, 2), 0);
        return insert_hi(sum_lo, extract(sum, 3), 1);
    }();

    __m128i mul_hi = _mm_mul_epu32(_mm_shuffle_epi32(sum_hi, _MM_SHUFFLE(3, 1, 2, 0)), _mm_shuffle_epi32(divres_hi, _MM_SHUFFLE(3, 1, 2, 0)));
    __m128i mul_lo = _mm_mul_epu32(_mm_shuffle_epi32(sum_lo, _MM_SHUFFLE(3, 1, 2, 0)), _mm_shuffle_epi32(divres_lo, _MM_SHUFFLE(3, 1, 2, 0)));

    __m128i hi = _mm_unpacklo_epi16(mul_hi, zeroes);
    __m128i hi1 = _mm_unpackhi_epi16(mul_hi, zeroes);
    __m128i lo = _mm_unpacklo_epi16(mul_lo, zeroes);
    __m128i lo1 = _mm_unpackhi_epi16(mul_lo, zeroes);

    __m128i result = [&]() {
        result = insert(result, extract(hi, 1), 0);
        result = insert(result, extract(hi1, 1), 1);
        result = insert(result, extract(lo, 1), 2);
        return insert(result, extract(lo1, 1), 3);
    }();

    // Weight with original depending on edge value
    result = _mm_add_epi32(mul_u32_(center_pixel, _mm_sub_epi32(_mm_set1_epi32(65535), str), 65535.0f),
        mul_u32_(str, result, 65535.0f));

    _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp), packus(result, result));
}

template <bool interlaced, bool hqy, bool hqc>
void SmoothUV2::smoothN_SSE2(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env)
{
    void (SmoothUV2:: * sum_pixels)(const uint8_t * origsp, const uint16_t * srcp, uint16_t * dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int strength);
    void (SmoothUV2:: * sum_pix_c)(const uint8_t * origsp, const uint16_t * srcp, uint16_t * dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int);

    const int offs = (interlaced) ? 2 : 1;
    int planes_y[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    int planecount = std::min(vi.NumComponents(), 3);
    for (int i = 0; i < planecount; ++i)
    {
        int radius_w, radius_h;
        int thr;
        int strength;
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
            strength = strengthY;

            if constexpr (hqy)
            {
                sum_pixels = &SmoothUV2::sshiq_sum_pixels_SSE2;
                sum_pix_c = &SmoothUV2::sshiq_sum_pixels_c;
            }
            else
            {
                sum_pixels = &SmoothUV2::sum_pixels_SSE2;
                sum_pix_c = &SmoothUV2::sum_pixels_c;
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
            strength = strengthC;

            if constexpr (hqc)
            {
                sum_pixels = &SmoothUV2::sshiq_sum_pixels_SSE2;
                sum_pix_c = &SmoothUV2::sshiq_sum_pixels_c;
            }
            else
            {
                sum_pixels = &SmoothUV2::sum_pixels_SSE2;
                sum_pix_c = &SmoothUV2::sum_pixels_c;
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

                (this->*sum_pix_c)(origsp + (static_cast<int64_t>(x) << 1), srcp + x, dstp + x, stride, offset + x0, xn, yn, thr, strength);

                if (interlaced)
                    (this->*sum_pix_c)(origsp + (static_cast<int64_t>(x) << 1), srcp2 + x, dstp2 + x, stride, offset + x0, xn, yn, thr, strength);
            }

            for (int x = 16; x < col; x += 4)
            {
                const int xn = (radius_w << 1) + 1;

                (this->*sum_pixels)(origsp + (static_cast<int64_t>(x) << 1), srcp + x, dstp + x, stride, offset + radius_w, xn, yn, thr, strength);

                if (interlaced)
                    (this->*sum_pixels)(origsp + (static_cast<int64_t>(x) << 1), srcp2 + x, dstp2 + x, stride, offset + radius_w, xn, yn, thr, strength);
            }

            for (int x = col; x < w; ++x)
            {
                const int xn = (x + radius_w < w - 1) ? (radius_w << 1) + 1 : std::max(radius_w + (w - x), 1);

                (this->*sum_pix_c)(origsp + (static_cast<int64_t>(x) << 1), srcp + x, dstp + x, stride, offset + radius_w, xn, yn, thr, strength);

                if (interlaced)
                    (this->*sum_pix_c)(origsp + (static_cast<int64_t>(x) << 1), srcp2 + x, dstp2 + x, stride, offset + radius_w, xn, yn, thr, strength);
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

                (this->*sum_pix_c)(origsp + (static_cast<int64_t>(x) << 1), srcp + x, dstp + x, stride, offset + x0, x0 + radius_w + 1, yn, thr, strength);
            }

            for (int x = 16; x < col; x += 4)
                (this->*sum_pixels)(origsp + (static_cast<int64_t>(x) << 1), srcp + x, dstp + x, stride, offset + radius_w, (radius_w << 1) + 1, yn, thr, strength);

            for (int x = col; x < w; ++x)
                (this->*sum_pix_c)(origsp + (static_cast<int64_t>(x) << 1), srcp + x, dstp + x, stride, offset + radius_w, (x + radius_w < w - 1) ? (radius_w << 1) + 1 : std::max(radius_w + (w - x), 1), yn, thr, strength);
        }
    }
}

template void SmoothUV2::smoothN_SSE2<true, true, true>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
template void SmoothUV2::smoothN_SSE2<true, true, false>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
template void SmoothUV2::smoothN_SSE2<true, false, true>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
template void SmoothUV2::smoothN_SSE2<true, false, false>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);

template void SmoothUV2::smoothN_SSE2<false, true, true>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
template void SmoothUV2::smoothN_SSE2<false, true, false>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
template void SmoothUV2::smoothN_SSE2<false, false, true>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
template void SmoothUV2::smoothN_SSE2<false, false, false>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
