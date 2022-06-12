#include "SmoothUV2.h"

AVS_FORCEINLINE void SmoothUV2::sum_pixels_AVX512(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int)
{
    const Vec16i zeroes = zero_si512();
    Vec16i sum = zeroes;
    Vec16i count = zeroes;

    const Vec16i thres = sqrt(static_cast<int64_t>(threshold) * threshold / 3);
    const Vec16i center_pixel = Vec16i().load_16us(srcp);

    srcp = srcp - diff;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const Vec16i neighbour_pixel = Vec16i().load_16us(srcp + x);

            const Vec16i abs_diff = abs(center_pixel - neighbour_pixel);

            // Absolute difference less than thres
            const Vec16i mask = select(thres > abs_diff, Vec16i(0xFFFFFFFF), 0);

            // Sum up the pixels that meet the criteria
            sum = sum + (neighbour_pixel & mask);

            // Keep track of how many pixels are in the sum
            count = count - mask;
        }

        srcp += stride;
    }

    Vec16i divres = [&]() {
        divres = divres.insert(0, divin[count.extract(0)]);
        divres = divres.insert(1, divin[count.extract(1)]);
        divres = divres.insert(2, divin[count.extract(2)]);
        divres = divres.insert(3, divin[count.extract(3)]);
        divres = divres.insert(4, divin[count.extract(4)]);
        divres = divres.insert(5, divin[count.extract(5)]);
        divres = divres.insert(6, divin[count.extract(6)]);
        divres = divres.insert(7, divin[count.extract(7)]);
        divres = divres.insert(8, divin[count.extract(8)]);
        divres = divres.insert(9, divin[count.extract(9)]);
        divres = divres.insert(10, divin[count.extract(10)]);
        divres = divres.insert(11, divin[count.extract(11)]);
        divres = divres.insert(12, divin[count.extract(12)]);
        divres = divres.insert(13, divin[count.extract(13)]);
        divres = divres.insert(14, divin[count.extract(14)]);
        return divres.insert(15, divin[count.extract(15)]);
    }();

    const Vec16i result = truncatei(to_float((sum + (count >> 1))) * to_float(divres) / 65535.0f);
    const Vec32us dest = compress_saturated_s2u(result, result);
    dest.store(dstp);
}

AVS_FORCEINLINE void SmoothUV2::sshiq_sum_pixels_AVX512(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int strength)
{
    const Vec16i zeroes = zero_si512();
    Vec16i sum = zeroes;
    Vec16i count = zeroes;

    const Vec16i thres = sqrt(static_cast<int64_t>(threshold) * threshold / 3);

    // Build edge values
    const Vec16i center_pixel = Vec16i().load_16us(srcp);
    const Vec16i add = Vec16i().load_16us(origsp + (static_cast<int64_t>(stride) << 1)) + Vec16i().load_16us(origsp + 1);
    const Vec16i sllq = center_pixel << 1;

    // Store weight with edge bias
    const Vec16i str = strength - (sllq - add | add - sllq);

    srcp = srcp - diff;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const Vec16i neighbour_pixel = Vec16i().load_16us(srcp + x);

            const Vec16i abs_diff = abs(center_pixel - neighbour_pixel);

            // Absolute difference less than thres
            const Vec16i mask = select(thres > abs_diff, Vec16i(0xFFFFFFFF), 0);

            // Sum up the pixels that meet the criteria
            sum = sum + (neighbour_pixel & mask);

            // Keep track of how many pixels are in the sum
            count = count - mask;
        }

        srcp += stride;
    }

    Vec16i divres = [&]() {
        divres = divres.insert(0, divin[count.extract(0)]);
        divres = divres.insert(1, divin[count.extract(1)]);
        divres = divres.insert(2, divin[count.extract(2)]);
        divres = divres.insert(3, divin[count.extract(3)]);
        divres = divres.insert(4, divin[count.extract(4)]);
        divres = divres.insert(5, divin[count.extract(5)]);
        divres = divres.insert(6, divin[count.extract(6)]);
        divres = divres.insert(7, divin[count.extract(7)]);
        divres = divres.insert(8, divin[count.extract(8)]);
        divres = divres.insert(9, divin[count.extract(9)]);
        divres = divres.insert(10, divin[count.extract(10)]);
        divres = divres.insert(11, divin[count.extract(11)]);
        divres = divres.insert(12, divin[count.extract(12)]);
        divres = divres.insert(13, divin[count.extract(13)]);
        divres = divres.insert(14, divin[count.extract(14)]);
        return divres.insert(15, divin[count.extract(15)]);
    }();

    // Weight with original depending on edge value
    const Vec16i result = truncatei(to_float(center_pixel) * to_float(65535 - str) / 65535.0f + to_float(str) * to_float((sum + (count >> 1))) * to_float(divres) / 65535.0f / 65535.0f);
    const Vec32us dest = compress_saturated_s2u(result, result);
    dest.store(dstp);
}

template <bool interlaced, bool hqy, bool hqc>
void SmoothUV2::smoothN_AVX512(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env)
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
                sum_pixels = &SmoothUV2::sshiq_sum_pixels_AVX512;
                sum_pix_c = &SmoothUV2::sshiq_sum_pixels_c;
            }
            else
            {
                sum_pixels = &SmoothUV2::sum_pixels_AVX512;
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
                sum_pixels = &SmoothUV2::sshiq_sum_pixels_AVX512;
                sum_pix_c = &SmoothUV2::sshiq_sum_pixels_c;
            }
            else
            {
                sum_pixels = &SmoothUV2::sum_pixels_AVX512;
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
        const int col = (w - 16) - ((w - 16) % 16);

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

            for (int x = 16; x < col; x += 16)
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

            for (int x = 16; x < col; x += 16)
                (this->*sum_pixels)(origsp + (static_cast<int64_t>(x) << 1), srcp + x, dstp + x, stride, offset + radius_w, (radius_w << 1) + 1, yn, thr, strength);

            for (int x = col; x < w; ++x)
                (this->*sum_pix_c)(origsp + (static_cast<int64_t>(x) << 1), srcp + x, dstp + x, stride, offset + radius_w, (x + radius_w < w - 1) ? (radius_w << 1) + 1 : std::max(radius_w + (w - x), 1), yn, thr, strength);
        }
    }
}

template void SmoothUV2::smoothN_AVX512<true, true, true>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
template void SmoothUV2::smoothN_AVX512<true, true, false>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
template void SmoothUV2::smoothN_AVX512<true, false, true>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
template void SmoothUV2::smoothN_AVX512<true, false, false>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);

template void SmoothUV2::smoothN_AVX512<false, true, true>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
template void SmoothUV2::smoothN_AVX512<false, true, false>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
template void SmoothUV2::smoothN_AVX512<false, false, true>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
template void SmoothUV2::smoothN_AVX512<false, false, false>(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env);
