#include "SmoothUV2.h"

AVS_FORCEINLINE void SmoothUV2::sum_pixels_c(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int)
{
    int64_t sum = 0;
    int count = 0;

    const int thres = sqrt(static_cast<int64_t>(threshold) * threshold / 3);
    const int center_pixel = srcp[0];

    srcp = srcp - diff;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const int neighbour_pixel = srcp[x];

            const int abs_diff = std::abs(center_pixel - neighbour_pixel);

            // Absolute difference less than thres
            const int mask = (thres > abs_diff) ? -1 : 0;

            // Sum up the pixels that meet the criteria
            sum += (neighbour_pixel & mask);

            // Keep track of how many pixels are in the sum
            count -= mask;
        }

        srcp += stride;
    }

    *(dstp) = (sum + (count >> 1)) * divin[count] / 65535;
}

AVS_FORCEINLINE void SmoothUV2::sshiq_sum_pixels_c(const uint8_t* origsp, const uint16_t* srcp, uint16_t* dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int strength)
{
    int64_t sum = 0;
    int count = 0;

    const int thres = sqrt(static_cast<int64_t>(threshold) * threshold / 3);

    // Build edge values
    const int center_pixel = srcp[0];
    const int add = *(origsp + (static_cast<int64_t>(stride) << 1)) + *(origsp + 1);
    const int sllq = center_pixel << 1;

    // Store weight with edge bias
    const int str = strength - ((sllq - add) | (add - sllq));

    srcp = srcp - diff;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const int neighbour_pixel = srcp[x];

            const int abs_diff = std::abs(center_pixel - neighbour_pixel);

            // Absolute difference less than thres
            const int mask = (thres > abs_diff) ? -1 : 0;

            // Sum up the pixels that meet the criteria
            sum += (neighbour_pixel & mask);

            // Keep track of how many pixels are in the sum
            count -= mask;
        }

        srcp += stride;
    }

    *(dstp) = center_pixel * (65535 - str) / 65535 + str * (sum + (count >> 1)) * divin[count] / 65535 / 65535;
}

template <bool interlaced, bool hqy, bool hqc>
void SmoothUV2::smoothN_c(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment* env)
{
    void (SmoothUV2:: * sum_pixels)(const uint8_t * origsp, const uint16_t * srcp, uint16_t * dstp, const int stride, const int diff, const int width, const int height, const int threshold, const int strength);

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
                sum_pixels = &SmoothUV2::sshiq_sum_pixels_c;
            else
                sum_pixels = &SmoothUV2::sum_pixels_c;

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
                sum_pixels = &SmoothUV2::sshiq_sum_pixels_c;
            else
                sum_pixels = &SmoothUV2::sum_pixels_c;
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

        if (interlaced)
        {
            stride *= 2;
            dst_stride *= 2;
            h2 >>= 1;
        }

        for (int y = 0; y < h2; ++y)
        {
            const int y0 = (y < radius_h) ? y : radius_h;

            int yn = (y < h2 - radius_h) ? y0 + radius_h + 1 * offs : y0 + (h2 - y);

            if (interlaced)
                yn--;

            const int offset = y0 * stride;

            for (int x = 0; x < w; ++x)
            {
                const int x0 = (x < radius_w) ? x : radius_w;

                const int xn = (x + radius_w < w - 1) ? x0 + radius_w + 1 : std::max(x0 + (w - x), 1);

                (this->*sum_pixels)(origsp + (static_cast<int64_t>(x) << 1), srcp + x, dstp + x, stride, offset + x0, xn, yn, thr, strength);

                if (interlaced)
                    (this->*sum_pixels)(origsp + (static_cast<int64_t>(x) << 1), srcp2 + x, dstp2 + x, stride, offset + x0, xn, yn, thr, strength);
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

            for (int x = 0; x < w; ++x)
            {
                const int x0 = (x < radius_w) ? x : radius_w;

                const int xn = (x + radius_w < w - 1) ? x0 + radius_w + 1 : std::max(x0 + (w - x), 1);

                (this->*sum_pixels)(origsp + (static_cast<int64_t>(x) << 1), srcp + x, dstp + x, stride, offset + x0, xn, yn, thr, strength);
            }
        }
    }
}

SmoothUV2::SmoothUV2(PClip _child, int radiusY, int radiusC, int thresholdY, int thresholdC, int strY, int strC, bool HQY, bool HQC, int interlaced, int opt, IScriptEnvironment* env)
    : GenericVideoFilter(_child), radiusy(radiusY), strengthY(strY), strengthC(strC), hqy(HQY), hqc(HQC), _interlaced(interlaced)
{
    if (thresholdY != -1337)
    {
        if (radiusY < 1 || radiusY > 7)
            env->ThrowError("SSHiQ2: radiusY must be between 1 and 7 (inclusive).");
        if (vi.Is420() && (radiusC < 1 || radiusC > 7))
            env->ThrowError("SSHiQ2: radiusC must be between 1 and 7 (inclusive).");
        if (!vi.Is420() && (radiusC < 1 || radiusC > 3))
            env->ThrowError("SSHiQ2: radiusC must be between 1 and 3 (inclusive) for subsampling other than 4:2:0.");
        if (thresholdY < 0 || thresholdY > 450)
            env->ThrowError("SSHiQ2: thresholdY must be between 0 and 450 (inclusive).");
        if (thresholdC < 0 || thresholdC > 450)
            env->ThrowError("SSHiQ2: thresholdC must be between 0 and 450 (inclusive).");
        if (strengthY < 0 || strengthY > 255)
            env->ThrowError("SSHiQ2: strY must be between 0 and 255 (inclusive).");
        if (strengthC < 0 || strengthC > 255)
            env->ThrowError("SSHiQ2: strC must be between 0 and 255 (inclusive).");
        if (_interlaced < -1 || _interlaced > 1)
            env->ThrowError("SSHiQ2: interlaced must be between -1 and 1 (inclusive).");
        if (opt < -1 || opt > 5)
            env->ThrowError("SSHiQ2: opt must be between -1 and 5 (inclusive).");
        if (opt == 1 && !(env->GetCPUFlags() & CPUF_SSE2))
            env->ThrowError("SSHiQ2: opt=1 requires SSE2.");
        if (opt == 2 && !(env->GetCPUFlags() & CPUF_SSSE3))
            env->ThrowError("SSHiQ2: opt=2 requires SSSE3.");
        if (opt == 3 && !(env->GetCPUFlags() & CPUF_SSE4_1))
            env->ThrowError("SSHiQ2: opt=3 requires SSE4.1.");
        if (opt == 4 && !(env->GetCPUFlags() & CPUF_AVX2))
            env->ThrowError("SSHiQ2: opt=4 requires AVX2.");
        if (opt == 5 && !(env->GetCPUFlags() & CPUF_AVX512F))
            env->ThrowError("SSHiQ2: opt=5 requires AVX512F.");
    }
    else
    {
        if (vi.Is420() && (radiusC < 1 || radiusC > 7))
            env->ThrowError("SmoothUV2: radius must be between 1 and 7 (inclusive).");
        if (!vi.Is420() && (radiusC < 1 || radiusC > 3))
            env->ThrowError("SmoothUV2: radius must be between 1 and 3 (inclusive) for subsampling other than 4:2:0.");
        if (thresholdC < 0 || thresholdC > 450)
            env->ThrowError("SmoothUV2: threshold must be between 0 and 450 (inclusive).");
        if (_interlaced < -1 || _interlaced > 1)
            env->ThrowError("SmoothUV2: interlaced must be between -1 and 1 (inclusive).");
        if (opt < -1 || opt > 5)
            env->ThrowError("SmoothUV2: opt must be between -1 and 5 (inclusive).");
        if (opt == 1 && !(env->GetCPUFlags() & CPUF_SSE2))
            env->ThrowError("SmoothUV2: opt=1 requires SSE2.");
        if (opt == 2 && !(env->GetCPUFlags() & CPUF_SSSE3))
            env->ThrowError("SmoothUV2: opt=2 requires SSSE3.");
        if (opt == 3 && !(env->GetCPUFlags() & CPUF_SSE4_1))
            env->ThrowError("SmoothUV2: opt=3 requires SSE4.1.");
        if (opt == 4 && !(env->GetCPUFlags() & CPUF_AVX2))
            env->ThrowError("SmoothUV2: opt=4 requires AVX2.");
        if (opt == 5 && !(env->GetCPUFlags() & CPUF_AVX512F))
            env->ThrowError("SmoothUV2: opt=5 requires AVX512F.");
    }

    _thresholdY = (thresholdY != -1337) ? (thresholdY * 65535 / 255) : thresholdY;
    _thresholdC = thresholdC * 65535 / 255;

    if (_thresholdY != -1337)
    {
        strengthY *= 65535 / 255;
        strengthC *= 65535 / 255;
    }

    radiuscw = radiusC << (1 - vi.GetPlaneWidthSubsampling(PLANAR_U));
    radiusch = radiusC << (1 - vi.GetPlaneHeightSubsampling(PLANAR_U));

    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; }

    divin = std::make_unique<uint16_t[]>(257);

    for (int i = 1; i < 257; ++i)
        divin[i] = static_cast<uint16_t>(65535 / i);

    sse2 = (opt < 0 && !!(env->GetCPUFlags() & CPUF_SSE2)) || opt == 1;
    ssse3 = (opt < 0 && !!(env->GetCPUFlags() & CPUF_SSSE3)) || opt == 2;
    sse41 = (opt < 0 && !!(env->GetCPUFlags() & CPUF_SSE4_1)) || opt == 3;
    avx2 = (opt < 0 && !!(env->GetCPUFlags() & CPUF_AVX2)) || opt == 4;
    avx512 = (opt < 0 && !!(env->GetCPUFlags() & CPUF_AVX512F)) || opt == 5;

    if (_interlaced > -1)
        field_based = _interlaced;
    else
    {
        if (!has_at_least_v8)
            field_based = 0;
    }

    if (!has_at_least_v8)
    {
        if (avx512)
        {
            if (HQY)
            {
                if (HQC)
                    smooth = (field_based) ? &SmoothUV2::smoothN_AVX512<true, true, true> : &SmoothUV2::smoothN_AVX512<false, true, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_AVX512<true, true, false> : &SmoothUV2::smoothN_AVX512<false, true, false>;
            }
            else
            {
                if (HQC)
                    smooth = (field_based) ? &SmoothUV2::smoothN_AVX512<true, false, true> : &SmoothUV2::smoothN_AVX512<false, false, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_AVX512<true, false, false> : &SmoothUV2::smoothN_AVX512<false, false, false>;
            }
        }
        else if (avx2)
        {
            if (HQY)
            {
                if (HQC)
                    smooth = (field_based) ? &SmoothUV2::smoothN_AVX2<true, true, true> : &SmoothUV2::smoothN_AVX2<false, true, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_AVX2<true, true, false> : &SmoothUV2::smoothN_AVX2<false, true, false>;
            }
            else
            {
                if (HQC)
                    smooth = (field_based) ? &SmoothUV2::smoothN_AVX2<true, false, true> : &SmoothUV2::smoothN_AVX2<false, false, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_AVX2<true, false, false> : &SmoothUV2::smoothN_AVX2<false, false, false>;
            }
        }
        else if (sse41)
        {
            if (HQY)
            {
                if (HQC)
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<true, true, true> : &SmoothUV2::smoothN_SSE41<false, true, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<true, true, false> : &SmoothUV2::smoothN_SSE41<false, true, false>;
            }
            else
            {
                if (HQC)
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<true, false, true> : &SmoothUV2::smoothN_SSE41<false, false, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<true, false, false> : &SmoothUV2::smoothN_SSE41<false, false, false>;
            }
        }
        else if (ssse3)
        {
            if (HQY)
            {
                if (HQC)
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSSE3<true, true, true> : &SmoothUV2::smoothN_SSSE3<false, true, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSSE3<true, true, false> : &SmoothUV2::smoothN_SSSE3<false, true, false>;
            }
            else
            {
                if (HQC)
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSSE3<true, false, true> : &SmoothUV2::smoothN_SSSE3<false, false, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSSE3<true, false, false> : &SmoothUV2::smoothN_SSSE3<false, false, false>;
            }
        }
        else if (sse2)
        {
            if (HQY)
            {
                if (HQC)
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE2<true, true, true> : &SmoothUV2::smoothN_SSE2<false, true, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE2<true, true, false> : &SmoothUV2::smoothN_SSE2<false, true, false>;
            }
            else
            {
                if (HQC)
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE2<true, false, true> : &SmoothUV2::smoothN_SSE2<false, false, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE2<true, false, false> : &SmoothUV2::smoothN_SSE2<false, false, false>;
            }
        }
        else
        {
            if (HQY)
            {
                if (HQC)
                    smooth = (field_based) ? &SmoothUV2::smoothN_c<true, true, true> : &SmoothUV2::smoothN_c<false, true, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_c<true, true, false> : &SmoothUV2::smoothN_c<false, true, false>;
            }
            else
            {
                if (HQC)
                    smooth = (field_based) ? &SmoothUV2::smoothN_c<true, false, true> : &SmoothUV2::smoothN_c<false, false, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_c<true, false, false> : &SmoothUV2::smoothN_c<false, false, false>;
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

        if (avx512)
        {
            if (hqy)
            {
                if (hqc)
                    smooth = (field_based) ? &SmoothUV2::smoothN_AVX512<true, true, true> : &SmoothUV2::smoothN_AVX512<false, true, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_AVX512<true, true, false> : &SmoothUV2::smoothN_AVX512<false, true, false>;
            }
            else
            {
                if (hqc)
                    smooth = (field_based) ? &SmoothUV2::smoothN_AVX512<true, false, true> : &SmoothUV2::smoothN_AVX512<false, false, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_AVX512<true, false, false> : &SmoothUV2::smoothN_AVX512<false, false, false>;
            }
        }
        else if (avx2)
        {
            if (hqy)
            {
                if (hqc)
                    smooth = (field_based) ? &SmoothUV2::smoothN_AVX2<true, true, true> : &SmoothUV2::smoothN_AVX2<false, true, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_AVX2<true, true, false> : &SmoothUV2::smoothN_AVX2<false, true, false>;
            }
            else
            {
                if (hqc)
                    smooth = (field_based) ? &SmoothUV2::smoothN_AVX2<true, false, true> : &SmoothUV2::smoothN_AVX2<false, false, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_AVX2<true, false, false> : &SmoothUV2::smoothN_AVX2<false, false, false>;
            }
        }
        else if (sse41)
        {
            if (hqy)
            {
                if (hqc)
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<true, true, true> : &SmoothUV2::smoothN_SSE41<false, true, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<true, true, false> : &SmoothUV2::smoothN_SSE41<false, true, false>;
            }
            else
            {
                if (hqc)
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<true, false, true> : &SmoothUV2::smoothN_SSE41<false, false, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE41<true, false, false> : &SmoothUV2::smoothN_SSE41<false, false, false>;
            }
        }
        else if (ssse3)
        {
            if (hqy)
            {
                if (hqc)
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSSE3<true, true, true> : &SmoothUV2::smoothN_SSSE3<false, true, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSSE3<true, true, false> : &SmoothUV2::smoothN_SSSE3<false, true, false>;
            }
            else
            {
                if (hqc)
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSSE3<true, false, true> : &SmoothUV2::smoothN_SSSE3<false, false, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSSE3<true, false, false> : &SmoothUV2::smoothN_SSSE3<false, false, false>;
            }
        }
        else if (sse2)
        {
            if (hqy)
            {
                if (hqc)
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE2<true, true, true> : &SmoothUV2::smoothN_SSE2<false, true, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE2<true, true, false> : &SmoothUV2::smoothN_SSE2<false, true, false>;
            }
            else
            {
                if (hqc)
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE2<true, false, true> : &SmoothUV2::smoothN_SSE2<false, false, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_SSE2<true, false, false> : &SmoothUV2::smoothN_SSE2<false, false, false>;
            }
        }
        else
        {
            if (hqy)
            {
                if (hqc)
                    smooth = (field_based) ? &SmoothUV2::smoothN_c<true, true, true> : &SmoothUV2::smoothN_c<false, true, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_c<true, true, false> : &SmoothUV2::smoothN_c<false, true, false>;
            }
            else
            {
                if (hqc)
                    smooth = (field_based) ? &SmoothUV2::smoothN_c<true, false, true> : &SmoothUV2::smoothN_c<false, false, true>;
                else
                    smooth = (field_based) ? &SmoothUV2::smoothN_c<true, false, false> : &SmoothUV2::smoothN_c<false, false, false>;
            }
        }
    }

    (this->*smooth)(dst, src, env);

    return dst;
}

AVSValue __cdecl Create_SmoothUV2(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    PClip clip = args[0].AsClip();
    const VideoInfo& vi = clip->GetVideoInfo();
    const int bits = vi.BitsPerComponent();

    if (vi.IsRGB() || vi.BitsPerComponent() == 32 || vi.NumComponents() < 3 || !vi.IsPlanar())
        env->ThrowError("SmoothUV2: only 8..16-bit YUV planar format supported with minimum three planes.");

    const bool convert = bits < 16;

    if (convert)
    {
        AVSValue args_[2] = { clip, 16 };
        clip = env->Invoke("ConvertBits", AVSValue(args_, 2)).AsClip();
    }

    clip = new SmoothUV2(
        clip,
        1,
        args[1].AsInt(3),
        -1337,
        args[2].AsInt(270),
        255,
        255,
        false,
        false,
        args[3].AsInt(-1),
        args[4].AsInt(-1),
        env);

    if (convert)
    {
        const char* names[3] = { NULL, NULL, "dither" };
        AVSValue args_[3] = { clip, bits, args[5].AsInt(-1) };
        clip = env->Invoke("ConvertBits", AVSValue(args_, 3), names).AsClip();
    }

    return clip;

}

AVSValue __cdecl Create_SSHiQ2(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    PClip clip = args[0].AsClip();
    const VideoInfo& vi = clip->GetVideoInfo();
    const int bits = vi.BitsPerComponent();

    if (vi.IsRGB() || vi.BitsPerComponent() == 32 || vi.NumComponents() < 3 || !vi.IsPlanar())
        env->ThrowError("SSHiQ2: only 8..16-bit YUV planar format supported with minimum three planes.");

    const bool convert = bits < 16;

    if (convert)
    {
        AVSValue args_[2] = { clip, 16 };
        clip = env->Invoke("ConvertBits", AVSValue(args_, 2)).AsClip();
    }

    const int strY = args[5].AsInt(240);

    clip = new SmoothUV2(
        clip,
        args[1].AsInt(5),
        args[2].AsInt(3),
        args[3].AsInt(20),
        args[4].AsInt(30),
        strY,
        args[6].AsInt(strY),
        args[7].AsBool(true),
        args[8].AsBool(true),
        args[9].AsInt(-1),
        args[10].AsInt(-1),
        env);

    if (convert)
    {
        const char* names[3] = { NULL, NULL, "dither" };
        AVSValue args_[3] = { clip, bits, args[11].AsInt(-1) };
        clip = env->Invoke("ConvertBits", AVSValue(args_, 3), names).AsClip();
    }

    return clip;
}

const AVS_Linkage* AVS_linkage;

extern "C" __declspec(dllexport)
const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    env->AddFunction("SmoothUV2", "c[radius]i[threshold]i[interlaced]i[opt]i[dither]i", Create_SmoothUV2, 0);
    env->AddFunction("SSHiQ2", "c[rY]i[rC]i[tY]i[tC]i[strY]i[strC]i[HQY]b[HQC]b[interlaced]i[opt]i[dither]i", Create_SSHiQ2, 0);

    return "SmoothUV2";
}
