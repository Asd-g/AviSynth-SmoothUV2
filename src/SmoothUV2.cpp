#include <algorithm>

#include <emmintrin.h>

#include <avisynth.h>

static inline void sum_pixels_SSE2(const uint8_t* srcp, uint8_t* dstp, const int stride,
	const int diff, const int width, const int height,
	const __m128i& thres,
	const uint16_t* divinp) {

	__m128i zeroes = _mm_setzero_si128();

	__m128i sum = zeroes;
	__m128i count = zeroes;

	__m128i center_pixel = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)srcp),
		zeroes);

	srcp = srcp - diff;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x <= width; x++) {
			__m128i neighbour_pixel = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srcp + x)),
				zeroes);

			__m128i abs_diff = _mm_or_si128(_mm_subs_epu16(center_pixel, neighbour_pixel),
				_mm_subs_epu16(neighbour_pixel, center_pixel));

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

	sum = _mm_adds_epu16(sum,
		_mm_srli_epi16(count, 1));

	__m128i divres = zeroes;

	int e = _mm_extract_epi16(count, 0);
	divres = _mm_insert_epi16(divres, divinp[e], 0);
	e = _mm_extract_epi16(count, 1);
	divres = _mm_insert_epi16(divres, divinp[e], 1);
	e = _mm_extract_epi16(count, 2);
	divres = _mm_insert_epi16(divres, divinp[e], 2);
	e = _mm_extract_epi16(count, 3);
	divres = _mm_insert_epi16(divres, divinp[e], 3);
	e = _mm_extract_epi16(count, 4);
	divres = _mm_insert_epi16(divres, divinp[e], 4);
	e = _mm_extract_epi16(count, 5);
	divres = _mm_insert_epi16(divres, divinp[e], 5);
	e = _mm_extract_epi16(count, 6);
	divres = _mm_insert_epi16(divres, divinp[e], 6);
	e = _mm_extract_epi16(count, 7);
	divres = _mm_insert_epi16(divres, divinp[e], 7);

	// Now multiply (divres/65536)
	sum = _mm_mulhi_epu16(sum, divres);
	sum = _mm_packus_epi16(sum, sum);
	_mm_storel_epi64((__m128i*)dstp, sum);
}


template <bool interlaced>
static void smoothN_SSE2(int radius,
	const uint8_t* origsrc, uint8_t* origdst,
	int stride, int dst_stride, int w, int h,
	const int threshold,
	const uint16_t* divin) {
	const uint8_t* srcp = origsrc;
	const uint8_t* srcp2 = origsrc + stride;
	uint8_t* dstp = origdst;
	uint8_t* dstp2 = origdst + stride;

	const int SqrtTsquared = (int)sqrt((threshold * threshold) / 3);

	const __m128i thres = _mm_set1_epi16(SqrtTsquared);

	int h2 = h;

	if (interlaced)
	{
		stride *= 2;
		h2 >>= 1;
	}


	for (int y = 0; y < h2; y++) {
		int y0 = (y < radius) ? y : radius;

		int yn = (y < h2 - radius) ? y0 + radius + 1
			: y0 + (h2 - y);

		if (interlaced)
			yn--;

		int offset = y0 * stride;

		for (int x = 0; x < w; x += 8) {
			int x0 = (x < radius) ? x : radius;

			int xn = (x + 7 + radius < w - 1) ? x0 + radius + 1
				: x0 + w - x - 7;

			sum_pixels_SSE2(srcp + x, dstp + x,
				stride,
				offset + x0,
				xn, yn,
				thres,
				divin);

			if (interlaced) {
				sum_pixels_SSE2(srcp2 + x, dstp2 + x,
					stride,
					offset + x0,
					xn, yn,
					thres,
					divin);
			}
		}

		dstp += stride;
		srcp += stride;
		dstp2 += stride;
		srcp2 += stride;

	}

	if (interlaced && h % 1) {
		int yn = radius;

		int offset = radius * stride;

		for (int x = 0; x < w; x += 8) {
			int x0 = (x < radius) ? x : radius;

			int xn = (x + 7 + radius < w - 1) ? x0 + radius + 1
				: x0 + w - x - 7;

			sum_pixels_SSE2(srcp + x, dstp + x,
				stride,
				offset + x0,
				xn, yn,
				thres,
				divin);
		}
	}
}

template <bool interlaced>
static void smoothN_SSE2t(int radius,
	const uint8_t* origsrc, uint8_t* origdst,
	int stride, int dst_stride, int w, int h,
	const int threshold,
	const uint16_t* divin) {
	const uint8_t* srcp = origsrc;
	const uint8_t* srcp2 = origsrc + stride;
	uint8_t* dstp = origdst;
	uint8_t* dstp2 = origdst + dst_stride;

	const int SqrtTsquared = (int)sqrt((threshold * threshold) / 3);

	const __m128i thres = _mm_set1_epi16(SqrtTsquared);

	int h2 = h;

	for (int y = 0; y < h2; y++) {
		int y0 = (y < radius) ? y : radius;

		int yn = (y < h2 - radius) ? y0 + radius + 1
			: y0 + (h2 - y);

		int offset = y0 * stride;

		for (int x = 0; x < w; x += 8) {
			int x0 = (x < radius) ? x : radius;

			int xn = (x + 7 + radius < w - 1) ? x0 + radius + 1
				: x0 + w - x - 7;

			sum_pixels_SSE2(srcp + x, dstp + x,
				stride,
				offset + x0,
				xn, yn,
				thres,
				divin);

			if (interlaced) {
				sum_pixels_SSE2(srcp2 + x, dstp2 + x,
					stride,
					offset + x0,
					xn, yn,
					thres,
					divin);
			}
		}

		dstp += dst_stride;
		srcp += stride;
		dstp2 += dst_stride;
		srcp2 += stride;
	}

	if (interlaced && h % 1) {
		int yn = radius;

		int offset = radius * stride;

		for (int x = 0; x < w; x += 8) {
			int x0 = (x < radius) ? x : radius;

			int xn = (x + 7 + radius < w - 1) ? x0 + radius + 1
				: x0 + w - x - 7;

			sum_pixels_SSE2(srcp + x, dstp + x,
				stride,
				offset + x0,
				xn, yn,
				thres,
				divin);
		}
	}
}

static void copy_plane(PVideoFrame& dst, PVideoFrame& src, int plane, IScriptEnvironment* env) {
	const uint8_t* srcp = src->GetReadPtr(plane);
	int src_pitch = src->GetPitch(plane);
	int height = src->GetHeight(plane);
	int row_size = src->GetRowSize(plane);
	uint8_t* destp = dst->GetWritePtr(plane);
	int dst_pitch = dst->GetPitch(plane);
	env->BitBlt(destp, dst_pitch, srcp, src_pitch, row_size, height);
}

class SmoothUV : public GenericVideoFilter
{
	int _radius, _threshold;
	bool _interlaced;
	bool has_at_least_v8;

	uint16_t divin[256];

public:
	SmoothUV(PClip _child, int radius, int threshold, bool interlaced, IScriptEnvironment* env)
		: GenericVideoFilter(_child), _radius(radius), _threshold(threshold), _interlaced(interlaced)
	{
		has_at_least_v8 = true;
		try { env->CheckVersion(8); } catch (const AvisynthError&) { has_at_least_v8 = false; }

		for (int i = 1; i < 256; i++)
			divin[i] = (uint16_t)std::min((int)(65536.0 / i + 0.5), 65535);

		if (vi.BitsPerComponent() != 8 || vi.NumComponents() != 3)
		{
			env->ThrowError("SmoothUV: only 8 bit YUV with constant format supported.");
		}

		if (radius < 1 || radius > 7)
		{
			env->ThrowError("SmoothUV: radius must be between 1 and 7 (inclusive).");
		}

		if (threshold < 0 || threshold > 450)
		{
			env->ThrowError("SmoothUV: threshold must be between 0 and 450 (inclusive).");
		}
	}
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
	{
		PVideoFrame src = child->GetFrame(n, env);
		PVideoFrame dst;
		if (has_at_least_v8) dst = env->NewVideoFrameP(vi, &src); else dst = env->NewVideoFrame(vi);

		const uint8_t* srcp;
		uint8_t* dstp;
		int src_stride, dst_stride, width, height;
		bool interlaced;

		int planes_y[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
		for (int i = 0; i < 3; i++)
		{
			const int plane = planes_y[i];
			if (plane == 1)
			{
				copy_plane(dst, src, plane, env);
			}
			else
			{
				src_stride = src->GetPitch(plane);
				dst_stride = dst->GetPitch(plane);
				width = src->GetRowSize(plane);
				height = dst->GetHeight(plane);
				srcp = src->GetReadPtr(plane);
				dstp = dst->GetWritePtr(plane);

				interlaced = vi.IsFieldBased();
				interlaced = _interlaced;

				if (interlaced && src_stride == dst_stride)
				{
					(smoothN_SSE2<true>)(_radius, srcp, dstp, src_stride, dst_stride, width, height, _threshold, divin);
				}
				else if (interlaced && src_stride != dst_stride)
				{
					(smoothN_SSE2t<true>)(_radius, srcp, dstp, src_stride, dst_stride, width, height, _threshold, divin);					
				}
				else if (!interlaced && src_stride == dst_stride)
				{
					(smoothN_SSE2<false>)(_radius, srcp, dstp, src_stride, dst_stride, width, height, _threshold, divin);
				}
				else
				{
					env->ThrowError("SmoothUV: incompatible interlaced value");
				}
			}
		}

		return dst;
	}
};

AVSValue __cdecl Create_SmoothUV(AVSValue args, void* user_data, IScriptEnvironment* env)
{
	PClip clip = args[0].AsClip();
	const VideoInfo& vi = clip->GetVideoInfo();

	return new SmoothUV(args[0].AsClip(), args[1].AsInt(3), args[2].AsInt(270), args[3].AsBool(vi.IsFieldBased()), env);
}

const AVS_Linkage* AVS_linkage;

extern "C" __declspec(dllexport)
const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
	AVS_linkage = vectors;

	env->AddFunction("SmoothUV", "c[radius]i[threshold]i[interlaced]b", Create_SmoothUV, NULL);
	return "SmoothUV";
}