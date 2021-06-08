#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <math.h>

#include "avisynth.h"
#include "SmoothFunc.h"

void (*smoothY) (int N, const unsigned char *src, unsigned char *dst, const int src_pitch,
				const int dst_pitch, int width, int height, const int t,
				unsigned short *div, unsigned char str);
void (*smoothC) (int N, const unsigned char *src, unsigned char *dst, const int src_pitch,
				const int dst_pitch, int width, int height, const int t,
				unsigned short *div, unsigned char str);

class SSmooth : public GenericVideoFilter {
private:
	bool			interlaced;
	int				DY, DC;
	int				Yth, Cth;
	unsigned short	divin[256];
	unsigned char	strength;
public:
	
	SSmooth(PClip _child, int radY, int radC, int thY, int thC, int str, bool hqy,
			bool hqc, bool field, IScriptEnvironment* env) : GenericVideoFilter(_child),
			Yth(thY), Cth(thC), interlaced(field)
	{
		if (!(env->GetCPUFlags() & CPUF_MMX))
		{
			if (thY == -1337) env->ThrowError("SmoothUV: requires a MMX-capable CPU");
			else env->ThrowError("SSHiQ: requires a MMX-capable CPU");
		}
		if (str<0 || str>255)
			env->ThrowError("SSHiQ: invalid strength");
		strength = str;

		DY = (radY<<1)+1;
		DC = (radC<<1)+1;
		if (DY<3 || DY>11)
			env->ThrowError("SSHiQ: invalid luma radius %",radY);
		if (DC<3 || DC>11)
		{
			if (thY != -1337) env->ThrowError("SSHiQ: invalid chroma radius %",radC);
			else env->ThrowError("SmoothUV: invalid chroma radius %",radC);
		}

		if (hqy) smoothY = smoothHQN_MMX;
		else smoothY = smoothN_MMX;


		if (hqc) smoothC = smoothHQN_MMX;
		else smoothC = smoothN_MMX;

		if (field)
		{
			if (thY > 0)
			{
				if (hqy) smoothY = smoothHQN_field_MMX;
				else smoothY = smoothN_field_MMX;
			}
			if (thC > 0)
			{
				if (hqc) smoothC = smoothHQN_field_MMX;
				else smoothC = smoothN_field_MMX;
			}
		}

		if (thY>450) env->ThrowError("SSHiQ: invalid Y threshold");
		if (thY == -1337)
		{
			if (thC<0)
				env->ThrowError("SmoothUV: bad thresholds, no-op filter");
			if (thC>450)
				env->ThrowError("SmoothUV: invalid chroma threshold");
		}
		/* We'll suppose the CPU hasn't changed since the start of the function */

		if (thC<=0)
		{
			if (env->GetCPUFlags() & CPUF_INTEGER_SSE) smoothC = asm_BitBlt_ISSE;
			else smoothC = asm_BitBlt_MMX;
		}
		if (thY<=0)
		{
			if (env->GetCPUFlags() & CPUF_INTEGER_SSE) smoothY = asm_BitBlt_ISSE;
			else smoothY = asm_BitBlt_MMX;
		}

		if (thC>450)
		{
			if (thY=-1337) env->ThrowError("SmoothUV: invalid chroma threshold");
			else env->ThrowError("SSHiQ: invalid U threshold");
		}
		
		/* 0 should be impossible to reach... though ...
		   Radii >= 5 are busted for now */
		divin[0] = 32767;
		for (unsigned short i=1; i<15*15; i++)
			divin[i] = (unsigned short)floor( 32767.0/i + 0.5 );

	}
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
};

PVideoFrame __stdcall SSmooth::GetFrame(int n, IScriptEnvironment* env)
{
	PVideoFrame 		src = child->GetFrame(n, env);
	PVideoFrame 		dst = env->NewVideoFrame(vi);
	
	const int			row_size = src->GetRowSize(PLANAR_Y);
	const int			height = src->GetHeight(PLANAR_Y);
	
	smoothY(DY, src->GetReadPtr(PLANAR_Y), dst->GetWritePtr(PLANAR_Y),
			src->GetPitch(PLANAR_Y), dst->GetPitch(PLANAR_Y),
			row_size, height, Yth, divin, strength);
	smoothC(DC, src->GetReadPtr(PLANAR_U), dst->GetWritePtr(PLANAR_U),
			src->GetPitch(PLANAR_U), dst->GetPitch(PLANAR_U),
			row_size>>1, height>>1, Cth, divin, strength);
	smoothC(DC, src->GetReadPtr(PLANAR_V), dst->GetWritePtr(PLANAR_V),
			src->GetPitch(PLANAR_V), dst->GetPitch(PLANAR_V),
			row_size>>1, height>>1, Cth, divin, strength);

	return dst;
}


AVSValue __cdecl Create_SmoothIQ(AVSValue args, void* user_data, IScriptEnvironment* env) 
{
	int 	radiusC = 3, thresC = 270, strength = 255;
	bool	field = false, hqy = false, hqc = false;
		  
	return new SSmooth(
		args[0].AsClip(),
		1,							//Doesn't apply anyway
		args[1].AsInt(radiusC),
		-1337,						//Doesn't apply anyway
		args[2].AsInt(thresC),
		255,						//Doesn't apply anyway
		false,
		false,
		args[3].AsBool(field),
		env);
}

AVSValue __cdecl Create_SSHiQ(AVSValue args, void* user_data, IScriptEnvironment* env) 
{
	int 	radiusY = 5, radiusC = 3;
	int		thresY = 20, thresC = 30;
	int		strength = 240;
	bool	hqy = true, hqc = true, interlaced = false;
		  
	return new SSmooth(
		args[0].AsClip(),
		args[1].AsInt(radiusY),
		args[2].AsInt(radiusC),
		args[3].AsInt(thresY),
		args[4].AsInt(thresC),
		args[5].AsInt(strength),
		args[6].AsBool(hqy),
		args[7].AsBool(hqc),
		args[8].AsBool(interlaced),
		env);
}

extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit2(IScriptEnvironment* env) {
	env->AddFunction("SmoothUV", "c[radius]i[T]i[field]b", Create_SmoothIQ, 0);
	env->AddFunction("SSHiQ", "c[rY]i[rC]i[tY]i[tC]i[str]i[HQY]b[HQC]b[field]b",
					Create_SSHiQ, 0);
	return "`SmartSmoother' spatial denoiser and derainbower";
}
