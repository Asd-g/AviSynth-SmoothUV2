#include <math.h>
#include <memory.h>

struct sum {
	int c_sum, count;
};

inline void sum_pixels(const unsigned char *src, int w2m1, int compareval,
					   unsigned char nucleus, struct sum *destarray)
{
	unsigned char	bc;
	int				csum = 0, num = 0;
	int				low = nucleus - compareval;
	int				hi  = nucleus + compareval;
	
	do
	{
		bc = *src++;
		if (bc <= hi && bc >= low)
		{
			csum += bc;
			num++;
		}
	} while(++w2m1);
	destarray->c_sum += csum;
	destarray->count += num;
}

inline void sumHiQ_pixels_MMX(const unsigned char *src, unsigned char *dst, const int pitch,
							  const int diff, const int width, const int height,
							  const __int64 thres, unsigned short *count, unsigned short *edge,
							  unsigned short *divres, unsigned short *divin, const __int64 strength)
{
	static const __int64 Keep01 = 0x0001000100010001i64;
	static const __int64 HiQ = 0x00FF00FF00FF00FFi64;
	__asm {
		//Prepair ptrs and registers
		pxor		mm7,mm7
		pxor		mm6,mm6
		pxor		mm5,mm5
		mov			ecx,[pitch]
		mov			esi,[src]
	
		//Build edge values
		movd		mm1,[esi+ecx]
		movd		mm0,[esi]
		movd		mm2,[esi+1]
		punpcklbw	mm1,mm7
		punpcklbw	mm2,mm7
		punpcklbw	mm0,mm7
		paddusw		mm1,mm2  s
		movq		mm4,mm0
		psllq		mm4,1 sllq
		movq		mm2,mm1
		psubusw		mm2,mm4 subs
		psubusw		mm4,mm1 subs1
		movq		mm3,[strength] str
		por			mm4,mm2 por
		
		//Store weight with edge bias
		//psllw		mm4,1
		psubusw		mm3,mm4 subs2
		//pxor		mm3,mm3
		
		movq		[strength],mm3
		
		sub			esi,diff
		mov			ebx,width
		mov			eax,height
		mov			edi,[count]
loopy:
		mov			edx,ebx
loopx:
		movd		mm1,[edx+esi] 
		movq		mm3,mm0
		punpcklbw	mm1,mm7 neighbour
		psubusw		mm3,mm1 subs3
		movq		mm4,thres thres
		movq		mm2,mm1
		psubusw		mm1,mm0 subs4
		por			mm3,mm1 por1
		pcmpgtw		mm4,mm3 pcm
		pand		mm2,mm4 and
		paddusw		mm6,mm2 sum
		pand		mm4,Keep01 and1
		paddusw		mm5,mm4
		
		dec			edx
		jnz			loopx
		
		add			esi,ecx
		dec			eax
		jnz			loopy
			
		movq		[edi],mm5
			
			//First word
		mov			esi,0
		mov			eax,[divin]
		mov			ebx,[divres]
		mov			si,[edi]
		mov			cx,[eax+2*esi]
		mov			[ebx],cx
			
		//2nd word
		add			edi,2
		add			ebx,2
		mov			si,[edi]
		mov			cx,[eax+2*esi]
		mov			[ebx],cx
		
		//3rd word
		add			edi,2
		add			ebx,2
		mov			si,[edi]
		mov			cx,[eax+2*esi]
		mov			[ebx],cx
		
		//4th word
		add			edi,2
		add			ebx,2
		mov			si,[edi]
		mov			cx,[eax+2*esi]
		mov			[ebx],cx

		//Address the signed multiply limitation
		sub			ebx,6
		movq		mm5,[ebx]
		psllw		mm6,1
		
		//Now multiply (divres/65536)
		movq		mm4,[strength]
		pmulhw		mm6,mm5 mulh
		
		//Now weight with original depending on edge value
		movq		mm3,[HiQ]
		psubusb		mm3,mm4 sub5
		
		psllw		mm4,4 sll
		psllw		mm6,4 sll1
		pmulhw		mm4,mm6 mul1
		psllw		mm3,4 sll2
		psllw		mm0,4 sll3
		pmulhw		mm0,mm3 mul2
		
		mov			eax,[dst]
		paddusw		mm0,mm4 add33
		packuswb	mm0,mm5
		
		movd		[eax],mm0
	}
}

inline void sum_pixels_MMX(const unsigned char *src, unsigned char *dst, const int pitch,
						   const int diff, const int width, const int height,
						   const __int64 thres, unsigned short *sum,
						   unsigned short *count, unsigned short *divres,
						   unsigned short *divin)
{
	static const __int64	Keep01 = 0x0001000100010001i64;
	__asm {
		pxor		mm7,mm7
		inc			width
		pxor		mm6,mm6
		pxor		mm5,mm5
		mov			ecx,[src]
		movd		mm0,[ecx]
		punpcklbw	mm0,mm7
		sub			ecx,diff
		mov			eax,height
		mov			edx,[count]
loopy:
		mov			esi,0
loopx:
		movd		mm1,[ecx+esi]
		movq		mm3,mm0
		punpcklbw	mm1,mm7
		psubusw		mm3,mm1
		movq		mm4,thres
		movq		mm2,mm1
		psubusw		mm1,mm0
		por			mm3,mm1
		pcmpgtw		mm4,mm3
		pand		mm2,mm4
		paddusw		mm6,mm2
		pand		mm4,Keep01
		paddusw		mm5,mm4
			
		inc			esi
		cmp			esi,width
		jne			loopx
			
		add			ecx,pitch
		dec			eax
		jnz			loopy
			
		movq		[edx],mm5
			
		//First word
		mov			esi,0
		mov			eax,[divin]
		mov			ebx,[divres]
		mov			si,[edx]
		mov			cx,[eax+2*esi]
		mov			[ebx],cx
		
		//2nd word
		add			edx,2
		add			ebx,2
		mov			si,[edx]
		mov			cx,[eax+2*esi]
		mov			[ebx],cx
		
		//3rd word
		add			edx,2
		add			ebx,2
		mov			si,[edx]
		mov			cx,[eax+2*esi]
		mov			[ebx],cx
		
		//4th word
		add			edx,2
		add			ebx,2
		mov			si,[edx]
		mov			cx,[eax+2*esi]
		mov			[ebx],cx
		
		//Address the signed multiply limitation
		sub			ebx,6
		movq		mm5,[ebx]
		psllw		mm6,1
		
		//Now multiply (divres/65536)
		mov			eax,[dst]
		pmulhw		mm6,mm5
		packuswb	mm6,mm5
		movd		[eax],mm6
	}
}

void smoothN_field_MMX(int N, const unsigned char *origsrc, unsigned char *origdst,
				   const int src_pitch, const int dst_pitch,
				   int w, int h, const int T, unsigned short *divin,
				   unsigned char str)
{	
	const unsigned char *srcp, *srcp2;
	unsigned char *dstp, *dstp2;
	
	const int Nover2 = N/2;
	const int s2pitch = src_pitch<<1, d2pitch = dst_pitch<<1;
	const int SqrtTsquared = (int)floor( sqrt((T * T)/3) );
	static const __int64	thres = ((__int64)SqrtTsquared<<48) + ((__int64)SqrtTsquared<<32) +
							((__int64)SqrtTsquared<<16) + (__int64)SqrtTsquared;
	static const __int64	strength =	((__int64)str<<48) + ((__int64)str<<32) +
								((__int64)str<<16) + (__int64)str;
	
	unsigned short		sum[4], count[4], divres[4];	
    int x, y, y0, x0, yn, offset, h2= h>>1;

	srcp = origsrc;
	dstp = origdst;
	srcp2 = origsrc + src_pitch;
	dstp2 = origdst + dst_pitch;
	for (y = 0; y < h2; y++)
	{
		if (y<Nover2) y0 = y;
		else y0 = Nover2;
		if (y<h2-Nover2) yn = y0 + Nover2;
		else yn = y0 + (h2-1-y);
		offset = y0*s2pitch;
		for (x = 0; x < w; x+=4)
		{
			if (x<Nover2) x0 = x;
			else x0 = Nover2;
			if (x+3+Nover2<w-1)
			{
				sum_pixels_MMX(srcp+x, dstp+x, s2pitch, offset+x0, x0 + Nover2 + 1 ,
							   yn, thres, sum, count, divres, divin);
				sum_pixels_MMX(srcp2+x, dstp2+x, s2pitch, offset+x0, x0 + Nover2 + 1,
							   yn, thres, sum, count, divres, divin);
			}
			else
			{
				sum_pixels_MMX(srcp+x, dstp+x, s2pitch, offset+x0, x0 + w - x-3, yn, thres,
								sum, count, divres, divin);
				sum_pixels_MMX(srcp2+x, dstp2+x, s2pitch, offset+x0, x0 + w - x-3, yn, thres,
								sum, count, divres, divin);
			}
		}
		dstp += d2pitch;
		srcp += s2pitch;
		dstp2 += d2pitch;
		srcp2 += s2pitch;
	}
	if (h%1)
	{
		y0 = yn = Nover2; //We don't need to do that
		offset = Nover2*s2pitch;
		for (x = 0; x < w; x+=4)
		{
			if (x<Nover2) x0 = x;
			else x0 = Nover2;
			if (x+3+Nover2<w-1)
				sum_pixels_MMX(srcp+x, dstp+x, s2pitch, offset+x0, x0 + Nover2 + 1,
							   yn, thres, sum, count, divres, divin);
			else
				sum_pixels_MMX(srcp+x, dstp+x, s2pitch, offset+x0, x0 + w - x-3,
							   yn, thres, sum, count, divres, divin);
		}
	}
	__asm emms;
}

void smoothHQN_field_MMX(int N, const unsigned char *origsrc, unsigned char *origdst,
				   const int src_pitch, const int dst_pitch,
				   int w, int h, const int T, unsigned short *divin,
				   unsigned char str)
{	
	const unsigned char *srcp, *srcp2;
	unsigned char *dstp, *dstp2;
	
	const int Nover2 = N/2;
	const int s2pitch = src_pitch<<1, d2pitch = dst_pitch<<1;
	const int SqrtTsquared = (int)floor( sqrt((T * T)/3) );
	static const __int64	thres = ((__int64)SqrtTsquared<<48) + ((__int64)SqrtTsquared<<32) +
							((__int64)SqrtTsquared<<16) + (__int64)SqrtTsquared;
	static const __int64	strength =	((__int64)str<<48) + ((__int64)str<<32) +
								((__int64)str<<16) + (__int64)str;
	
	unsigned short		sum[4], count[4], divres[4];	
    int x, y, y0, x0, yn, offset, h2 = h>>1;

	srcp = origsrc;
	dstp = origdst;
	srcp2 = origsrc + src_pitch;
	dstp2 = origdst + dst_pitch;
	for (y = 0; y < h2; y++)
	{
		if (y<Nover2) y0 = y;
		else y0 = Nover2;
		if (y<h2-Nover2) yn = y0 + Nover2 + 1;
		else yn = y0 + (h2-y);
		offset = y0*s2pitch;
		for (x = 0; x < w; x+=4)
		{
			if (x<Nover2) x0 = x;
			else x0 = Nover2;
			if (x+3+Nover2<w-1)
			{
				sumHiQ_pixels_MMX(srcp+x, dstp+x, s2pitch, offset+x0, x0 + Nover2 + 1,
								  yn, thres, sum, count, divres, divin, strength);
				sumHiQ_pixels_MMX(srcp2+x, dstp2+x, s2pitch, offset+x0, x0 + Nover2 + 1,
								  yn, thres, sum, count, divres, divin, strength);
			}
			else
			{
				sumHiQ_pixels_MMX(srcp+x, dstp+x, s2pitch, offset+x0, x0 + w - x-3,
								  yn, thres, sum, count, divres, divin, strength);
				sumHiQ_pixels_MMX(srcp2+x, dstp2+x, s2pitch, offset+x0, x0 + w - x-3,
								  yn, thres, sum, count, divres, divin, strength);
			}
		}
		dstp += d2pitch;
		srcp += s2pitch;
		dstp2 += d2pitch;
		srcp2 += s2pitch;
	}
	/* Don't forget the last odd line in chroma :) */
	if (h%1)
	{
		y0 = yn = Nover2; //We don't need to do that
		offset = Nover2*s2pitch;
		for (x = 0; x < w; x+=4)
		{
			if (x<Nover2) x0 = x;
			else x0 = Nover2;
			if (x+3+Nover2<w-1)
				sumHiQ_pixels_MMX(srcp+x, dstp+x, s2pitch, offset+x0, x0 + Nover2 + 1,
								  yn, thres, sum, count, divres, divin, strength);
			else
				sumHiQ_pixels_MMX(srcp+x, dstp+x, s2pitch, offset+x0, x0 + w - x-3,
								  yn, thres, sum, count, divres, divin, strength);
		}
	}
	__asm emms;
}

void smoothN_MMX(int N, const unsigned char *origsrc, unsigned char *origdst,
				 const int src_pitch, const int dst_pitch,
				 int w, int h, const int T, unsigned short *divin,
				 unsigned char str)
{
	const unsigned char	*srcp;
	unsigned char		*dstp;
	unsigned short		sum[4], count[4], divres[4];	
	
	int x, y, y0, x0, yn, offset;
	const int Nover2 = N>>1;
	const int Tsquared = (T * T)/3;
	const int SqrtTsquared = (int)floor( sqrt(Tsquared) );
	static const __int64	thres = ((__int64)SqrtTsquared<<48) + ((__int64)SqrtTsquared<<32) +
							((__int64)SqrtTsquared<<16) + (__int64)SqrtTsquared;
	static const __int64	strength =	((__int64)str<<48) + ((__int64)str<<32) +
								((__int64)str<<16) + (__int64)str;

	srcp = origsrc;
	dstp = origdst;

	for (y = 0; y < h; y++)
	{
		if (y<Nover2) y0 = y;
		else y0 = Nover2;
		if (y<h-Nover2) yn = y0 + Nover2 + 1;
		else yn = y0 + (h-y);
		offset = y0*src_pitch;
		for (x = 0; x < w; x+=4)
		{
			if (x<Nover2) x0 = x;
			else x0 = Nover2;
			if (x+3+Nover2<w-1)
				sum_pixels_MMX(srcp+x, dstp+x, src_pitch, offset+x0, x0 + Nover2 + 1,
							   yn, thres, sum, count, divres, divin);
			else
				sum_pixels_MMX(srcp+x, dstp+x, src_pitch, offset+x0, x0 + w - x-3,
							   yn, thres, sum, count, divres, divin);
		}
		dstp += dst_pitch;
		srcp += src_pitch;
	}
	__asm emms;
}

void smoothHQN_MMX(int N, const unsigned char *origsrc, unsigned char *origdst,
				   const int src_pitch, const int dst_pitch,
				   int w, int h, const int T, unsigned short *divin,
				   unsigned char str)
{
	const unsigned char	*srcp;
	unsigned char		*dstp;
	unsigned short		sum[4], count[4], divres[4];	
	
	int x, y, y0, x0, yn, offset;
	const int Nover2 = N>>1;
	const int Tsquared = (T * T)/3;
	const int SqrtTsquared = (int)floor( sqrt(Tsquared) );
	static const __int64	thres = ((__int64)SqrtTsquared<<48) + ((__int64)SqrtTsquared<<32) +
							((__int64)SqrtTsquared<<16) + (__int64)SqrtTsquared;
	static const __int64	strength =	((__int64)str<<48) + ((__int64)str<<32) +
								((__int64)str<<16) + (__int64)str;
	
	srcp = origsrc;
	dstp = origdst;
	for (y = 0; y < h; y++)
	{
		if (y<Nover2) y0 = y;
		else y0 = Nover2;
		if (y<h-Nover2) yn = y0 + Nover2 + 1;
		else yn = y0 + (h-y);
		offset = y0*src_pitch;
		for (x = 0; x < w; x+=4)
		{
			if (x<Nover2) x0 = x;
			else x0 = Nover2;
			if (x+3+Nover2<w-1)
				sumHiQ_pixels_MMX(srcp+x, dstp+x, src_pitch, offset+x0, x0 + Nover2 + 1,
								  yn, thres, sum, count, divres, divin, strength);
			else
				sumHiQ_pixels_MMX(srcp+x, dstp+x, src_pitch, offset+x0, x0 + w - x-3,
								  yn, thres, sum, count, divres, divin, strength);
		}
		dstp += dst_pitch;
		srcp += src_pitch;
	}
	__asm emms;
}

/*****************************
* Assembler bitblit by Steady
*****************************/


void asm_BitBlt_ISSE(int N, const unsigned char *srcp, unsigned char *dstp,
					 const int src_pitch, const int dst_pitch,
					 int row_size, int height, const int T, unsigned short *divin,
					 unsigned char str)
{
	if(row_size==0 || height==0) return; //abort on goofs
	//move backwards for easier looping and to disable hardware prefetch
	const unsigned char *srcStart=srcp+src_pitch*(height-1);
	unsigned char		*dstStart=dstp+dst_pitch*(height-1);
	
	if(row_size < 64) {
		_asm {
			mov   esi,srcStart  //move rows from bottom up
				mov   edi,dstStart
				mov   edx,row_size
				dec   edx
				mov   ebx,height
				align 16
memoptS_rowloop:
			mov   ecx,edx
				//      rep movsb
memoptS_byteloop:
			mov   AL,[esi+ecx]
				mov   [edi+ecx],AL
				sub   ecx,1
				jnc   memoptS_byteloop
				sub   esi,src_pitch
				sub   edi,dst_pitch
				dec   ebx
				jne   memoptS_rowloop
		};
		return;
	}//end small version
	
	else if( (int(dstp) | row_size | src_pitch | dst_pitch) & 7) {//not QW aligned
		//unaligned version makes no assumptions on alignment
		
		_asm {
			//****** initialize
			mov   esi,srcStart  //bottom row
				mov   AL,[esi]
				mov   edi,dstStart
				mov   edx,row_size
				mov   ebx,height
				
				//********** loop starts here ***********
				
				align 16
memoptU_rowloop:
			mov   ecx,edx     //row_size
				dec   ecx         //offset to last byte in row
				add   ecx,esi     //ecx= ptr last byte in row
				and   ecx,~63     //align to first byte in cache line
memoptU_prefetchloop:
			mov   AX,[ecx]    //tried AL,AX,EAX, AX a tiny bit faster
				sub   ecx,64
				cmp   ecx,esi
				jae   memoptU_prefetchloop
				
				//************ write *************
				
				movq    mm6,[esi]     //move the first unaligned bytes
				movntq  [edi],mm6
				//************************
				mov   eax,edi
				neg   eax
				mov   ecx,eax
				and   eax,63      //eax=bytes from [edi] to start of next 64 byte cache line
				and   ecx,7       //ecx=bytes from [edi] to next QW
				align 16
memoptU_prewrite8loop:        //write out odd QW's so 64 bit write is cache line aligned
			cmp   ecx,eax           //start of cache line ?
				jz    memoptU_pre8done  //if not, write single QW
				movq    mm7,[esi+ecx]
				movntq  [edi+ecx],mm7
				add   ecx,8
				jmp   memoptU_prewrite8loop
				
				align 16
memoptU_write64loop:
			movntq  [edi+ecx-64],mm0
				movntq  [edi+ecx-56],mm1
				movntq  [edi+ecx-48],mm2
				movntq  [edi+ecx-40],mm3
				movntq  [edi+ecx-32],mm4
				movntq  [edi+ecx-24],mm5
				movntq  [edi+ecx-16],mm6
				movntq  [edi+ecx- 8],mm7
memoptU_pre8done:
			add   ecx,64
				cmp   ecx,edx         //while(offset <= row_size) do {...
				ja    memoptU_done64
				movq    mm0,[esi+ecx-64]
				movq    mm1,[esi+ecx-56]
				movq    mm2,[esi+ecx-48]
				movq    mm3,[esi+ecx-40]
				movq    mm4,[esi+ecx-32]
				movq    mm5,[esi+ecx-24]
				movq    mm6,[esi+ecx-16]
				movq    mm7,[esi+ecx- 8]
				jmp   memoptU_write64loop
memoptU_done64:
			
			sub     ecx,64    //went to far
				align 16
memoptU_write8loop:
			add     ecx,8           //next QW
				cmp     ecx,edx         //any QW's left in row ?
				ja      memoptU_done8
				movq    mm0,[esi+ecx-8]
				movntq  [edi+ecx-8],mm0
				jmp   memoptU_write8loop
memoptU_done8:
			
			movq    mm1,[esi+edx-8] //write the last unaligned bytes
				movntq  [edi+edx-8],mm1
				sub   esi,src_pitch
				sub   edi,dst_pitch
				dec   ebx               //row counter (=height at start)
				jne   memoptU_rowloop
				
				sfence
				emms
		};
		return;
	}//end unaligned version
	
	else {//QW aligned version (fastest)
		//else dstp and row_size QW aligned - hope for the best from srcp
		//QW aligned version should generally be true when copying full rows
		_asm {
			mov   esi,srcStart  //start of bottom row
				mov   edi,dstStart
				mov   ebx,height
				mov   edx,row_size
				align 16
memoptA_rowloop:
			mov   ecx,edx //row_size
				dec   ecx     //offset to last byte in row
				
				//********forward routine
				add   ecx,esi
				and   ecx,~63   //align prefetch to first byte in cache line(~3-4% faster)
				align 16
memoptA_prefetchloop:
			mov   AX,[ecx]
				sub   ecx,64
				cmp   ecx,esi
				jae   memoptA_prefetchloop
				
				mov   eax,edi
				xor   ecx,ecx
				neg   eax
				and   eax,63            //eax=bytes from edi to start of cache line
				align 16
memoptA_prewrite8loop:        //write out odd QW's so 64bit write is cache line aligned
			cmp   ecx,eax           //start of cache line ?
				jz    memoptA_pre8done  //if not, write single QW
				movq    mm7,[esi+ecx]
				movntq  [edi+ecx],mm7
				add   ecx,8
				jmp   memoptA_prewrite8loop
				
				align 16
memoptA_write64loop:
			movntq  [edi+ecx-64],mm0
				movntq  [edi+ecx-56],mm1
				movntq  [edi+ecx-48],mm2
				movntq  [edi+ecx-40],mm3
				movntq  [edi+ecx-32],mm4
				movntq  [edi+ecx-24],mm5
				movntq  [edi+ecx-16],mm6
				movntq  [edi+ecx- 8],mm7
memoptA_pre8done:
			add   ecx,64
				cmp   ecx,edx
				ja    memoptA_done64    //less than 64 bytes left
				movq    mm0,[esi+ecx-64]
				movq    mm1,[esi+ecx-56]
				movq    mm2,[esi+ecx-48]
				movq    mm3,[esi+ecx-40]
				movq    mm4,[esi+ecx-32]
				movq    mm5,[esi+ecx-24]
				movq    mm6,[esi+ecx-16]
				movq    mm7,[esi+ecx- 8]
				jmp   memoptA_write64loop
				
memoptA_done64:
			sub   ecx,64
				
				align 16
memoptA_write8loop:           //less than 8 QW's left
			add   ecx,8
				cmp   ecx,edx
				ja    memoptA_done8     //no QW's left
				movq    mm7,[esi+ecx-8]
				movntq  [edi+ecx-8],mm7
				jmp   memoptA_write8loop
				
memoptA_done8:
			sub   esi,src_pitch
				sub   edi,dst_pitch
				dec   ebx               //row counter (height)
				jne   memoptA_rowloop
				
				sfence
				emms
		};
		return;
	}//end aligned version
}//end BitBlt_memopt()


void asm_BitBlt_MMX(int N, const unsigned char *srcp, unsigned char *dstp,
					const int src_pitch, const int dst_pitch,
					int row_size, int height, const int T, unsigned short *divin,
					unsigned char str)
{
	int bytesleft=0;
	if (row_size&15) {
		int a=(row_size+15)&(~15);
		if ((a<=src_pitch) && (a<=dst_pitch)) {
			row_size=a;
		} else {
			bytesleft=(row_size&15);
			row_size&=~15;
		}
	}
	int src_modulo = src_pitch - (row_size+bytesleft);
	int dst_modulo = dst_pitch - (row_size+bytesleft);
	if (height==0 || row_size==0) return;
    __asm {
		mov edi,[dstp]
			mov esi,[srcp]
			xor eax, eax;  // Height counter
		xor ebx, ebx;  // Row counter
		mov edx, [row_size]
new_line_mmx:
		mov ecx,[bytesleft]
			cmp ebx,edx
			jge nextline_mmx
			align 16
nextpixels_mmx:
		movq mm0,[esi+ebx]
			movq mm1,[esi+ebx+8]
			movq [edi+ebx],mm0
			movq [edi+ebx+8],mm1
			add ebx,16
			cmp ebx,edx
			jl nextpixels_mmx
			align 16
nextline_mmx:
		add esi,edx
			add edi,edx
			cmp ecx,0
			jz do_next_line_mmx
			rep movsb         ; the last 1-7 bytes
			
			align 16
do_next_line_mmx:
		add esi, [src_modulo]
			add edi, [dst_modulo]
			xor ebx, ebx;  // Row counter
		inc eax
			cmp eax,[height]
			jl new_line_mmx
	}
	__asm {emms};
}
