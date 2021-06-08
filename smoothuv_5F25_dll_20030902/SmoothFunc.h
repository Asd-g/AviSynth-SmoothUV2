void smoothN_MMX(int N, const unsigned char *src, unsigned char *dst, const int src_pitch,
				 const int dst_pitch, int w, int h, const int t,
				 unsigned short *div, unsigned char str);
void smoothHQN_MMX(int N, const unsigned char *src, unsigned char *dst, const int src_pitch,
				   const int dst_pitch, int w, int h, const int t,
				   unsigned short *div, unsigned char str);
void smoothHQN_field_MMX(int N, const unsigned char *origsrc, unsigned char *origdst,
				   const int src_pitch, const int dst_pitch,
				   int w, int h, const int T, unsigned short *divin,
				   unsigned char str);
void smoothN_field_MMX(int N, const unsigned char *origsrc, unsigned char *origdst,
				   const int src_pitch, const int dst_pitch,
				   int w, int h, const int T, unsigned short *divin,
				   unsigned char str);



/* Copy Functions */
void asm_BitBlt_ISSE(int N, const unsigned char *srcp, unsigned char *dstp,
					 const int src_pitch, const int dst_pitch,
					 int row_size, int height, const int T, unsigned short *divin,
					 unsigned char str);
void asm_BitBlt_MMX(int N, const unsigned char *srcp, unsigned char *dstp,
					const int src_pitch, const int dst_pitch,
					int row_size, int height, const int T, unsigned short *divin,
					unsigned char str);
