#include "cudashared.h"
#include "stdio.h"


/**
- * Extracts a bit field from source and places the zero or sign-extended result
- * in extract
- */

/*
template <unsigned int BIT_START, unsigned int NUM_BITS>
struct ExtractKeyBits
{

    __device__ __forceinline__ static unsigned int Extract(const unsigned int &source)
    {
        unsigned int bits;
        //               asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "n"(BIT_START), "n"(NUM_BITS));
        unsigned int _BIT_START = BIT_START;
        unsigned int _NUM_BITS = NUM_BITS;
        asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "r"(_BIT_START), "r"(_NUM_BITS));
        return bits;
    }
};

template <unsigned int BIT_START, unsigned int NUM_BITS>
struct InsertKeyBits
{

    __device__ __forceinline__ static unsigned int Insert(const unsigned int &orig, const unsigned int &source)
    {
        unsigned int bits;
        //               asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "n"(BIT_START), "n"(NUM_BITS));
        unsigned int _BIT_START = BIT_START;
        unsigned int _NUM_BITS = NUM_BITS;
        asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(bits) : "r"(source), "r"(orig), "r"(_BIT_START), "r"(_NUM_BITS));
        return bits;
    }
};

struct Rot
{

    __device__ __forceinline__ static unsigned int rot(const unsigned int &a, const unsigned int bit)
    {
        unsigned int out;
        //               asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "n"(BIT_START), "n"(NUM_BITS));
        asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(a), "r"(bit));
        return out;
    }
};

//template <unsigned int SHIFT>

struct Shift
{

    __device__ __forceinline__ unsigned int operator()(const unsigned int &source, const unsigned int& _BIT_START, const unsigned int& _NUM_BITS)
    {
        unsigned int bits;
        //               asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "n"(BIT_START), "n"(NUM_BITS));
        //                 const unsigned int _BIT_START = SHIFT;
        //                 const unsigned int _NUM_BITS = 32-SHIFT;
        asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "r"(_BIT_START), "r"(_NUM_BITS));
        return bits;
    }
};
*/

__constant__ cuda_in cin;


#define byteswap(x) (((x>>24) & 0x000000ff) | ((x>>8) & 0x0000ff00) | ((x<<8) & 0x00ff0000) | ((x<<24) & 0xff000000))
#define rotateright(x,bits) (((x ) >> bits) | (x << (32 - bits)))
#define R(x) (work[x] = (rotateright(work[x-2],17)^rotateright(work[x-2],19)^((work[x-2])>>10)) + work[x -  7] + (rotateright(work[x-15],7)^rotateright(work[x-15],18)^((work[x-15])>>3)) + work[x - 16])
#define sharound(a,b,c,d,e,f,g,h,x,K) {t1=h+(rotateright(e,6)^rotateright(e,11)^rotateright(e,25))+(g^(e&(f^g)))+K+x; t2=(rotateright(a,2)^rotateright(a,13)^rotateright(a,22))+((a&b)|(c&(a|b))); d+=t1; h=t1+t2;}


extern "C" __global__ void cuda_process(cuda_in* in, unsigned int *out, const unsigned int loops, const unsigned int bits)
{
    /*exit as fast as posible if one block has finished with solution*/
    if (*out != 0) return;

    unsigned int work[64];
    unsigned int A, B, C, D, E, F, G, H;
    const unsigned int myid = (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int nonce = cin.m_nonce + (myid << bits);
    unsigned int t1, t2;
    //unsigned int bestnonce=0;
    //unsigned int bestg=~0;

    const unsigned int cin_m_merkle = cin.m_merkle;
    const unsigned int cin_m_ntime = cin.m_ntime;
    const unsigned int cin_m_nbits = cin.m_nbits;

    // the first 3 rounds we can do outside the loop because they depend on work[0] through work[2] which won't change
    /* move old A1, ... H1 to shared to solve registers
     * can also calculated on host and give to kernel, because its se same for all threads and blocks
     */
    __shared__ unsigned int AH[8];
    __shared__ unsigned int AH2[8]; //cache for second round
    if (threadIdx.x < 8)
    {
        AH2[threadIdx.x] = AH[threadIdx.x] = cin.m_AH[threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        sharound(AH[0], AH[1], AH[2], AH[3], AH[4], AH[5], AH[6], AH[7], cin_m_merkle, 0x428A2F98);
        sharound(AH[7], AH[0], AH[1], AH[2], AH[3], AH[4], AH[5], AH[6], cin_m_ntime, 0x71374491);
        sharound(AH[6], AH[7], AH[0], AH[1], AH[2], AH[3], AH[4], AH[5], cin_m_nbits, 0xB5C0FBCF);
    }
    __syncthreads();

#pragma unroll 1
    for (unsigned int it = 0; it < loops; ++it)
    {
        if (*out != 0) return;

        A = AH[0];
        B = AH[1];
        C = AH[2];
        D = AH[3];
        E = AH[4];
        F = AH[5];
        G = AH[6];
        H = AH[7];
        work[0] = cin_m_merkle;
        work[1] = cin_m_ntime;
        work[2] = cin_m_nbits;
        //work[3]=byteswap(nonce+it);
        work[3] = nonce + it;
        work[4] = 0x80000000;
        work[5] = 0x00000000;
        work[6] = 0x00000000;
        work[7] = 0x00000000;
        work[8] = 0x00000000;
        work[9] = 0x00000000;
        work[10] = 0x00000000;
        work[11] = 0x00000000;
        work[12] = 0x00000000;
        work[13] = 0x00000000;
        work[14] = 0x00000000;
        work[15] = 0x00000280;

        //sharound(A,B,C,D,E,F,G,H,work[0],0x428A2F98);
        //sharound(H,A,B,C,D,E,F,G,work[1],0x71374491);
        //sharound(G,H,A,B,C,D,E,F,work[2],0xB5C0FBCF);
        sharound(F, G, H, A, B, C, D, E, work[3], 0xE9B5DBA5);
        sharound(E, F, G, H, A, B, C, D, work[4], 0x3956C25B);
        sharound(D, E, F, G, H, A, B, C, work[5], 0x59F111F1);
        sharound(C, D, E, F, G, H, A, B, work[6], 0x923F82A4);
        sharound(B, C, D, E, F, G, H, A, work[7], 0xAB1C5ED5);
        sharound(A, B, C, D, E, F, G, H, work[8], 0xD807AA98);
        sharound(H, A, B, C, D, E, F, G, work[9], 0x12835B01);
        sharound(G, H, A, B, C, D, E, F, work[10], 0x243185BE);
        sharound(F, G, H, A, B, C, D, E, work[11], 0x550C7DC3);
        sharound(E, F, G, H, A, B, C, D, work[12], 0x72BE5D74);
        sharound(D, E, F, G, H, A, B, C, work[13], 0x80DEB1FE);
        sharound(C, D, E, F, G, H, A, B, work[14], 0x9BDC06A7);
        sharound(B, C, D, E, F, G, H, A, work[15], 0xC19BF174);
        sharound(A, B, C, D, E, F, G, H, R(16), 0xE49B69C1);
        sharound(H, A, B, C, D, E, F, G, R(17), 0xEFBE4786);
        sharound(G, H, A, B, C, D, E, F, R(18), 0x0FC19DC6);
        sharound(F, G, H, A, B, C, D, E, R(19), 0x240CA1CC);
        sharound(E, F, G, H, A, B, C, D, R(20), 0x2DE92C6F);
        sharound(D, E, F, G, H, A, B, C, R(21), 0x4A7484AA);
        sharound(C, D, E, F, G, H, A, B, R(22), 0x5CB0A9DC);
        sharound(B, C, D, E, F, G, H, A, R(23), 0x76F988DA);
        sharound(A, B, C, D, E, F, G, H, R(24), 0x983E5152);
        sharound(H, A, B, C, D, E, F, G, R(25), 0xA831C66D);
        sharound(G, H, A, B, C, D, E, F, R(26), 0xB00327C8);
        sharound(F, G, H, A, B, C, D, E, R(27), 0xBF597FC7);
        sharound(E, F, G, H, A, B, C, D, R(28), 0xC6E00BF3);
        sharound(D, E, F, G, H, A, B, C, R(29), 0xD5A79147);
        sharound(C, D, E, F, G, H, A, B, R(30), 0x06CA6351);
        sharound(B, C, D, E, F, G, H, A, R(31), 0x14292967);
        sharound(A, B, C, D, E, F, G, H, R(32), 0x27B70A85);
        sharound(H, A, B, C, D, E, F, G, R(33), 0x2E1B2138);
        sharound(G, H, A, B, C, D, E, F, R(34), 0x4D2C6DFC);
        sharound(F, G, H, A, B, C, D, E, R(35), 0x53380D13);
        sharound(E, F, G, H, A, B, C, D, R(36), 0x650A7354);
        sharound(D, E, F, G, H, A, B, C, R(37), 0x766A0ABB);
        sharound(C, D, E, F, G, H, A, B, R(38), 0x81C2C92E);
        sharound(B, C, D, E, F, G, H, A, R(39), 0x92722C85);
        sharound(A, B, C, D, E, F, G, H, R(40), 0xA2BFE8A1);
        sharound(H, A, B, C, D, E, F, G, R(41), 0xA81A664B);
        sharound(G, H, A, B, C, D, E, F, R(42), 0xC24B8B70);
        sharound(F, G, H, A, B, C, D, E, R(43), 0xC76C51A3);
        sharound(E, F, G, H, A, B, C, D, R(44), 0xD192E819);
        sharound(D, E, F, G, H, A, B, C, R(45), 0xD6990624);
        sharound(C, D, E, F, G, H, A, B, R(46), 0xF40E3585);
        sharound(B, C, D, E, F, G, H, A, R(47), 0x106AA070);
        sharound(A, B, C, D, E, F, G, H, R(48), 0x19A4C116);
        sharound(H, A, B, C, D, E, F, G, R(49), 0x1E376C08);
        sharound(G, H, A, B, C, D, E, F, R(50), 0x2748774C);
        sharound(F, G, H, A, B, C, D, E, R(51), 0x34B0BCB5);
        sharound(E, F, G, H, A, B, C, D, R(52), 0x391C0CB3);
        sharound(D, E, F, G, H, A, B, C, R(53), 0x4ED8AA4A);
        sharound(C, D, E, F, G, H, A, B, R(54), 0x5B9CCA4F);
        sharound(B, C, D, E, F, G, H, A, R(55), 0x682E6FF3);
        sharound(A, B, C, D, E, F, G, H, R(56), 0x748F82EE);
        sharound(H, A, B, C, D, E, F, G, R(57), 0x78A5636F);
        sharound(G, H, A, B, C, D, E, F, R(58), 0x84C87814);
        sharound(F, G, H, A, B, C, D, E, R(59), 0x8CC70208);
        sharound(E, F, G, H, A, B, C, D, R(60), 0x90BEFFFA);
        sharound(D, E, F, G, H, A, B, C, R(61), 0xA4506CEB);
        sharound(C, D, E, F, G, H, A, B, R(62), 0xBEF9A3F7);
        sharound(B, C, D, E, F, G, H, A, R(63), 0xC67178F2);

        // hash the hash now

        work[0] = AH2[0] + A;
        work[1] = AH2[1] + B;
        work[2] = AH2[2] + C;
        work[3] = AH2[3] + D;
        work[4] = AH2[4] + E;
        work[5] = AH2[5] + F;
        work[6] = AH2[6] + G;
        work[7] = AH2[7] + H;
        work[8] = 0x80000000;
        work[9] = 0x00000000;
        work[10] = 0x00000000;
        work[11] = 0x00000000;
        work[12] = 0x00000000;
        work[13] = 0x00000000;
        work[14] = 0x00000000;
        work[15] = 0x00000100;

        A = 0x6a09e667;
        B = 0xbb67ae85;
        C = 0x3c6ef372;
        D = 0xa54ff53a;
        E = 0x510e527f;
        F = 0x9b05688c;
        G = 0x1f83d9ab;
        H = 0x5be0cd19;

        sharound(A, B, C, D, E, F, G, H, work[0], 0x428A2F98);
        sharound(H, A, B, C, D, E, F, G, work[1], 0x71374491);
        sharound(G, H, A, B, C, D, E, F, work[2], 0xB5C0FBCF);
        sharound(F, G, H, A, B, C, D, E, work[3], 0xE9B5DBA5);
        sharound(E, F, G, H, A, B, C, D, work[4], 0x3956C25B);
        sharound(D, E, F, G, H, A, B, C, work[5], 0x59F111F1);
        sharound(C, D, E, F, G, H, A, B, work[6], 0x923F82A4);
        sharound(B, C, D, E, F, G, H, A, work[7], 0xAB1C5ED5);
        sharound(A, B, C, D, E, F, G, H, work[8], 0xD807AA98);
        sharound(H, A, B, C, D, E, F, G, work[9], 0x12835B01);
        sharound(G, H, A, B, C, D, E, F, work[10], 0x243185BE);
        sharound(F, G, H, A, B, C, D, E, work[11], 0x550C7DC3);
        sharound(E, F, G, H, A, B, C, D, work[12], 0x72BE5D74);
        sharound(D, E, F, G, H, A, B, C, work[13], 0x80DEB1FE);
        sharound(C, D, E, F, G, H, A, B, work[14], 0x9BDC06A7);
        sharound(B, C, D, E, F, G, H, A, work[15], 0xC19BF174);
        sharound(A, B, C, D, E, F, G, H, R(16), 0xE49B69C1);
        sharound(H, A, B, C, D, E, F, G, R(17), 0xEFBE4786);
        sharound(G, H, A, B, C, D, E, F, R(18), 0x0FC19DC6);
        sharound(F, G, H, A, B, C, D, E, R(19), 0x240CA1CC);
        sharound(E, F, G, H, A, B, C, D, R(20), 0x2DE92C6F);
        sharound(D, E, F, G, H, A, B, C, R(21), 0x4A7484AA);
        sharound(C, D, E, F, G, H, A, B, R(22), 0x5CB0A9DC);
        sharound(B, C, D, E, F, G, H, A, R(23), 0x76F988DA);
        sharound(A, B, C, D, E, F, G, H, R(24), 0x983E5152);
        sharound(H, A, B, C, D, E, F, G, R(25), 0xA831C66D);
        sharound(G, H, A, B, C, D, E, F, R(26), 0xB00327C8);
        sharound(F, G, H, A, B, C, D, E, R(27), 0xBF597FC7);
        sharound(E, F, G, H, A, B, C, D, R(28), 0xC6E00BF3);
        sharound(D, E, F, G, H, A, B, C, R(29), 0xD5A79147);
        sharound(C, D, E, F, G, H, A, B, R(30), 0x06CA6351);
        sharound(B, C, D, E, F, G, H, A, R(31), 0x14292967);
        sharound(A, B, C, D, E, F, G, H, R(32), 0x27B70A85);
        sharound(H, A, B, C, D, E, F, G, R(33), 0x2E1B2138);
        sharound(G, H, A, B, C, D, E, F, R(34), 0x4D2C6DFC);
        sharound(F, G, H, A, B, C, D, E, R(35), 0x53380D13);
        sharound(E, F, G, H, A, B, C, D, R(36), 0x650A7354);
        sharound(D, E, F, G, H, A, B, C, R(37), 0x766A0ABB);
        sharound(C, D, E, F, G, H, A, B, R(38), 0x81C2C92E);
        sharound(B, C, D, E, F, G, H, A, R(39), 0x92722C85);
        sharound(A, B, C, D, E, F, G, H, R(40), 0xA2BFE8A1);
        sharound(H, A, B, C, D, E, F, G, R(41), 0xA81A664B);
        sharound(G, H, A, B, C, D, E, F, R(42), 0xC24B8B70);
        sharound(F, G, H, A, B, C, D, E, R(43), 0xC76C51A3);
        sharound(E, F, G, H, A, B, C, D, R(44), 0xD192E819);
        sharound(D, E, F, G, H, A, B, C, R(45), 0xD6990624);
        sharound(C, D, E, F, G, H, A, B, R(46), 0xF40E3585);
        sharound(B, C, D, E, F, G, H, A, R(47), 0x106AA070);
        sharound(A, B, C, D, E, F, G, H, R(48), 0x19A4C116);
        sharound(H, A, B, C, D, E, F, G, R(49), 0x1E376C08);
        sharound(G, H, A, B, C, D, E, F, R(50), 0x2748774C);
        sharound(F, G, H, A, B, C, D, E, R(51), 0x34B0BCB5);
        sharound(E, F, G, H, A, B, C, D, R(52), 0x391C0CB3);
        sharound(D, E, F, G, H, A, B, C, R(53), 0x4ED8AA4A);
        sharound(C, D, E, F, G, H, A, B, R(54), 0x5B9CCA4F);
        sharound(B, C, D, E, F, G, H, A, R(55), 0x682E6FF3);
        sharound(A, B, C, D, E, F, G, H, R(56), 0x748F82EE);
        sharound(H, A, B, C, D, E, F, G, R(57), 0x78A5636F);
        sharound(G, H, A, B, C, D, E, F, R(58), 0x84C87814);
        sharound(F, G, H, A, B, C, D, E, R(59), 0x8CC70208);
        sharound(E, F, G, H, A, B, C, D, R(60), 0x90BEFFFA);
        sharound(D, E, F, G, H, A, B, C, R(61), 0xA4506CEB);
        //we don't need to do these last 2 rounds as they update F, B, E and A, but we only care about G and H
        //sharound(C,D,E,F,G,H,A,B,R(62),0xBEF9A3F7);
        //sharound(B,C,D,E,F,G,H,A,R(63),0xC67178F2);

        H += 0x5be0cd19;

        if ((H == 0))// && (G<=bestg))
        {
            *out = nonce + it; /*we only need one solution*/
        }

    }

}

//__global__ void setValue(uint32* ptr)
//{
//    *ptr = 0u;
//}
////#include <iostream>

//void bitcoinSearch(cudaStream_t stream, int grid, int threads, uint32 *in, uint32 *out, const uint32 nonce, const int unsigned loops, const unsigned int bits)
//{
//    //    std::cout << "stream=" << (int*) stream << " grid" << grid << " th=" << threads << " none=" << nonce << " bits=" << bits << std::endl;
//    cuda_process << <grid, threads, 0, stream >> >(in, out, nonce, loops, bits);
//}

//void setToZero(cudaStream_t stream, uint32* ptr)
//{
//    setValue << <1, 1, 0, stream >> >(ptr);
//}

void cuda_process_helper(cuda_in *in, cuda_out *out, const int unsigned loops, const unsigned int bits, const int grid, const int threads)
{
    //printf(".");fflush( stdout );
    //cudaFuncSetCacheConfig( cuda_process, cudaFuncCachePreferShared );

	//cuda_process<<<grid,threads>>>(in,out,loops,bits-1);
}
