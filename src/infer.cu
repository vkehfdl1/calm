#include "model.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <cooperative_groups.h>

#include "helpers.cuh"

#define CUDA_CHECK(x)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
			        cudaGetErrorString(err), cudaGetErrorName(err), err);                                \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

#define PROF_TOKEN(bytes) ((0xCDAFull << 48) | (bytes))

template <typename T>
struct CoopLayer {
	float* rms_att_weight;
	T* wq;
	T* wk;
	T* wv;
	T* wo;
	float* bqkv;

	float* rms_ffn_weight;
	T* moegate;
	T* w1;
	T* w2;
	T* w3;
};

static cudaStream_t stream;

static int coopsms;

static __constant__ CoopLayer<void> cooplayers[MAX_LAYERS];

static void* cuda_devicecopy(void* host, size_t size) {
	void* device = NULL;
	CUDA_CHECK(cudaMalloc(&device, size));
	CUDA_CHECK(cudaMemcpyAsync(device, host, size, cudaMemcpyHostToDevice));
	return device;
}

static void* cuda_devicealloc(size_t size) {
	void* ptr = NULL;
	CUDA_CHECK(cudaMalloc(&ptr, size));
	return ptr;
}

static void* cuda_hostalloc(size_t size) {
	void* ptr = NULL;
	CUDA_CHECK(cudaHostAlloc(&ptr, size, 0));
	return ptr;
}

extern "C" void* upload_cuda(void* host, size_t size) {
	return cuda_devicecopy(host, size);
}

extern "C" void prepare_cuda(struct Transformer* transformer) {
	struct Config* config = &transformer->config;
	struct Weights* weights = &transformer->weights;
	struct RunState* state = &transformer->state;

	cudaDeviceProp devprop = {};
	CUDA_CHECK(cudaGetDeviceProperties(&devprop, 0));

	printf("# CUDA: %s, compute %d.%d, %d SMs, %.1f GiB, peak bandwidth %.0f GB/s (ECC %d)\n",
	       devprop.name, devprop.major, devprop.minor, devprop.multiProcessorCount,
	       (double)devprop.totalGlobalMem / (1024 * 1024 * 1024),
	       (double)devprop.memoryClockRate * (devprop.memoryBusWidth / 8) * 2 / 1e6, devprop.ECCEnabled);

	coopsms = devprop.multiProcessorCount;

	CUDA_CHECK(cudaStreamCreate(&stream));

	int dim = config->dim;
	int hidden_dim = config->hidden_dim;
	int q_dim = config->head_dim * config->n_heads;
	int kv_dim = config->head_dim * config->n_kv_heads;

	state->x = (float*)cuda_devicealloc(dim * sizeof(float));
	state->hb = (float*)cuda_devicealloc(hidden_dim * sizeof(float));
	state->he = (float*)cuda_devicealloc(config->n_experts_ac * hidden_dim * sizeof(float));
	state->q = (float*)cuda_devicealloc(q_dim * sizeof(float));
	state->att = (float*)cuda_devicealloc(config->n_heads * config->seq_len * sizeof(float));

	assert(state->kvbits == 8 || state->kvbits == 16);
	state->key_cache = cuda_devicealloc((size_t)config->n_layers * config->seq_len * kv_dim * (state->kvbits / 8));
	state->value_cache = cuda_devicealloc((size_t)config->n_layers * config->seq_len * kv_dim * (state->kvbits / 8));

	// logits are going to be read by the host so we just allocate them in host and write to host directly
	state->logits = (float*)cuda_hostalloc(config->vocab_size * sizeof(float));

	CoopLayer<void> layers[MAX_LAYERS];
	for (int l = 0; l < config->n_layers; ++l) {
		layers[l].rms_att_weight = weights->rms_att_weight[l];
		layers[l].wq = weights->wq[l];
		layers[l].wk = weights->wk[l];
		layers[l].wv = weights->wv[l];
		layers[l].wo = weights->wo[l];
		layers[l].bqkv = weights->bqkv[l];

		layers[l].rms_ffn_weight = weights->rms_ffn_weight[l];
		layers[l].moegate = weights->moegate[l];
		layers[l].w1 = weights->w1[l];
		layers[l].w2 = weights->w2[l];
		layers[l].w3 = weights->w3[l];
	}

	cudaMemcpyToSymbol(cooplayers, layers, sizeof(layers));
}

template <typename T>
__device__ inline float embed(T* weight, int idx) {
	return float(weight[idx]);
}

__device__ inline float embed(uint32_t* weight, int idx) {
	return gf4_ff(weight[idx / 8], idx % 8);
}

template <typename T>
__global__ static void kernel_embed(float* o, T* weight, int token, int n, float scale) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	assert(i < n);

	o[i] = embed(weight, token * n + i) * scale;
}

template <typename KVT>
__global__ static void kernel_rotate_sink(uint64_t, int kvd, KVT* key_cache, int head_dim, int kv_sink, float theta_log2, int seq_len, int rotary_dim) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	assert(i < kv_sink * kvd);

	int l = blockIdx.y;

	int j_head = i % head_dim;
	float freq = j_head >= rotary_dim ? 0.f : exp2f(-theta_log2 * (float)j_head / (float)rotary_dim);

	// rotate sink tokens forward to keep pace with non-sink tokens
	float fcr, fci;
	sincosf(freq, &fci, &fcr);

	size_t loff = (size_t)l * seq_len * kvd;
	KVT* kb = key_cache + loff;

	// note: k layout is transposed (we store all positions for a given head *pair* contiguously) to improve attn_score performance
	int t = i / kvd;
	int o = t * 2 + seq_len * (i % kvd);

	float k0 = float(kb[o + 0]);
	float k1 = float(kb[o + 1]);

	float rk0 = k0 * fcr - k1 * fci;
	float rk1 = k0 * fci + k1 * fcr;

	kb[o + 0] = KVT(rk0);
	kb[o + 1] = KVT(rk1);
}

__device__ inline float gelu(float x) {
	return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}

__device__ inline float silu(float x) {
	return x / (1.0f + expf(-x));
}

__device__ static void moe_gate_warp(float* moe_weights, int* moe_experts, float* weights, int experts, int active) {
	int i = threadIdx.x;

	// (unscaled) softmax across experts
	float w = (i < experts) ? weights[i] : -FLT_MAX;
	float max_val = warpreduce_max(w);
	w = expf(w - max_val);

	// weight in top 24 bits, index in bottom 8
	int wi = (__float_as_int(w) & 0xffffff00) | i;

	// top k within warp
	float sumw = 0.f;
	int acti = -1;

	for (int k = 0; k < active; ++k) {
		int maxi = warpreduce_maxi(wi);

		sumw += __int_as_float(maxi);

		// keeps top weight in thread k, clears weight for thread with max thread to avoid re-selection
		acti = (i == k) ? maxi : acti;
		wi = (wi == maxi) ? 0 : wi;
	}

	// write normalized weights
	if (i < active) {
		assert(acti >= 0);

		moe_experts[i] = acti & 0xff;
		moe_weights[i] = __int_as_float(acti) / sumw;
	}
}

union half4 {
	float2 g;
	half h[4];
};

__device__ inline float4 attn_load4(half* p) {
	half4 h = *(half4*)p;
	return {__half2float(h.h[0]), __half2float(h.h[1]), __half2float(h.h[2]), __half2float(h.h[3])};
}

__device__ inline float4 attn_load4(__nv_fp8_e5m2* p) {
	return fp8x4_e5m2_ff(*(__nv_fp8x4_e5m2*)p);
}

template <typename KVT>
__device__ inline float2 attn_score2(KVT* kht, float* qh, int head_dim, int seq_len, int t) {
	float score1 = 0.0f;
	float score2 = 0.0f;
	for (int j = 0; j < head_dim; j += 2) {
		float4 kk = attn_load4(&kht[j * seq_len + t * 2]);
		float2 qq = *(float2*)&qh[j];
		score1 += kk.x * qq.x;
		score1 += kk.y * qq.y;
		score2 += kk.z * qq.x;
		score2 += kk.w * qq.y;
	}

	score1 /= sqrtf(head_dim);
	score2 /= sqrtf(head_dim);

	return {score1, score2};
}

template <typename KVT>
__device__ inline float attn_warpdot(KVT* val, float* atth, int kv_len) {
	int kv_len4 = kv_len & ~3;
	int lane = threadIdx.x % warpSize;

	float res = 0.0f;
	float sum = 0.0f;
	for (int t = lane * 4; t < kv_len4; t += warpSize * 4) {
		float4 vv = attn_load4(&val[t]);
		float4 aa = *(float4*)&atth[t];
		res += vv.x * aa.x;
		res += vv.y * aa.y;
		res += vv.z * aa.z;
		res += vv.w * aa.w;
		sum += aa.x + aa.y + aa.z + aa.w;
	}

	if (kv_len4 + lane < kv_len) {
		float a = atth[kv_len4 + lane];
		res += a * float(val[kv_len4 + lane]);
		sum += a;
	}

	res = warpreduce_sum(res);
	sum = warpreduce_sum(sum);

	return res / sum;
}

__device__ static void softmax(float* x, int size) {
	int i = threadIdx.x;

	// find max value per thread (for numerical stability)
	float max_val = -FLT_MAX;
	for (int j = i; j < size; j += blockDim.x) {
		max_val = max(max_val, x[j]);
	}

	// max across threads in block
	max_val = blockreduce_max(max_val);

	// exp per thread
	for (int j = i; j < size; j += blockDim.x) {
		x[j] = expf(x[j] - max_val);
	}
}

__device__ static float rmsmean(float* x, int size) {
	int i = threadIdx.x;
	int blockSize = blockDim.x;

	float sum = 0.0f;
	for (int j = i; j < size; j += blockSize) {
		sum += x[j];
	}

	return blockreduce_sum(sum) / size;
}

__device__ static float rmsnorm(float* o, float* x, float* weight, int size, float eps, float mean) {
	int i = threadIdx.x;
	int blockSize = blockDim.x;

	// calculate sum of squares (per thread)
	float ss = 0.0f;
	for (int j = i; j < size; j += blockSize) {
		float v = x[j] - mean;
		ss += v * v;
		o[j] = v * weight[j];
	}

	// sum across threads in block
	ss = blockreduce_sum(ss);

	// caller is responsible for normalization
	return rsqrtf(ss / size + eps);
}

__device__ static void syncgrid() {
	using namespace cooperative_groups::details;
	grid::sync(&get_grid_workspace()->barrier);
}

template <typename T, typename KVT>
struct CoopArgs {
	uint64_t bw;

	float* x;
	float* hb;
	float* q;
	float* att;

	KVT* key_cache;
	KVT* val_cache;

	int n_layers;

	int dim;
	int hidden_dim;
	int head_dim;
	int n_heads;
	int n_kv_heads;
	int n_experts;
	int n_experts_ac;
	int seq_len;
	int rotary_dim;

	bool norm_mean;
	bool act_gelu;

	int kv_len;
	int kv_pos;
	int pos;

	float norm_eps;
	float theta_log2;
};

template <typename T, typename KVT>
__global__ __launch_bounds__(1024, 1) static void kernel_forward(const __grid_constant__ CoopArgs<T, KVT> args) {
	extern __shared__ float xs[];
	__shared__ float rmsscale;

	__shared__ float moe_weights[32];
	__shared__ int moe_experts[32];

	int dim = args.dim;
	int hidden_dim = args.hidden_dim;
	int head_dim = args.head_dim;

	int kv_mul = args.n_heads / args.n_kv_heads;
	int q_dim = args.head_dim * args.n_heads;
	int kv_dim = args.head_dim * args.n_kv_heads;

	const int IK = 2; // K consecutive warps per block, groups of K are interleaved across SMs for better work distribution
	int io = blockIdx.x * IK + (threadIdx.x / warpSize % IK) + gridDim.x * IK * (threadIdx.x / warpSize / IK);
	int ib = (gridDim.x * blockDim.x) / warpSize;

	// dummy moe weights for non-moe models; will be overwritten by moe gate
	moe_weights[0] = 1.f;
	moe_experts[0] = 0;

	for (int l = 0; l < args.n_layers; ++l) {
		const CoopLayer<T>* L = (const CoopLayer<T>*)&cooplayers[l];

		// pre-attention rmsnorm (into shared memory)
		rmsscale = rmsnorm(xs, args.x, L->rms_att_weight, dim, args.norm_eps, args.norm_mean ? rmsmean(args.x, dim) : 0.f);

		size_t loff = (size_t)l * args.seq_len * kv_dim; // kv cache layer offset for convenience
		KVT* keyb = args.key_cache + loff;
		KVT* valb = args.val_cache + loff;

		// qkv matmul + RoPE encoding + update KV cache
		for (int j = io * 2; j < q_dim + kv_dim * 2; j += ib * 2) {
			T* w = j < q_dim ? L->wq : (j < q_dim + kv_dim ? L->wk : L->wv);
			int k = j < q_dim ? j : (j < q_dim + kv_dim ? j - q_dim : j - q_dim - kv_dim);

			float v0 = matmul_warppar(xs, w, k + 0, dim) * rmsscale;
			float v1 = matmul_warppar(xs, w, k + 1, dim) * rmsscale;

			if (L->bqkv) {
				v0 += L->bqkv[j + 0];
				v1 += L->bqkv[j + 1];
			}

			if (threadIdx.x % warpSize == 0) {
				int j_head = j % head_dim;
				float freq = j_head >= args.rotary_dim ? 0.f : exp2f(-args.theta_log2 * (float)j_head / (float)args.rotary_dim);
				float fcr, fci;
				sincosf(args.pos * freq, &fci, &fcr);

				if (j < q_dim) {
					args.q[k + 0] = v0 * fcr - v1 * fci;
					args.q[k + 1] = v0 * fci + v1 * fcr;
				} else if (j < q_dim + kv_dim) {
					// note: k layout is transposed (we store all positions for a given head *pair* contiguously) to improve attn_score performance
					keyb[args.kv_pos * 2 + 0 + args.seq_len * k] = KVT(v0 * fcr - v1 * fci);
					keyb[args.kv_pos * 2 + 1 + args.seq_len * k] = KVT(v0 * fci + v1 * fcr);
				} else {
					// note: v layout is transposed (we store all positions for a given head contiguously) to improve attn_mix performance
					valb[args.kv_pos + args.seq_len * (k + 0)] = KVT(v0);
					valb[args.kv_pos + args.seq_len * (k + 1)] = KVT(v1);
				}
			}
		}

		syncgrid();

		// attention score
		int kv_len32 = (args.kv_len + 31) / 32;

		for (int j = io; j < kv_len32 * args.n_heads; j += ib) {
			int h = j % args.n_heads;
			int kvh = h / kv_mul;
			int t = ((j / args.n_heads) * warpSize + (threadIdx.x % warpSize)) * 2;

			if (t < args.kv_len) {
				float* qh = args.q + h * head_dim;
				KVT* kh = keyb + kvh * head_dim * args.seq_len;
				float* atth = args.att + h * args.seq_len;

				float2 score = attn_score2(kh, qh, head_dim, args.seq_len, t);

				atth[t + 0] = score.x;
				atth[t + 1] = score.y;
			}
		}

		syncgrid();

		// attention softmax
		if (blockIdx.x < args.n_heads) {
			int h = blockIdx.x;
			float* atth = args.att + h * args.seq_len;

			softmax(atth, args.kv_len);
		}

		syncgrid();

		// attention mix
		for (int j = io; j < q_dim; j += ib) {
			int h = j / head_dim;
			int kvh = h / kv_mul;
			int j_head = j % head_dim;

			float* atth = args.att + h * args.seq_len;
			KVT* vh = valb + kvh * head_dim * args.seq_len;
			KVT* val = vh + j_head * args.seq_len;

			float res = attn_warpdot(val, atth, args.kv_len);

			if (threadIdx.x % warpSize == 0) {
				args.q[j] = res;
			}
		}

		syncgrid();

		// attention output
		for (int j = io; j < dim; j += ib) {
			float val = matmul_warppar(args.q, L->wo, j, q_dim);

			if (threadIdx.x % warpSize == 0) {
				args.x[j] += val;
			}
		}

		syncgrid();

		// post-attention rmsnorm (into shared memory)
		rmsscale = rmsnorm(xs, args.x, L->rms_ffn_weight, dim, args.norm_eps, args.norm_mean ? rmsmean(args.x, dim) : 0.f);

		// moegate
		if (args.n_experts) {
			__shared__ float exp[32];
			int j = threadIdx.x / warpSize;

			if (j < args.n_experts) {
				float val = matmul_warppar(xs, L->moegate, j, dim) * rmsscale;

				exp[j] = val;
			}

			__syncthreads();

			if (threadIdx.x < warpSize) {
				moe_gate_warp(moe_weights, moe_experts, exp, args.n_experts, args.n_experts_ac);
			}

			__syncthreads();
		}

		// F.silu(self.w1(x)) * self.w3(x)
		for (int j = io; j < hidden_dim * args.n_experts_ac; j += ib) {
			int je = (j % hidden_dim) + moe_experts[j / hidden_dim] * hidden_dim;
			float v1 = matmul_warppar(xs, L->w1, je, dim) * rmsscale;
			float v3 = matmul_warppar(xs, L->w3, je, dim) * rmsscale;

			float val = (args.act_gelu ? gelu(v1) : silu(v1)) * v3;

			if (threadIdx.x % warpSize == 0) {
				args.hb[j] = val;
			}
		}

		syncgrid();

		// self.w2(...) + pre-rmsnorm residual
		for (int j = io; j < dim * args.n_experts_ac; j += ib) {
			int je = (j % dim) + moe_experts[j / dim] * dim;
			float val = matmul_warppar(args.hb + (j / dim) * hidden_dim, L->w2, je, hidden_dim);

			if (threadIdx.x % warpSize == 0) {
				atomicAdd(&args.x[j % dim], val * moe_weights[j / dim]);
			}
		}

		syncgrid();
	}
}

template <typename T>
__global__ static void kernel_output(uint64_t, float* xout, float* x, T* w, float* rms_weight, int n, int d, float norm_eps, bool norm_mean) {
	extern __shared__ float xs[];

	float rmsscale = rmsnorm(xs, x, rms_weight, n, norm_eps, norm_mean ? rmsmean(x, n) : 0.f);

	int io = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	int ib = (gridDim.x * blockDim.x) / warpSize;

	for (int j = io; j < d; j += ib) {
		float val = matmul_warppar(xs, w, j, n) * rmsscale;

		// instead of writing one value per block, we transpose the values and write all results from first warp
		val = blocktranspose(val, 0.f);

		if (threadIdx.x < blockDim.x / warpSize) {
			xout[j + threadIdx.x] = val;
		}
	}
}

template <typename T, typename KVT>
static float* forward(struct Transformer* transformer, int token, int pos, unsigned flags) {
	struct Config* p = &transformer->config;
	struct Weights* w = &transformer->weights;
	struct RunState* s = &transformer->state;

	// a few convenience variables
	float* x = s->x;
	int dim = p->dim;
	int hidden_dim = p->hidden_dim;
	int kv_dim = p->head_dim * p->n_kv_heads;
	size_t dbits = w->dbits; // size_t prevents integer overflow in multiplications below

	// following "attention sinks" from StreamingLLM we keep the first few tokens in the KV cache as is
	int kv_sink = pos >= p->seq_len ? KV_SINKS : 0;
	int kv_pos = kv_sink + (pos - kv_sink) % (p->seq_len - kv_sink);
	int kv_len = pos >= p->seq_len ? p->seq_len : pos + 1;

	// ensure all dimensions are warp-aligned
	assert(dim % 32 == 0 && kv_dim % 32 == 0 && hidden_dim % 32 == 0);

	// copy the token embedding into x
	assert(token < p->vocab_size);
	kernel_embed<<<dim / 32, 32, 0, stream>>>(x, (T*)w->token_embedding_table, token, dim, p->embed_scale);

	// rotate sink tokens forward to keep pace with non-sink tokens
	if (kv_sink > 0) {
		kernel_rotate_sink<<<dim3(kv_sink * kv_dim / 64, p->n_layers), 32, 0, stream>>>(
			PROF_TOKEN(kv_sink * kv_dim * sizeof(KVT)), kv_dim, (KVT*)s->key_cache, p->head_dim, kv_sink, log2(p->rope_theta), p->seq_len, p->rotary_dim);
	}

	// forward all the layers
	size_t kvbw = p->n_kv_heads * p->head_dim * kv_len * sizeof(KVT) + p->n_heads * kv_len * sizeof(float);

	uint64_t bw = 0;
	bw += (dim + kv_dim * 2) * dim * dbits / 8; // QKV
	bw += kvbw * 2; // attn scoring and mixing
	bw += dim * dim * dbits / 8; // attn output
	bw += 3 * (hidden_dim * dim * dbits / 8) * max(p->n_experts_ac, 1); // MLP
	bw *= p->n_layers;

	CoopArgs<T, KVT> args = {
		PROF_TOKEN(bw),

		x,
		p->n_experts ? s->he : s->hb,
		s->q,
		s->att,

		(KVT*)s->key_cache,
		(KVT*)s->value_cache,

		p->n_layers,

		dim,
		hidden_dim,
		p->head_dim,
		p->n_heads,
		p->n_kv_heads,
		p->n_experts,
		max(p->n_experts_ac, 1),
		p->seq_len,
		p->rotary_dim,

		p->norm_mean,
		p->act_gelu,

		kv_len,
		kv_pos,
		pos,

		p->norm_eps,
		log2(p->rope_theta),
	};
	void* argsp = &args;

	CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel_forward<T, KVT>, coopsms, 1024, &argsp, dim * sizeof(float), stream));

	if (flags & FF_UPDATE_KV_ONLY) {
		// only update kv cache and don't output logits
		return NULL;
	}

	// classifier into logits
	kernel_output<<<coopsms, 32 * 32, dim * sizeof(float), stream>>>(
	    PROF_TOKEN(p->vocab_size * dim * dbits / 8), s->logits, x, (T*)w->wcls, w->rms_final_weight, dim, p->vocab_size, p->norm_eps, p->norm_mean);

	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors; they might fail with OOM due to lazy kernel compilation

	return s->logits;
}

extern "C" float* forward_cuda(struct Transformer* transformer, int token, int pos, unsigned flags) {
#define CASE(dbits_, dtype, kvbits_, kvtype)                                          \
	if (transformer->weights.dbits == dbits_ && transformer->state.kvbits == kvbits_) \
	return forward<dtype, kvtype>(transformer, token, pos, flags)

	CASE(4, uint32_t, 8, __nv_fp8_e5m2);
	CASE(4, uint32_t, 16, __half);
	CASE(8, __nv_fp8_e5m2, 8, __nv_fp8_e5m2);
	CASE(8, __nv_fp8_e5m2, 16, __half);
	CASE(16, __half, 8, __nv_fp8_e5m2);
	CASE(16, __half, 16, __half);

	assert(!"Unsupported dbits/kvbits combination for CUDA: dbits must be 4, 8 or 16, kvbits must be 8 or 16");
	return NULL;

#undef CASE
}
