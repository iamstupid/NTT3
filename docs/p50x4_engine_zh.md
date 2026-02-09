# p50x4 引擎：四素数 50-bit FMA NTT 乘法器

本文档是对 `ntt/p50x4/` 引擎（原 `sd_ntt`）的完整技术报告，涵盖算法原理、数据流、SIMD 实现细节与性能特性。

---

## 0. 定位与动机

`p50x4` 是面向 **u64 (base 2^64) 大整数乘法** 的 NTT 引擎，与面向 u32 的 `p30x3` 引擎平级共存。

|                 | p30x3                          | p50x4                          |
|-----------------|--------------------------------|--------------------------------|
| limb 类型        | u32 (base 2^32)               | u64 (base 2^64)               |
| 素数             | 3 个 ~30-bit                   | 4 个 ~50-bit                   |
| 模运算           | 整数 Montgomery (`__m256i`)    | 浮点 FMA Barrett (`__m256d`)   |
| 系数打包         | 1 limb → 1 系数               | 5 limbs → 4 × 80-bit 系数     |
| CRT 重构         | 3-prime 整数 CRT              | Garner (FP SIMD) + Horner (INT) |
| 命名空间         | `ntt::` (模板参数 `B, Mod`)    | `ntt::p50x4`                   |

**为什么需要 80-bit 打包？** u64 limb 乘积最大 128 bits，卷积后一个系数上限约 128 + log₂(N) bits。80-bit 打包在损失可忽略的精度前提下，将 limb→系数转换率从 1:1 提升到 5:4，减少约 20% 的 NTT 变换长度。

**为什么是 4 个素数？** 4 个 ~50-bit 素数之积 ≈ 2^196，而 80-bit 系数对的卷积系数上限约 160 + log₂(N) bits（N ≤ 2^39 时 < 200 bits），留有足够裕量。

---

## 1. 素数选择

四个素数均满足形式 `105 × 2^39 × k + 1`（其中 `c = 105 = 3 × 5 × 7`）：

| k  | p                    | bits | p-1 因子分解                       |
|----|----------------------|------|------------------------------------|
| 9  | 519519244124161      | 49   | 2^39 × 3^3 × 5 × 7               |
| 13 | 750416685957121      | 50   | 2^39 × 3 × 5 × 7 × 13            |
| 15 | 865865406873601      | 50   | 2^39 × 3^2 × 5^2 × 7             |
| 19 | 1096762848706561     | 50   | 2^39 × 3 × 5 × 7 × 19            |

**关键性质：**
- 均支持 NTT 长度 2^39、3 × 2^39、5 × 2^39（`p-1` 中含 2^39 × 3 × 5 × 7）
- 在 double 精度 52-bit mantissa 内可精确表示（50 bits < 52 bits）
- 乘积 ≈ 2^196，满足 80-bit 系数卷积上限

---

## 2. 端到端计算流程

`Ntt4::multiply()` 的完整流水线：

```
输入: a[0..na), b[0..nb)  (u64 limbs, little-endian)
  │
  ▼ ① 系数数量计算
  nca = ceil(na × 64 / 80)
  ncb = ceil(nb × 64 / 80)
  conv_len = nca + ncb - 1
  │
  ▼ ② NTT 长度选择
  N = ceil_ntt_size(conv_len)     // min{2^k, 3·2^k, 5·2^k} ≥ conv_len
  │
  ▼ ③ 80-bit 系数提取 + 模约化 (对 a: 4 个素数并行; 对 b: 逐素数)
  extract_4x80_simd: 5 u64 → 4 × (lo48, hi32)
  reduce_4x1p:       lo48 + hi32 × 2^48 mod p  → double ∈ (-p/2, p/2)
  │
  ▼ ④ 对每个素数 p_i (i = 0..3):
  │     fft_mixed(fa[i], N)       // 前向混合基 FFT
  │     fft_mixed(fb, N)          // b 的前向变换（fb 被复用）
  │     point_mul(fa[i], fb, N)   // 频域逐点乘
  │     ifft_mixed(fa[i], N)      // 逆混合基 FFT
  │     scale_mixed(fa[i], N)     // 乘以 1/N
  │
  ▼ ⑤ CRT 重构
  garner_phase1:  4 个 double 残差 → 4 个混合基位 (FP SIMD, V4)
  horner_phase2:  混合基位 → 4-limb 整数 (INT, _umul128)
  accum_shifted:  按 16-bit 间距移位累加到输出缓冲区
  flush_to_output: 进位传播 → 最终 u64 limb 结果
  │
  ▼
输出: out[0..out_len)  (u64 limbs)
```

---

## 3. 80-bit 系数提取

### 3.1 打包方案

5 个连续 u64 limb（320 bits）被切分为 4 个 80-bit 系数：

```
limb[0]  limb[1]  limb[2]  limb[3]  limb[4]
|--64--|--64--|--64--|--64--|--64--|
|---80---|---80---|---80---|---80---|
 coeff 0  coeff 1  coeff 2  coeff 3
```

每个 80-bit 系数被拆为低 48 位（`lo48`）和高 32 位（`hi32`）：
- `value = lo48 + hi32 × 2^48`

### 3.2 SIMD 提取：`extract_4x80_simd`

一次性从 5 个 u64 中提取 4 组 `(lo48, hi32)`：

1. 将 5 个 u64 加载到 `__m256i` 寄存器
2. 通过移位和 OR 组合相邻 limb 边界的 bits
3. 用掩码 `0x0000FFFFFFFFFFFF` 提取 lo48
4. 用右移 48 提取 hi32
5. 通过 magic 常量（2^52 偏置法）转换为 `V4 (__m256d)` — 避免慢速的 `cvtsi2sd`

### 3.3 模约化：`reduce_4x1p`

将 `lo48 + hi32 × 2^48` 约化到 `(-p/2, p/2)`:

```
t48 = 2^48 mod p                   // 预计算常量
h = hi32 × t48                     // 第一步：高位折叠
q = round(h × pinv)
l = FMA(hi32, t48, -h)             // FMA 获得精确余数
t = h - q × p + l                  // Barrett 约化
r = t + lo48                       // 合并低位
q2 = round(r × pinv)
result = r - q2 × p                // 最终约化
```

---

## 4. FMA Barrett 模乘

核心的浮点模乘算法：

```c
s_mulmod(a, b, n, ninv):
  h = a × b                        // 高位（52-bit 精度）
  q = round(h × ninv)              // 商的估计
  l = FMA(a, b, -h)                // 精确低位：a*b - h（无舍入误差）
  result = FMA(-q, n, h) + l       // h - q*n + l
```

**关键性质：**
- FMA 保证 `a*b = h + l`（精确分解），因此 `h - q*n + l ≡ a*b (mod n)`
- 结果在 `(-n, n)` 范围内，可能需要一次 reduce 归到 `(-n/2, n/2)`
- AVX2 向量版 (`v4_mulmod`) 同时处理 4 个独立模乘

---

## 5. 混合基 FFT

### 5.1 支持的 NTT 长度

`ceil_ntt_size(x)` 返回满足 N ≥ x 的最小值，其中 N 属于集合：

```
{ 2^k,  3 × 2^k,  5 × 2^k }     (k ≥ 0)
```

**优势：** 最坏情况填充从 2x（仅 2 的幂）降到约 1.33x。

### 5.2 分解与调度：`fft_mixed`

对 N = m × 2^k（m ∈ {1, 3, 5}）：

1. **m = 1**（纯 2 的幂）：直接调用 `fft_auto(k)` — 小 k 用递归 DIF，L ≥ 27 用 Bailey
2. **m = 3**：先做一遍 radix-3 DIF pass，将 N 分为 3 个 2^k 块，再各自做 power-of-2 FFT
3. **m = 5**：先做一遍 radix-5 DIF pass，将 N 分为 5 个 2^k 块，再各自做 power-of-2 FFT

逆变换 `ifft_mixed` 顺序相反（DIT：先递归子块 IFFT，再做 radix-3/5 DIT pass，融合 1/m 缩放因子）。

### 5.3 Radix-3 蝴蝶（DIF 方向）

输入 a, b, c（三段各 sub_n 点），输出 A, B, C：

```
s = b + c
d = b - c
A = a + s
B = (a + s × neg_half + d × j3_half) × ω^j        // neg_half = -1/2 mod p
C = (a + s × neg_half - d × j3_half) × ω^(2j)     // j3_half = (ω₃ - ω₃²)/2
```

**向量化：** 4 路并行（V4），twiddle 通过 `v4_build_tw` 初始化后以 `step^4` 推进。

### 5.4 Radix-5 蝴蝶（DIF 方向）

使用 Karatsuba 风格的 6 次乘法（而非朴素的 8 次）：

```
s1 = b + e,  t1 = b - e
s2 = c + d,  t2 = c - d

p1 = s1 × c1h,  p2 = s2 × c2h      // 余弦半值
q1 = t1 × j1h,  q2 = t2 × j2h      // 正弦半值

alpha = a + p1 + p2
beta  = q1 + q2
gamma = a + (c12h项) - (p1 + p2)
delta = (j12s项)
```

其中 c1h, c2h, j1h, j2h 等是由 ω₅ 的实虚部预计算的常量（存于 `FftCtx`）。

---

## 6. Radix-2/4 核心 FFT

### 6.1 基本结构

- **DIF（前向）：** 先蝴蝶操作，再递归子问题
- **DIT（逆向）：** 先递归子问题，再蝴蝶操作

### 6.2 Basecase（深度 4，16 点）

`fft_basecase_4` 处理 16 个点的完整 radix-4 FFT：

1. **列 FFT（4 × radix-4 蝴蝶）：** 从内存加载 4 个 V4（列方向），执行蝴蝶变换
2. **4×4 转置：** 寄存器内通过 `v4_transpose`（unpack + permute2f128）
3. **行 FFT（4 × radix-4 蝴蝶）：** 对转置后的数据执行第二轮蝴蝶
4. **写回内存**

### 6.3 扩展 Basecase（深度 6–8，64–256 点）

`fft_basecase_extend` 递归地：
1. 执行 radix-4 蝴蝶（当前层）
2. 递归到 depth-2 的子问题
3. 直到 depth = 4（16 点 basecase）

**块大小：** `BLK_SZ = 256 = 2^8`，所有 ≤ 256 点的变换在 basecase 内完成。

### 6.4 块级递归：`fft_block`

处理 k 层、步长 S 的子问题：
- k ≥ 2：radix-4 蝴蝶 + 递归
- k = 1：radix-2 蝴蝶

### 6.5 外层递归：`fft_internal`

对 k 层问题做近似均分：k = k1 + k2，先处理块，再递归子组。

### 6.6 Twiddle 表

**分层存储：**
- `w2tab[0]`: 单个条目 `{1.0}`
- `w2tab[k]` (k ≥ 1): `2^(k-1)` 个条目
- 语义：`w2tab[k][r] = ω_{2^(k+1)}^(2 × bitrev(r, k-1) + 1)`

**逆 twiddle 恒等式：**
```
(w2tab[k][r])^{-1} = -w2tab[k][2^{k-1} - 1 - r]
```

**按需扩展：** 初始构建 depth 0–11（`W2TAB_INIT = 12`），更深层通过 `fit_depth()` 惰性分配。

---

## 7. Bailey 四步 FFT

### 7.1 触发条件

当 L ≥ `BAILEY_MIN_L = 27`（N ≥ 2^27 ≈ 1.34 亿点）时自动启用。

### 7.2 前向算法 (`fft_bailey`)

对 N = 2^L，分解 L = L1 + L2，R = 2^L1，C = 2^L2：

```
步骤 1: 对 R 行各做 C 点 FFT
步骤 2: 乘以 twiddle — data[r][c] *= ω_N^(r×c)
步骤 3: 转置 R×C → C×R（到临时缓冲区）
步骤 4: 对 C 行各做 R 点 FFT
步骤 5: 转置 C×R → R×C（写回原数组）
```

逆向 (`ifft_bailey`) 步骤相反（先转置、IFFT、再转置、乘逆 twiddle、IFFT）。

### 7.3 Cache-Oblivious 转置

**递归分治：** 在较长维度上二分，直到 tile 大小 ≤ 64。

**4×4 微核（`transpose_4x4_kernel`）：**
1. 4 次 `v4_load`（4 个源行各 4 个 double）
2. `v4_transpose`（2 次 unpack + 2 次 permute2f128）
3. 4 次 `v4_stream`（NT 写，绕过 L1/L2 直写主存）
4. `_mm_sfence()` 确保 NT 写完成

**列优先迭代：** Tile 内以列为外层循环（`c_outer`），限制同时活跃的写流为 4 行，避免 2 的幂步长造成的 L1 cache set 冲突。

### 7.4 Twiddle 乘法（4x 展开）

```
对每行 r:
  初始化 4 条独立 twiddle 链: tw0, tw1, tw2, tw3
  对每 16 列: 4 次 V4 mulmod, 各链以 step^16 推进
```

**性能：** 从 9–18 GB/s 提升到 45–49 GB/s（4x 展开消除了 twiddle 链的串行依赖）。

### 7.5 性能交叉点

| NTT 长度 (L) | Bailey / 直接 DIF 比率 |
|--------------|----------------------|
| 2^26         | 0.77x（直接更快）    |
| 2^27         | 1.19x（Bailey 领先） |
| 2^29         | 1.56x               |

**重要：** Bailey 输出的频域顺序与直接 DIF 不同——两个操作数必须使用同一条 FFT 路径。

---

## 8. CRT 重构

### 8.1 三阶段管线

```
Phase 1 (FP port, AVX2):   4 个系数 × Garner 链   → 混合基位
Phase 2 (INT port, scalar): Horner 求值            → 4-limb 整数
Phase 3 (INT port, scalar): 移位累加               → 输出 u64 limbs
```

### 8.2 Phase 1：SIMD Garner

预计算上下文 (`CrtCtx`)：
- `p[4]`: 4 个素数
- `vp[4], vpinv[4]`: 广播的 V4 素数和逆
- `vc01..vc23`: Garner 提升系数 `c_ij = p_i^{-1} mod p_j`

Garner 链（处理 4 个系数并行）：
```
v0 = round(r0)                                    ∈ [0, p0)
v1 = round((r1 - v0) × c01)                       ∈ [0, p1)
t  = round((r2 - v0) × c02);  v2 = round((t - v1) × c12)   ∈ [0, p2)
t  = round((r3 - v0) × c03);  t = round((t - v1) × c13)
v3 = round((t - v2) × c23)                        ∈ [0, p3)
```

**关键：** 每步 mulmod 后必须 `reduce + round + to_unsigned`（`GARNER_UINT` 宏）。
- 必须使用 unsigned `[0, p)` 而非 balanced `(-p/2, p/2)` 中间值
- 不做 round 则 0.06 的 mulmod 误差 × Garner 逆 (~2^49) = 灾难性误差

### 8.3 Phase 2：Horner 求值

将混合基位 (v0, v1, v2, v3) 转为整数：
```
result = v0 + v1 × p0 + v2 × (p0 × p1) + v3 × (p0 × p1 × p2)
```

使用 128-bit 乘法（MSVC: `_umul128` + `_addcarry_u64`; GCC: `__uint128_t`），输出 4 个 u64 limb。

### 8.4 Phase 3：移位累加

每 4 个系数一组，各系数间隔 80 bits（= 16 bits 对齐到 u64 边界上的偏移）：
- 系数 j 在组内偏移 `j × 16` bits
- `accum_shifted`: 将 4-limb Horner 结果移位后加到累加缓冲区
- `flush_to_output`: 进位传播写入最终输出 `z[]`

---

## 9. 代码结构

### 9.1 文件组织

```
ntt/
├── common.hpp              共享: 类型定义、位运算、模运算、对齐分配
├── api.hpp                 公开 API: big_multiply (u32), big_multiply_u64 (u64)
├── arena.hpp               共享: 缓冲池分配器（标签回收）
├── profile.hpp             共享: RAII 性能计时器
├── simd/
│   ├── avx2.hpp            __m256i SIMD（p30x3 用）
│   └── v4.hpp              __m256d SIMD（p50x4 用）
├── p30x3/                  u32 Montgomery 三素数引擎
│   ├── mont_scalar.hpp     标量 Montgomery 算术
│   ├── mont_vec.hpp        向量 Montgomery 算术（模板 B）
│   ├── root_plan.hpp       编译期 twiddle 预计算
│   ├── radix2.hpp          radix-2 DIF/DIT
│   ├── radix3.hpp          radix-3 外层 pass
│   ├── radix4.hpp          radix-4 核心蝴蝶
│   ├── radix5.hpp          radix-5 外层 pass
│   ├── cyclic_conv.hpp     8 点 twisted 循环卷积
│   ├── scheduler.hpp       NTT 调度器（前向/逆向/频域乘）
│   └── crt.hpp             3-prime CRT + 进位传播
└── p50x4/                  f64 FMA Barrett 四素数引擎
    ├── common.hpp          素数常量、内存分配、标量/向量 twiddle 构建
    ├── fft_ctx.hpp         每素数 FFT 上下文（分层 twiddle 表）
    ├── fft.hpp             radix-2/4 DIF/DIT + basecase（16–256 点）
    ├── bailey.hpp          Bailey 四步 FFT（转置 + twiddle）
    ├── mixed_radix.hpp     radix-3/5 pass + ceil_ntt_size + fft_mixed 调度
    ├── pointmul.hpp        频域逐点乘 + 缩放
    ├── crt.hpp             Garner (SIMD) + Horner (INT) CRT 重构
    └── multiply.hpp        80-bit 提取 + Ntt4 引擎类
```

### 9.2 依赖关系

```
common.hpp
  └── simd/v4.hpp
        └── p50x4/common.hpp
              └── p50x4/fft_ctx.hpp
                    ├── p50x4/fft.hpp
                    ├── p50x4/bailey.hpp
                    └── p50x4/mixed_radix.hpp
                          └── p50x4/pointmul.hpp
                                └── p50x4/crt.hpp
                                      └── p50x4/multiply.hpp ← 入口
```

### 9.3 设计原则

- **两个引擎平级共存：** p30x3 和 p50x4 各自独立，共享基础设施（common.hpp, arena.hpp, profile.hpp, simd/）
- **p30x3 保留 `namespace ntt`：** 因为其代码大量使用模板参数 `<B, Mod>`，移入子命名空间改动量过大
- **p50x4 使用 `namespace ntt::p50x4`：** 因为不模板化，命名空间隔离自然
- **命名规范：** 常量 `SCREAMING_CASE`（如 `PRIMES`, `MAGIC`），函数 `snake_case`（如 `fft_mixed`, `point_mul`），类 `PascalCase`（如 `FftCtx`, `Ntt4`）

---

## 10. 性能特性

### 10.1 对比 p30x3

| limb 数 | p50x4 / p30x3 耗时比 | 备注                             |
|---------|----------------------|----------------------------------|
| 1K      | ~2.9x                | p50x4 开销大（4 素数 + 80-bit）  |
| 10K     | ~1.8x                | 差距缩小                         |
| 100K    | ~1.3x                | FFT 占比增大，FMA 优势显现       |
| 1M      | ~1.1x                | 接近持平                         |
| 3.2M    | ~1.0x                | 持平（内存带宽瓶颈）            |

**趋势：** 小规模 p50x4 因 4-prime + 80-bit + CRT 开销而较慢；大规模 FMA 吞吐优势和更少的变换长度（80-bit 打包）使其追平。

### 10.2 混合基 vs 纯 2 的幂

混合基向量化修复后（原 radix-3/5 pass 标量实现慢 ~20x），混合基 NTT 长度与同级 2 的幂性能接近，填充节省直接转化为吞吐提升。

### 10.3 Bailey 四步分解

| 组成部分          | 占 Bailey 总时间 |
|-------------------|-----------------|
| 子 FFT (两轮)     | ~74%            |
| 转置 (两次)       | ~15%            |
| Twiddle 乘法      | ~6%             |
| 其他              | ~5%             |

Twiddle 4x 展开后从瓶颈（~30%）降到次要因素。当前瓶颈是子 FFT 本身。

---

## 附录 A：`FftCtx` 预计算常量一览

| 字段             | 含义                                          |
|------------------|-----------------------------------------------|
| `p, pinv`        | 素数 (double) 和 1/p                          |
| `w2tab[0..39]`   | 分层 twiddle 表                               |
| `neg_half_d`     | -1/2 mod p（radix-3 用）                      |
| `j3_half_d`      | (ω₃ - ω₃²)/2 mod p（radix-3 用）            |
| `inv3_d`         | 1/3 mod p                                     |
| `inv5_d`         | 1/5 mod p                                     |
| `c1h, c2h, ...`  | radix-5 蝴蝶的 6 个预计算常量                |
| `tw3_roots_d[]`  | radix-3 各深度的 twiddle root（前向）         |
| `tw3i_roots_d[]` | radix-3 各深度的 twiddle root（逆向）         |
| `tw5_roots_d[]`  | radix-5 各深度的 twiddle root（前向）         |
| `tw5i_roots_d[]` | radix-5 各深度的 twiddle root（逆向）         |
| `bailey_tmp`     | Bailey 转置用临时缓冲区                      |

## 附录 B：已知陷阱与教训

1. **Garner 链必须用 unsigned 中间值：** 有符号平衡表示下 Garner 位与无符号不同，Horner 求值只对无符号位正确
2. **FP mulmod 必须 round：** 不 round 的 ~0.06 误差乘以 Garner 逆 (~2^49) 导致灾难
3. **Twiddle 逆的镜像索引：** `mirror(2j_r)` 是 `2^{j_bits} - 1 - 2j_r`，而非 `2j_mr`
4. **Bailey 频域顺序不同于直接 DIF：** 两个操作数必须走同一条 FFT 路径
5. **Fused twiddle+transpose 不可行：** 需要 18 个 V4 寄存器，AVX2 仅 16 个 → 寄存器溢出 → 2x 慢
