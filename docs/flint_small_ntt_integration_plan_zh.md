# NTT3 引入 FLINT `fft_small` 思路的集成方案（草案）

## 0. 结论先行

基于 `flint_fft_small_analysis.md` 和 FLINT 源码（`experimental/flint-src`，commit `569bcf54dd4af73e00a185f42832226ad8edbebb`）的对照分析，建议按下面顺序推进：

1. **先移植 FLINT 的“profile + 通用 CRT + 输入打包”思想**，保持现有 AVX2 Montgomery NTT/`twisted_conv` 主内核不动。  
2. **再考虑截断 FFT/TFT**（仅在确认大量 padding 浪费时启用）。  
3. **最后才评估 double-FMA 后端**，作为可选路径，不替换主线整数 NTT。

这个顺序风险最低，也最符合当前 NTT3 的代码结构。

---

## 1. 研究基线（对照对象）

主要参考文件：

- `flint_fft_small_analysis.md`
- `experimental/flint-src/src/fft_small.h`
- `experimental/flint-src/src/fft_small/sd_fft.c`
- `experimental/flint-src/src/fft_small/sd_ifft.c`
- `experimental/flint-src/src/fft_small/sd_fft_ctx.c`
- `experimental/flint-src/src/fft_small/mpn_mul.c`
- `experimental/flint-src/src/fft_small/mpn_helpers.c`
- `experimental/flint-src/src/machine_vectors.h`
- `experimental/flint-src/src/fft_small/mulmod_satisfies_bounds.c`

与本库对应主线文件：

- `ntt/api.hpp`
- `ntt/engine/scheduler.hpp`
- `ntt/roots/root_plan.hpp`
- `ntt/kernels/cyclic_conv.hpp`
- `ntt/multi/crt.hpp`
- `ntt/common.hpp`

---

## 2. FLINT 里最值得借鉴的机制

### 2.1 Profile 驱动的参数选择（高优先级）

FLINT 核心不是固定 `np/bits`，而是按输入规模动态选 profile。  
关键约束（以 FLINT 的 64-bit limb 版本为例）：

\[
\left\lceil \frac{64 \cdot b_n}{bits} \right\rceil \cdot 2^{2\cdot bits} \le \prod_{i=0}^{np-1} p_i
\]

NTT3 是 32-bit limb，可改成：

\[
\left\lceil \frac{32 \cdot b_n}{bits} \right\rceil \cdot 2^{2\cdot bits} \le P(np)
\]

其中 `P(np)` 是所选 `np` 个素数乘积。

### 2.2 经典余项因子 CRT（高优先级）

FLINT 用的是：

\[
x = \left(\sum_{i=0}^{np-1} C_i \cdot \left(r_i \cdot C_i^{-1} \bmod p_i\right)\right)\bmod P,\quad C_i=P/p_i
\]

优点是主循环变成“**多精度常量 × 单 limb**”累加，适合流水化，不依赖 Garner 链式数据依赖。  
对 NTT3 的意义：可以从固定三素数 CRT 扩展到可变 `np`。

### 2.3 输入打包（base-2^bits）与 fast/slow 双路径（高优先级）

FLINT 在 `mpn_mul.c` 里通过 `bits` 控制变换长度，同时维护快速路径（向量化）和慢速兜底路径。  
NTT3 当前是固定 base-2^32 直接卷积，移植后能在大规模时更灵活地折中“变换长度 vs CRT 位宽”。

### 2.4 TFT/截断递归（中优先级）

`sd_fft_trunc` / `sd_ifft_trunc` 能显著减少非满长计算。  
但它和当前 NTT3 的 `twisted_conv` 批处理、lazy reduction、mixed-radix 调度耦合较深，建议放到第二阶段。

### 2.5 三阶段线程调度（中优先级）

FLINT 把 `mod -> fft -> crt` 三段分别并行。  
NTT3 当前偏单线程内核，后续可按同样阶段化引入线程池，但不是当前最紧急瓶颈。

---

## 3. 与 NTT3 当前主线的关键差异

### 3.1 数值域差异

- FLINT：`double` + FMA mulmod，残差常在对称区间（如 `[-p/2,p/2]`）。
- NTT3：`u32` Montgomery，核心区间约束是 `[0,M)/[0,2M)/[0,4M)`。

结论：**不能直接复用 FLINT 的算术内核代码**，只能迁移设计思想。

### 3.2 频域乘法模型差异

- FLINT：普通点乘（或平方）+ 缩放。
- NTT3：`x^8-w` twisted cyclic convolution（`ntt/kernels/cyclic_conv.hpp`）。

结论：profile/CRT 可以复用思想，但必须保留 `twisted_conv` 兼容条件。

### 3.3 长度/根约束差异

NTT3 mixed-radix 依赖 `m in {1,3,5}`，并要求每个参与素数都满足：

\[
2^k \mid (p_i-1), \quad m \mid (p_i-1)
\]

若扩展 prime 池，必须给每个素数打能力标签（`max_k`、`has_m3`、`has_m5`），并按交集选变换形状。

---

## 4. 建议的集成架构

### 4.1 新增数据结构（建议）

```cpp
struct PrimeSpec {
    u32 mod;
    u32 two_adicity;   // v2(mod-1)
    bool has_m3;
    bool has_m5;
};

struct RuntimeProfile {
    u32 np;            // 使用前 np 个 prime
    u32 bits;          // 输入打包 base = 2^bits
    u32 m;             // 变换外层因子 {1,3,5}
    u32 k;             // N = m * 2^k
};

struct CRTPlan {
    u32 np;
    // P, cofactors, cofactor inverse mod p_i ...
};
```

### 4.2 执行流水（保持主内核不动）

1. `profile_select(an,bn)` 选 `np/bits/m/k`。  
2. 把输入从 base-2^32 打包为 base-2^bits 系数。  
3. 每个 prime 仍走现有 `NTTScheduler::forward/freq_multiply/inverse`。  
4. 用通用 `CRTPlan` 重建并进位。

也就是：**先动外层调度和数据准备，不动 `radix2/3/4/5 + twisted_conv` 的热点代码**。

---

## 5. 与 `twisted_conv` 的兼容性约束

当前 mixed-radix 频域乘法（`ntt/engine/scheduler.hpp`）对第 `r` 个子变换使用：

\[
rr_r = \omega_N^r,\quad N=n_{vec}
\]

并传给 `twisted_conv(..., rr_r)`。

只要 profile 改动不改变这两个事实，内核兼容：

1. 每个 prime 的 forward/inverse 仍产出同定义的频域值；  
2. `N=m\cdot2^k` 与该 prime 的可用根阶匹配。

若某个 profile 的 prime 不支持 `m=3/5`，就必须降级 `m=1`（纯 2 幂长度）。

---

## 6. 分阶段落地计划

### Phase A（推荐先做）

- 目标：引入 profile 框架 + 通用 CRT（保留当前 3-prime 快路径）。
- 代码点：
  - 新增 `ntt/profile/`（选择 `np/bits/m/k`）
  - 新增 `ntt/multi/crt_plan.hpp`（通用 CRT）
  - `ntt/api.hpp` 增加 runtime profile 路径
- 验证：
  - 与现有 3-prime 结果 bit-exact；
  - profile 固定为旧参数时，性能回归 < 2%。

### Phase B

- 目标：引入 base-2^bits 打包（fast/slow 双路径）。
- 代码点：
  - 新增 `ntt/pack/coeff_pack.hpp`
  - AVX2 快速路径 + 标量兜底
- 验证：
  - 不同 `bits` 下与 schoolbook/GMP 对比正确；
  - 测 `unbalanced` 乘法收益。

### Phase C（可选）

- 目标：截断 FFT/TFT。
- 代码点：
  - `NTTScheduler` 增加 `forward_trunc/inverse_trunc`
  - 与 `twisted_conv` 的长度对齐逻辑
- 风险：代码复杂度高，调错成本高。

### Phase D（可选）

- 目标：独立 double-FMA 后端（实验性）。
- 建议放 `experimental/`，不并入主线热路径。

---

## 7. 基准与验收标准

建议统一输出：

- `time_total`
- `time_pack`
- `time_forward`
- `time_freqmul`
- `time_inverse`
- `time_crt`
- `effective GB/s`

测试矩阵：

1. 平衡乘法：`2^k` limbs，`k=10..24`。  
2. 非平衡乘法：`an:bn = 1:4, 1:16`。  
3. 特殊形态：`a==b`（平方）单独统计。  
4. mixed-radix 与 pure power-of-two 两组分别测。

验收门槛（建议）：

- 正确性：随机回归 1e4 组无错误。  
- 性能：大尺寸（>= `2^20` limbs）整体提升 >= 8%。  
- 维护性：主线 `scheduler` 热路径代码增量可控，不引入额外分支污染。

---

## 8. 许可与工程边界

FLINT 是 LGPL 许可证。  
建议采取“**只迁移算法与结构思想，代码重新实现**”策略，避免直接复制实现代码进入主线。

---

## 9. 我对当前库的具体建议

最优先做的不是重写 NTT 内核，而是：

1. **通用 CRT + profile 选择器**（直接打开 `np/bits` 维度）。  
2. **base-2^bits 打包**（给 profile 提供可用杠杆）。  
3. **平方专用快路径**（FLINT 已验证有效，工程成本低）。

这样能先吃到 FLINT 的“系统层收益”，同时保持你现在已经很强的 `twisted_conv + scheduler` 内核优势。

