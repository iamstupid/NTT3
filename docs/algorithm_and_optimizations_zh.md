# NTT3 算法与优化详解（源码级）

本文基于当前仓库源码（`ntt/` 全部核心模块 + `api/test/bench`）整理，目标是把实现中的算法、公式与优化策略逐层展开，做到可以直接对照代码维护。

## 1. 目标问题与记号

- 大整数以 `u32` limb 小端表示，基数为
  - `B = 2^32`
- 输入
  - `A = \sum_{i=0}^{n_a-1} a_i B^i`
  - `E = \sum_{j=0}^{n_b-1} b_j B^j`
- 目标卷积系数
  - `c_k = \sum_{i+j=k} a_i b_j`, `k = 0..(n_a+n_b-2)`
- 输出整数
  - `C = A * E = \sum_k c_k B^k`

`big_multiply()` 通过三模数 NTT 计算 `c_k`，再做 CRT 与 base-`2^32` 进位恢复最终 limb。

## 2. 端到端计算流水

对 `api.hpp` 的流程可抽象为：

1. 选择变换长度 `N`（limb 数，`N >= n_a+n_b`），并满足平滑形状约束。
2. 对每个素数 `p_t`（3 个）执行：
   - 输入按 `p_t` 约化并零填充到 `N`。
   - 前向 NTT。
   - 频域乘法（twisted 8-lane cyclic convolution）。
   - 逆 NTT。
3. 用 3 模数 CRT 还原每个卷积系数到 `[0, P)`（`P=p0*p1*p2`）。
4. 按 base `2^32` 做线性进位传播。

### 2.1 变换长度选择

代码使用 `ceil_smooth()`，平滑表来自 `common.hpp`：

- 表项是 limb 级长度 `N`。
- 对应向量数 `n_vec = N / 8`。
- `n_vec` 形态被限制为 `m * 2^k`，`m ∈ {1,3,5}`，并且 `2^k >= 4`（满足 twisted-conv 的批处理需求）。

这比“仅 2 的幂”填充更省（常见最坏填充从约 `2x` 降到约 `1.33x`）。

### 2.2 三个 NTT 素数

`multi/crt.hpp` 中固定：

- `p0 = 880803841 = 105 * 2^23 + 1`
- `p1 = 754974721 =  90 * 2^23 + 1`
- `p2 = 377487361 =  45 * 2^23 + 1`

都支持较大 2-adic 阶，且 `3 | (p-1)`, `5 | (p-1)`，可直接做 radix-3/radix-5 外层。

## 3. Montgomery 模运算体系（核心数值层）

对应 `mont/mont_scalar.hpp` 与 `mont/mont_vec.hpp`。

令 `R = 2^32`，模数 `M` 为某个 NTT 素数。

### 3.1 标量 Montgomery 约化

- `niv ≡ -M^{-1} (mod R)`
- 对 `x < M*R`：
  - `reduce(x) = (x + (x mod R) * niv * M) / R`

它满足 `reduce(x) ≡ x * R^{-1} (mod M)`。

### 3.2 本库的“一侧 Montgomery”用法

本库数据（系数数组）大多保持在普通剩余域；根与常数预先放 Montgomery 形式。

若 `bM = b*R (mod M)`，则

- `reduce(a * bM) ≡ a*b (mod M)`

即“普通系数 × Montgomery 常数 -> 仍是普通系数”。

这就是代码里大量 `mont_mul_bsm/mont_mul_precomp` 的设计基础：

- 数据不反复做域转换。
- 常量（根、1/3、1/5 等）预先转好，乘法可直接用 Montgomery pipeline。

### 3.3 向量范围不变量

`MontVec` 定义了一组区间变换，避免频繁 full reduction：

- `shrink`: `[0, 2M) -> [0, M)`
- `shrink2`: `[0, 4M) -> [0, 2M)`
- `dilate2`: 有符号差值包装回 `[0, 2M)`
- `add2/sub2`: 在 `[0, 2M)` 闭合
- `lazy_add/lazy_sub`: 暂不约化，允许进入 `[0, 4M)`

这样蝶形内部只在必要点收缩，减少比较/减法次数。

## 4. 根表预计算 RootPlan

对应 `roots/root_plan.hpp`。

## 4.1 原根与幂链

1. 搜索 primitive root `g`（检查 `g^{(M-1)/q} != 1` 对所有素因子 `q | (M-1)`）。
2. 构造 `2^k` 阶根链（`k = v2(M-1)`）：
   - `omega_{2^k}`
   - 逐级平方得到 `omega_{2^i}` 与逆根链。
3. `img = omega_4`，用于 radix-4 中 `i = sqrt(-1)` 角色。

## 4.2 Ruler-sequence 更新表

`rt3/rt3i` 存打包状态（每项含 `r, r^2, r^3` 及各自 `*niv`），供 scheduler 用 `ctz(~k)` 做 O(1) 根状态跳转。

核心思想：

- 遍历块时，twiddle 指数按“进位链”变化。
- `ctz(~k)` 等于 `k` 的 trailing-ones 个数，正好定位这次进位深度。
- 用预计算 jump factor 一次更新 `(r, r^2, r^3)` 状态。

避免逐点 `pow` 或大 root table 随机访存。

## 4.3 radix-3 / radix-5 常数

### radix-3

设 `ω = ω3`：

- `neg_half = -1/2`
- `j3_half = (ω - ω^2)/2`
- `inv3 = 1/3`

### radix-5

设 `ω = ω5`：

- `c1h = (ω + ω^4)/2`
- `c2h = (ω^2 + ω^3)/2`
- `j1h = (ω - ω^4)/2`
- `j2h = (ω^2 - ω^3)/2`
- `inv5 = 1/5`

同时预计算 `tw3_root/tw5_root` 及逆根链，供外层 mixed-radix pass。

## 4.4 逆变换缩放因子 `compute_scale()`

函数返回：

- `fx = (N^{-1} mod M) * R^2 mod M`

其中 `N` 是 **向量级长度**（`n_vec`，2 的幂）。

代码写法：

- `invN = M - ((M-1)/N)`（因 `N | (M-1)`）
- `fx = mul_s(invN, r3)`，而 `r3 = R^3`，`mul_s` 自带一次 `R^{-1}`，故得到 `invN * R^2`。

为什么需要 `R^2` 而不是常见 `R`：

- 频域 `twisted_conv` 里每个系数做过一次 Montgomery reduction，额外引入了 `R^{-1}` 因子。
- 逆变换最内层融合缩放时补一阶 `R`，最终回到普通剩余域。

## 5. Base-2 引擎：radix-4/radix-2 + 分块调度

对应 `engine/scheduler.hpp`、`kernels/radix4.hpp`、`kernels/radix2.hpp`。

### 5.1 Forward (`fwd_b2`) 三阶段

1. 若 `log2(n)` 为奇数，先做一次 radix-2 DIF。
2. 走 `r=0` 的纯 radix-4 链（无 twiddle）。
3. 进入 j-block 遍历（`BLOCK_SIZE=64` vec），按 `ctz(~k)` 更新 twiddle 状态并执行 twiddle-fused radix-4 DIF。

### 5.2 Inverse (`inv_b2`) 三阶段

1. j-block DIT：
   - 最内层 `L=1` 使用 `dit_butterfly_scale`，把 `N^{-1}` 缩放融合进去。
   - 外层用 `dit_butterfly`。
2. 若 `log2(n)` 为奇数，尾部 radix-2 DIT。
3. 末尾统一 `shrink` 到 `[0, M)`。

### 5.3 radix-2 蝶形公式

DIF:

- `y0 = x0 + x1`
- `y1 = x0 - x1`

DIT 同形（逆序应用），并在末端收缩范围。

### 5.4 radix-4 蝶形（实现等价式）

#### DIF no-twiddle

设输入 `(f0,f1,f2,f3)`：

- `g1 = f1 + f3`
- `g3 = i*(f1 - f3)`
- `g0 = f0 + f2`
- `g2 = f0 - f2`

输出：

- `y0 = g0 + g1`
- `y1 = g0 - g1`
- `y2 = g2 + g3`
- `y3 = g2 - g3`

#### DIF twiddle-fused

代码先把分支乘上 `(r, r^2, -r^3)`，再做同型组合，减少单独 twiddle pass。

#### DIT

`dit_butterfly / dit_butterfly_scale` 是对应逆过程；`scale` 版本在 `p0` 分支融合 `fx`，其他分支通过 root-state 自然携带同一尺度。

## 6. Mixed-radix 外层（m=3/5）

对应 `kernels/radix3.hpp`、`kernels/radix5.hpp`。

设向量长度 `n = m * 2^k`，子长度 `s = 2^k`。

- Forward：先做 outer DIF-m，再对每个子数组做 `fwd_b2(s)`。
- Inverse：先每个子数组 `inv_b2(s)`，再 outer DIT-m（融合 `1/m`）。

## 6.1 radix-3 外层公式

设 `(a,b,c)`，

- `s = b + c`
- `d = b - c`
- `h = a - s/2`
- `j = ((ω3 - ω3^2)/2) * d`

则 DIF 基础输出：

- `y0 = a + s`
- `y1 = h + j`
- `y2 = h - j`

对 `j>0` 再乘 twiddle：

- `y1 *= tw^j`
- `y2 *= tw^{2j}`

DIT 中按逆 twiddle 解开并乘 `inv3`，符号与 DIF 对偶（代码中对 `y1/y2` 的 `±j` 顺序与 DIF 对应）。

## 6.2 radix-5 外层公式（Karatsuba 化简）

设输入 `(a,b,c,d,e)`，

- `s1 = b+e`, `t1 = b-e`
- `s2 = c+d`, `t2 = c-d`

定义：

- `alpha = a + c1h*s1 + c2h*s2`
- `gamma = a + c2h*s1 + c1h*s2`
- `beta  = j1h*t1 + j2h*t2`
- `delta = j2h*t1 - j1h*t2`

DIF 输出：

- `y0 = a + s1 + s2`
- `y1 = alpha + beta`
- `y2 = gamma + delta`
- `y3 = gamma - delta`
- `y4 = alpha - beta`

并对 `y1..y4` 乘 `tw^{j,2j,3j,4j}`。

Karatsuba 降乘法数：

- `c`-项从 4 乘降到 3 乘：
  - `p1=c1h*s1`, `p2=c2h*s2`, `p3=(c1h+c2h)*(s1+s2)`
  - `gamma = a + (p3 - p1 - p2)`
- `j`-项同理 3 乘：
  - `q1=j1h*t1`, `q2=j2h*t2`, `q3=(j1h+j2h)*(t1-t2)`
  - `delta = q3 - q1 + q2`

DIT 过程使用逆 twiddle 并融合 `inv5`，输出符号顺序与 DIF 互逆。

## 7. 频域乘法：8-lane twisted cyclic convolution

对应 `kernels/cyclic_conv.hpp`。

### 7.1 数学模型

对每个 AVX2 向量，令

- `A(z)=a0+a1 z+...+a7 z^7`
- `B(z)=b0+b1 z+...+b7 z^7`

计算

- `C(z)=A(z)B(z) mod (z^8 - w)`

系数公式：

- `c_k = \sum_{i+j=k} a_i b_j + w * \sum_{i+j=k+8} a_i b_j`, `k=0..7`

即高于 7 次的项按 `z^8 = w` 折返，避免把每向量扩到 16 点再卷积。

### 7.2 批 4 优化 `conv8_batch4`

每次处理 4 个向量，构造 4 个 `w`：

- `{RR, -RR, RR*i, -RR*i}`（代码用 `[0,2M)` 表示负号）

并一次性完成 4 组 8x8 卷积累加，最后各做一次 Montgomery reduction。

优势：

- 降低循环与访存开销。
- 使用 AVX2 `mul_epu32` 分偶/奇 lane 累加 64-bit 部分和。
- 在批内共享逻辑，提升吞吐。

### 7.3 Mixed-radix 时的 twist 偏移

`NTTScheduler::freq_multiply()` 中，若 `n = m*2^k`：

- 每个子数组 `r` 的初始 twist 取 `ω_N^r`（`N=n_vec`）。
- 通过 `rr_init` 传给 `twisted_conv()`。

这样保证 mixed-radix 拆分后仍满足全局卷积相位关系。

## 8. CRT 重建与进位

对应 `multi/crt.hpp`。

设 `P = p0*p1*p2`，预计算

- `PI_t = (P/p_t) * (P/p_t)^{-1} mod p_t`

则

- `x = (n0*PI0 + n1*PI1 + n2*PI2) mod P`

代码用 `u128` 实现 96~128 位运算，并用高位近似商优化取模：

1. `sum = n0*PI0 + n1*PI1 + n2*PI2`
2. `q ≈ floor(sum.hi / P.hi)`（double reciprocal）
3. `r = sum - q*P`
4. 若 `r >= P` 再减一次 `P`（至多一次修正）

得到每个卷积系数后，线性进位：

- `t_i = r_i + carry`
- `out_i = t_i mod 2^32`
- `carry = floor(t_i / 2^32)`

循环直到输出长度。

## 9. 内存与工程优化细节

## 9.1 输入约化与填充（单趟）

`reduce_and_pad()`：

- 向量化加载输入。
- 逐级减 `8M/4M/2M`（若常量不溢出则启用）近似“快速模”。
- 尾部标量处理。
- 一次 `memset` 零填充。

减少了“先拷贝再约化”的额外带宽。

## 9.2 Arena 池化 + 指针高位 tag

`arena.hpp`：

- bin 按平滑尺寸顺序编码。
- `alloc` 可向后最多借 3 个 bin（复用略大缓存，减少 malloc/free）。
- 在指针 bit62~63 存 offset tag，`dealloc` 还原到实际 bin。

这对大量重复 benchmark/calls 有明显分配器收益。

## 9.3 分块遍历与缓存友好

- 固定块长 `BLOCK_SIZE=64 vec`。
- j-block 遍历让内层访问更局部。
- 根更新用状态机，不依赖大 twiddle 表随机读取。

## 9.4 常量折叠与预计算

- `RootPlan` 在静态对象中一次构建。
- 乘法常量同时预存 `b` 与 `b*niv`，便于 `mont_mul_precomp`。
- `img/imgniv`、`inv3/inv5`、radix-5 组合常量都提前算好。

## 10. 复杂度与边界

- 单模数卷积：`O(N log N)`
- 三模数总复杂度：`O(3N log N)`（常数项受 SIMD 与 kernel 设计显著影响）
- 额外空间：
  - 4 个 `N`-limb 缓冲（`f0/f1/f2/g`）+ root/state 小缓存
- 当前平滑表最大 `N = 25,165,824` limbs（源码注释注明更大规模可能触发 CRT 安全边界问题）
- 需要 AVX2。

## 11. 模块与职责映射

- `ntt/common.hpp`：基础类型、位运算、平滑长度表、常量
- `ntt/arena.hpp`：NTT 缓冲池化分配
- `ntt/simd/avx2.hpp`：AVX2 包装层
- `ntt/mont/mont_scalar.hpp`：标量 Montgomery
- `ntt/mont/mont_vec.hpp`：向量 Montgomery 与 range 操作
- `ntt/roots/root_plan.hpp`：根与 twiddle 常量预计算
- `ntt/kernels/radix2.hpp`：radix-2 kernel
- `ntt/kernels/radix4.hpp`：radix-4 kernel（DIF/DIT）
- `ntt/kernels/radix3.hpp`：mixed-radix 外层 radix-3
- `ntt/kernels/radix5.hpp`：mixed-radix 外层 radix-5（Karatsuba 化简）
- `ntt/kernels/cyclic_conv.hpp`：8-lane twisted cyclic convolution
- `ntt/engine/scheduler.hpp`：整体调度（forward/inverse/freq multiply）
- `ntt/multi/crt.hpp`：三模数 CRT + carry
- `ntt/api.hpp`：公开 API 与三素数流水线

## 12. 实现要点总结（可作为维护检查单）

1. 所有 kernel 需维持区间不变量（`[0,M)/[0,2M)/[0,4M)`），否则会破坏 lazy reduction 假设。
2. mixed-radix 的 forward/inverse 顺序必须保持“外层一次 + 子 NTT 若干次”的互逆结构。
3. `twisted_conv` 的 `R^{-1}` 因子必须由 inverse scale 端正确补偿。
4. CRT 还原需确保 `P` 对目标系数范围足够；长度上限变化时应重新评估。
5. arena 的 tag 位依赖 64-bit 指针高位可用性（当前桌面 x64 平台成立）。

---

如果后续要继续扩展（例如 AVX-512、更多素数、Karatsuba/Toom 前端切换阈值），建议先保持本文中的“数学等价式 + 区间不变量 + 域表示约定”三项不变，再做局部替换。
