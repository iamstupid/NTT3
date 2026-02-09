# 代码结构重组记录

## 变更概述

将 `ntt/` 目录从"p30x3 直接平铺 + sd 子目录"的非对称结构，重组为两个引擎平级共存的对称结构。

## 变更前

```
ntt/
├── common.hpp
├── api.hpp
├── arena.hpp
├── mont/
│   ├── mont_scalar.hpp
│   └── mont_vec.hpp
├── roots/
│   └── root_plan.hpp
├── kernels/
│   ├── radix2.hpp
│   ├── radix3.hpp
│   ├── radix4.hpp
│   ├── radix5.hpp
│   └── cyclic_conv.hpp
├── engine/
│   └── scheduler.hpp
├── multi/
│   └── crt.hpp
├── simd/
│   └── avx2.hpp
└── sd/                         ← 嵌套在内，非对称
    ├── sd_common.hpp
    ├── sd_fft_ctx.hpp
    ├── sd_fft.hpp
    ├── sd_bailey.hpp
    ├── sd_mixed_radix.hpp
    ├── sd_pointmul.hpp
    ├── sd_crt.hpp
    └── sd_multiply.hpp
```

**问题：**
1. p30x3 引擎的文件分散在 4 个子目录（mont/, roots/, kernels/, engine/, multi/），而 sd 引擎的文件全在一个 sd/ 里——两者不对称
2. `sd` 名称无意义（源自 FLINT 的 "small dot" 内部命名，外部读者无从理解）
3. 所有 sd 文件以 `sd_` 前缀命名，进入 `sd/` 目录后前缀冗余
4. 内部标识符也全部带 `sd_` 前缀（如 `sd_fft_basecase_4`、`SdFftCtx`），在 `namespace ntt::sd` 内同样冗余
5. 常量风格混合（`kPrimes` camelCase vs `BLK_SZ` SCREAMING_CASE）

## 变更后

```
ntt/
├── common.hpp                  共享基础设施
├── api.hpp                     公开 API
├── arena.hpp                   缓冲池分配器
├── profile.hpp                 性能计时
├── simd/
│   ├── avx2.hpp                __m256i 原语
│   └── v4.hpp                  __m256d 原语
├── p30x3/                      u32 三素数引擎（~30-bit primes × 3）
│   ├── mont_scalar.hpp
│   ├── mont_vec.hpp
│   ├── root_plan.hpp
│   ├── radix2.hpp
│   ├── radix3.hpp
│   ├── radix4.hpp
│   ├── radix5.hpp
│   ├── cyclic_conv.hpp
│   ├── scheduler.hpp
│   └── crt.hpp
└── p50x4/                      f64 四素数引擎（~50-bit primes × 4）
    ├── common.hpp
    ├── fft_ctx.hpp
    ├── fft.hpp
    ├── bailey.hpp
    ├── mixed_radix.hpp
    ├── pointmul.hpp
    ├── crt.hpp
    └── multiply.hpp
```

## 具体变更

### 目录移动

| 原路径                              | 新路径                          |
|-------------------------------------|---------------------------------|
| `ntt/mont/mont_scalar.hpp`          | `ntt/p30x3/mont_scalar.hpp`    |
| `ntt/mont/mont_vec.hpp`             | `ntt/p30x3/mont_vec.hpp`       |
| `ntt/roots/root_plan.hpp`           | `ntt/p30x3/root_plan.hpp`      |
| `ntt/kernels/radix2.hpp`            | `ntt/p30x3/radix2.hpp`         |
| `ntt/kernels/radix3.hpp`            | `ntt/p30x3/radix3.hpp`         |
| `ntt/kernels/radix4.hpp`            | `ntt/p30x3/radix4.hpp`         |
| `ntt/kernels/radix5.hpp`            | `ntt/p30x3/radix5.hpp`         |
| `ntt/kernels/cyclic_conv.hpp`       | `ntt/p30x3/cyclic_conv.hpp`    |
| `ntt/engine/scheduler.hpp`          | `ntt/p30x3/scheduler.hpp`      |
| `ntt/multi/crt.hpp`                 | `ntt/p30x3/crt.hpp`            |
| `ntt/sd/sd_common.hpp`              | `ntt/p50x4/common.hpp`         |
| `ntt/sd/sd_fft_ctx.hpp`             | `ntt/p50x4/fft_ctx.hpp`        |
| `ntt/sd/sd_fft.hpp`                 | `ntt/p50x4/fft.hpp`            |
| `ntt/sd/sd_bailey.hpp`              | `ntt/p50x4/bailey.hpp`         |
| `ntt/sd/sd_mixed_radix.hpp`         | `ntt/p50x4/mixed_radix.hpp`    |
| `ntt/sd/sd_pointmul.hpp`            | `ntt/p50x4/pointmul.hpp`       |
| `ntt/sd/sd_crt.hpp`                 | `ntt/p50x4/crt.hpp`            |
| `ntt/sd/sd_multiply.hpp`            | `ntt/p50x4/multiply.hpp`       |

### 命名空间

- `namespace ntt::sd` → `namespace ntt::p50x4`
- p30x3 保留 `namespace ntt`（模板化代码改动量过大，不值得搬）

### 标识符重命名

| 原名                    | 新名              | 类别     |
|-------------------------|--------------------|----------|
| `SdFftCtx`              | `FftCtx`           | 类       |
| `SdNtt4`                | `Ntt4`             | 类       |
| `sd_fft`                | `fft`              | 函数     |
| `sd_ifft`               | `ifft`             | 函数     |
| `sd_fft_bailey`         | `fft_bailey`       | 函数     |
| `sd_ifft_bailey`        | `ifft_bailey`      | 函数     |
| `sd_fft_mixed`          | `fft_mixed`        | 函数     |
| `sd_ifft_mixed`         | `ifft_mixed`       | 函数     |
| `sd_scale_mixed`        | `scale_mixed`      | 函数     |
| `sd_point_mul`          | `point_mul`        | 函数     |
| `sd_scale`              | `scale`            | 函数     |
| `sd_fft_basecase_*`     | `fft_basecase_*`   | 函数     |
| `sd_ifft_basecase_*`    | `ifft_basecase_*`  | 函数     |
| `sd_radix3_dif_pass`    | `radix3_dif_pass`  | 函数     |
| `sd_radix3_dit_pass`    | `radix3_dit_pass`  | 函数     |
| `sd_radix5_dif_pass`    | `radix5_dif_pass`  | 函数     |
| `sd_radix5_dit_pass`    | `radix5_dit_pass`  | 函数     |
| `SD_GARNER_UINT`        | `GARNER_UINT`      | 宏       |
| `kPrimes`               | `PRIMES`           | 常量     |
| `kMagic`                | `MAGIC`            | 常量     |
| `kMagicBits`            | `MAGIC_BITS`       | 常量     |

### #include 路径更新

p30x3 文件原先跨目录引用（如 `"../mont/mont_vec.hpp"`），现改为同目录引用（`"mont_vec.hpp"`）。引用共享头文件（如 `"../common.hpp"`）路径不变。

p50x4 文件原先引用 `"sd_common.hpp"`，现改为 `"common.hpp"`。引用 `"../simd/v4.hpp"` 等共享头路径不变。

## 设计决策

1. **名称 `p30x3` / `p50x4`：** 编码了核心参数——素数大小和素数个数，简洁且自描述
2. **p30x3 不创建子命名空间：** 其代码通过模板 `<B, Mod>` 参数化，每处调用都需要改签名，工程量大而收益低
3. **arena.hpp 保持共享：** 未来 p50x4 也将使用 arena 替代自己的 `alloc_doubles`/`free_doubles`
4. **SIMD 风格差异保留：** p30x3 用 `struct Avx2`（作为模板参数 B），p50x4 用自由函数 `v4_*()`——这是架构驱动的差异，不值得强行统一
