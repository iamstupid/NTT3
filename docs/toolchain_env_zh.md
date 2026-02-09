# 本机工具链环境记录（2026-02-08）

## 1. 机器信息
- OS: `Microsoft Windows NT 10.0.19044.0` (Windows 10)
- PowerShell: `5.1.19041.1237`
- CPU: `AMD Ryzen 9 7950X 16-Core Processor`

## 2. Visual Studio / MSVC 安装
- 通过 `vswhere -all` 检测到:
  - `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools`
  - `C:\Program Files\Microsoft Visual Studio\2022\Community`
  - `C:\Program Files\Microsoft Visual Studio\18\Community` (Visual Studio Community 2026, 18.1.1)

## 3. 关键编译器版本
- VS 2026 (`18\Community`) 工具集:
  - Toolset dir: `...\MSVC\14.50.35717`
  - `cl.exe /Bv`: `19.50.35721.0`
  - `link.exe`: `14.50.35721.0`
- VS 2022 (`2022\Community`) 工具集:
  - Toolset dir: `...\MSVC\14.44.35207`
  - `cl.exe /Bv`: `19.44.35222.0`
  - `link.exe`: `14.44.35222.0`

## 4. CMake / Ninja
- 当前 shell `cmake --version`: `3.31.6-msvc6`
- VS bundled CMake:
  - Path: `C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`
  - Version: `4.1.1-msvc1`
- VS bundled Ninja:
  - Path: `C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe`
  - Version: `1.12.1`

## 5. Windows SDK
- 已安装 SDK lib 版本:
  - `10.0.19041.0`
  - `10.0.22621.0`
  - `10.0.26100.0`
- 当前 CMake (MSVC) 常见选择: `10.0.26100.0`

## 6. MSYS2 / GCC（对照）
- GCC path: `C:\msys64\ucrt64\bin\g++.exe`
- GCC version: `14.2.0`
- 说明: 这是 `MSYS2 UCRT64`，不是 Cygwin。

## 7. 项目构建目录与实际工具链
- `build/`
  - Generator: `Ninja`
  - Compiler: MSVC (`cl.exe`)
  - Make program: VS bundled `ninja.exe`
- `build-msvc/`
  - Generator: `Visual Studio 17 2022`
- `build-gcc/`
  - Generator: `Ninja`
  - Compiler: `C:/msys64/ucrt64/bin/g++.exe`
- `build-msvc-vcpkg/` (本次新增)
  - Generator: `Visual Studio 17 2022`
  - Toolchain: `C:/Users/zball/Documents/vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake`
  - Triplet: `x64-windows`
  - FFTW3_DIR: `C:/Users/zball/Documents/vcpkg/vcpkg/installed/x64-windows/share/fftw3`
  - 注意: 该目录实际选用的是 VS 2022 的 `cl 19.44.35222.0`

## 8. 项目 CMake 编译选项
- MSVC:
  - Compile: `/arch:AVX2 /O2 /Oi /GL /fp:fast`
  - Link: `/LTCG`
- GCC/Clang:
  - Compile: `-mavx2 -O2 -march=native`

## 9. FFTW 安装现状
- MSYS2 版本:
  - Header: `C:\msys64\ucrt64\include\fftw3.h`
  - DLL: `C:\msys64\ucrt64\bin\libfftw3-3.dll`
- vcpkg 版本 (MSVC 可直接链接):
  - Header: `C:\Users\zball\Documents\vcpkg\vcpkg\installed\x64-windows\include\fftw3.h`
  - Lib: `...\installed\x64-windows\lib\fftw3.lib`
  - DLL: `...\installed\x64-windows\bin\fftw3.dll`
  - 已安装 feature: `fftw3[avx2]:x64-windows`

## 10. 可复现命令（MSVC + vcpkg）
```powershell
cmake -S . -B build-msvc-vcpkg -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:/Users/zball/Documents/vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows

cmake --build build-msvc-vcpkg --config Release `
  --target bench_fftw_double bench_fft_double bench_single_prime_ntt
```

## 11. 当前对比数据文件
- MSVC + vcpkg + FFTW(avx2) 三轮中位数对比:
  - `bench_msvc_compare_20260208.csv`
- 字段说明:
  - `fftw_measure_ms`: FFTW (`--plan=measure`) 单次前向变换耗时
  - `fft_custom_ms`: 当前自研 `bench_fft_double` 单次前向变换耗时
  - `ntt_3prime_sum_ms`: 三个 prime 前向 NTT 耗时总和（近似完整 CRT 前向成本）
