$env:PATH = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;' + $env:PATH
$env:INCLUDE = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\shared'
$env:LIB = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.19041.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.19041.0\um\x64'
Set-Location 'C:\Users\zball\Documents\bigint\NTT3'

$src = $args[0]
$exe = $args[1]
if (-not $exe) { $exe = 'test_bigint.exe' }
if (-not $src) { $src = 'test_bigint.cpp' }

cl /std:c++17 /O2 /EHsc /arch:AVX2 $src /Fe:$exe
if ($LASTEXITCODE -eq 0) {
    Write-Host "--- Running $exe ---"
    & ".\$exe"
    Write-Host "--- Exit code: $LASTEXITCODE ---"
}
