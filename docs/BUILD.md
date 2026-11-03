# Building Carbon-OCR

Carbon-OCR builds with CMake on Windows, Linux and macOS. The CPU build needs only a C++20 compiler, because CMake downloads every dependency at a pinned version and checks its SHA-256. The GPU build also needs NVIDIA's CUDA toolkit and TensorRT, which you unpack yourself.

**Contents**
1. Requirements
2. The CPU build
3. The GPU build
4. Options
5. The Python module
6. Tests

## <img src="assets/chip-01.svg" height="22" alt=""> 1. Requirements

| what | version |
|---|---|
| CMake | 3.25 or newer (the presets need it) |
| a C++20 compiler | MSVC 2022, GCC 11+ or Clang 14+ |
| Ninja | for the `ninja-*` presets |
| CUDA | 12.9 or 13.x, only for the GPU build |
| TensorRT | 11, matching the CUDA, only for the GPU build |
| Python | 3.12, only for the Python module |

CMake downloads these and checks each one's SHA-256:

| dependency | pinned | used for |
|---|---|---|
| ONNX Runtime | 1.31.0 | the CPU backend that runs any ONNX model |
| PDFium | chromium/8076 | reading PDFs, in the `carbon-pdf` helper processes |
| libdeflate | 1.24 | the searchable PDF's images |
| stb_image | 2.30 | reading PNG, JPEG, BMP and PNM pages |
| GoogleTest | 1.18.0 | the C++ tests |

ONNX Runtime downloads exist for Windows x64, Linux x64 and macOS arm64. On any other platform, set `ONNXRUNTIME_ROOT` to an unpacked release, or build with `-DCARBON_WITH_ONNXRUNTIME=OFF`; the engine's own int8 backend still runs. PDFium downloads exist for Windows x64, Linux x64 and arm64, and macOS arm64. Elsewhere, set `PDFIUM_ROOT` to an unpacked build, or turn PDFs off with `-DCARBON_WITH_PDFIUM=OFF`.

## <img src="assets/chip-02.svg" height="22" alt=""> 2. The CPU build

```shell
# Linux, macOS, or a Visual Studio developer shell
cmake --preset ninja-release
cmake --build --preset ninja-release
ctest --preset ninja-release

# Windows, Visual Studio 2022
cmake --preset vs2022
cmake --build --preset vs2022
ctest --preset vs2022
```

The build goes to `build/<preset>/`, and the programs are in `build/<preset>/cli/`:
- `carbon-ocr`, the command line.
- `carbon-pdf`, the helper that PDFs are rendered in. It must stay beside `carbon-ocr`, because the engine looks for it there.

On Windows, the DLLs both programs need are copied next to them.

The CPU kernels are compiled once for each instruction set the compiler supports, and the engine picks one at run time: AVX2, AVX-VNNI and AVX-512 on x86-64; NEON and the dot product on Arm. The rest of the library stays at the baseline (x86-64 or Armv8-A), so one binary runs on any CPU of its architecture. To see the kernels a machine runs, use `carbon-ocr cpus`.

## <img src="assets/chip-03.svg" height="22" alt=""> 3. The GPU build

1. Install or unpack CUDA 12.9 or 13.x.
2. Unpack a TensorRT 11 release that matches the CUDA, and set `TensorRT_ROOT` to that folder, either as an environment variable or as a CMake cache variable.
3. Put TensorRT's and CUDA's library folders on `PATH` (Windows) or `LD_LIBRARY_PATH` (Linux).
4. Configure with the `ninja-trt` preset (on Windows, in a Visual Studio developer shell) or the `vs2022-trt` preset:

```shell
cmake --preset ninja-trt -DCARBON_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build --preset ninja-trt
ctest --preset ninja-trt
```

`CMAKE_CUDA_ARCHITECTURES` defaults to `native`, which builds for the GPU in the machine you build on. Set it to build for others: 89 for Ada (an L4, an RTX 40-series card), 86 or 80 for Ampere. With a CUDA that is unpacked rather than installed, also pass `-DCMAKE_CUDA_COMPILER=<cuda>/bin/nvcc` and `-DCUDAToolkit_ROOT=<cuda>`. The `ninja-cuda` preset builds the CUDA kernels without TensorRT.

TensorRT 11 builds strongly typed networks, so a model runs in FP16 only when its ONNX file is FP16. That is why a model set ships `<model>.fp16.onnx` beside each FP32 file. Building a TensorRT engine takes time; `--engine-cache DIR` keeps the built engines, so later runs reuse them. A cached engine belongs to its GPU and its TensorRT version.

## <img src="assets/chip-04.svg" height="22" alt=""> 4. Options

| option | default | what |
|---|---|---|
| `CARBON_WITH_ONNXRUNTIME` | ON | the ONNX Runtime backend (CPU) |
| `CARBON_WITH_TENSORRT` | OFF | the TensorRT backend (GPU); needs `TensorRT_ROOT` |
| `CARBON_WITH_CUDA` | OFF | the CUDA kernels: the crop rule, CTC decoding, the fused CTC head, the detector's threshold; with TensorRT, the batched GPU pipeline |
| `CARBON_WITH_PDFIUM` | ON | PDF input, through PDFium in helper processes |
| `CARBON_BUILD_CLI` | ON | `carbon-ocr` and `carbon-pdf` |
| `CARBON_BUILD_TESTS` | ON when this is the top-level project | the C++ tests |
| `CARBON_BUILD_PYTHON` | OFF | the Python module (`pip install ./python` turns it on) |
| `ONNXRUNTIME_ROOT` | empty: download | an unpacked ONNX Runtime release |
| `PDFIUM_ROOT` | empty: download | an unpacked PDFium build |
| `TensorRT_ROOT` | the environment variable | an unpacked TensorRT release |

| preset | what |
|---|---|
| `ninja-release`, `ninja-debug` | Ninja, CPU |
| `vs2022` | Windows, Visual Studio 2022, CPU |
| `ninja-trt`, `vs2022-trt` | CPU and GPU (TensorRT) |
| `ninja-cuda` | CPU and the CUDA kernels, without TensorRT |

To use the library from another CMake project, add this repository with `add_subdirectory` (or FetchContent) and link `carbon::carbon`. Its public headers are in `carbon/include/carbon/`, and `carbon/carbon.h` includes them all.

## <img src="assets/chip-05.svg" height="22" alt=""> 5. The Python module

```shell
pip install ./python                                        # ONNX Runtime, the CPU
pip install ./python -C cmake.define.CARBON_WITH_CUDA=ON    # and the CUDA kernels
```

scikit-build-core builds the module `carbon_ocr` from this CMake project, with ONNX Runtime and without the CLI or the C++ tests. ONNX Runtime's library is installed next to the module, which finds it there. TensorRT is not in the module yet, so `Pipeline(device="cuda")` needs a build of `carbon-ocr` instead (`carbon-ocr ocr --device cuda`). [python/README.md](../python/README.md) lists the module's API.

## <img src="assets/chip-06.svg" height="22" alt=""> 6. Tests

`ctest --preset <preset>` runs the C++ tests: the tensors, the crop rule, the CPU kernels against their scalar reference, the native backend against ONNX Runtime, the detector's and order model's decoders, the engine and the batched pipeline on small fake models, PDFs, the output formats and the command line. The tests that need a GPU are built with CUDA or TensorRT and run on a machine with one.

```shell
pip install ./python pytest==8.4.2 pillow==13.0.0
python -m pytest tests/python/test_carbon_ocr.py
```

The Python tests hold the module to the C++, and its crop rule to Pillow's bilinear resize, byte for byte. `tests/python/test_ocrmypdf_plugin.py` runs the OCRmyPDF plugin on a built `carbon-ocr` (`CARBON_OCR` names it). It needs the packages in `tools/ocrmypdf/requirements.txt`, and `tesseract` on `PATH` for whole OCRmyPDF runs.
