# models/

Fullerene model sets: `meta.json` and vocabularies in git; the ONNX weights are on Hugging Face (`CarbonResearchLabs/fullerene-3M`; `*.onnx` here is ignored).

A Fullerene model is one complete set: detector, recognizers, classifiers and order model, shipped and versioned together. When there is more than one, each is named `fullerene-<N>M`, where N is the set's total parameter count rounded to the nearest million, halves up (a 7.3 M set is `fullerene-7M`, a 7.5 M set `fullerene-8M`). Inside a set, files are named by role:

```
models/fullerene-7M/
  meta.json            versions, vocabularies, input sizes, thresholds
  orient.onnx          page orientation
  det.onnx             line detector
  script.onnx          line script classifier
  rec-latin.onnx       recognizers, one per script group (FP32, for CPUs)
  rec-latin.vocab.json          its classes, class 0 the CTC blank
  rec-latin.int8.onnx           the same in int8 (Q/DQ), for the CPU's native backend; every model has one
                                (order's run by ONNX Runtime and TensorRT)
  rec-latin.int8-sym.onnx       the same in symmetric int8, for TensorRT
  rec-latin.fp16.onnx           the same in FP16, for TensorRT
  rec-latin.features.fp16.onnx  its trunk: TensorRT runs it, then the fused CTC head
  rec-latin.head.bin            the head's weights, for that kernel
  rec-cjk.onnx
  ...
  order.onnx           reading order and layout
  order.native.bin     its weights for the engine's own CPU kernels
  lm-latin.bin         n-gram language models
```
