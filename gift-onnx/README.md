# GIFT ONNX Flight Lab

Independent deployment experiment: https://my-85.github.io/gift-onnx/

The original gift/ website and GPU service remain unchanged.
Archived replays retain their original model identities; online custom flights
use ONNX Runtime FP32 with the frozen 9a baseline and original A4 CPU dynamics.
Start Flight opens the separate owner-authorized Site connection window.
Keep it open until the result is returned. No service token or model weights
are included in this public folder.

On one complete CPU closed-loop test, all 1,024 actions and 1,025 physical
samples match the original JAX CPU baseline exactly. CPU and GPU dynamics
produce different long-route trajectories; this is an experimental deployment,
not a claim of equivalence to the original GPU website. See engine.json.
