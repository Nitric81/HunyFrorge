# HunyForge Verification Ledger

Updated: 2026-09-11

## Proven in the current environment

- `npm run lint` — pass
- `npm test` — pass; 2 frontend utility tests
- `npm run build` — pass; Vite emits a Three.js bundle-size warning
- `python -m compileall backend` — pass
- `python -m unittest backend.test_pipeline backend.test_api` — pass; 15 tests
- `scripts/verify-demo.ps1` against a live API — pass; project, demo job, 8 artifacts, Unity validation
- `scripts/check-runtime.ps1` against a live API — pass; correctly reports demo mode, CPU-only runtime, and missing model root
- Supplied model snapshot inspection — pass; shape, VAE, and PBR checkpoint files are present under `D:\AI_Models\Hunyuan3D-2.1`; official source checkout is present under `D:\AI_Models\Hunyuan3D-2.1-src`.
- `scripts/start-hunyuan-service.ps1` — PowerShell syntax pass; uses the official 2.1 `--low_vram_mode` launch flag
- Browser smoke evidence — Three.js preview rendered and generated GLB returned HTTP 200

## Required target-machine gates

- Install and execute Hunyuan3D 2.1 with CUDA on the RTX 4080 16 GB system.
- Measure shape and texture peak VRAM separately and confirm no combined model residency.
- Execute the configured Hunyuan3D-Omni wrapper for each control type: `point`, `voxel`, `pose`, `bbox`.
- Confirm any Omni shape-to-PBR handoff on the target setup.
- Import the generated Unity package in Unity Editor and verify materials, LODs, collision, scale, and axis orientation.

Do not mark these gates verified from demo output or CPU-only tests.
