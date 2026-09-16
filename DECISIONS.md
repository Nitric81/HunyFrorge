# HunyForge Decisions

## D001 — Local-first

**Date:** 2026-09-11  
Inference, project data, and artifacts remain local by default. This is a product differentiator and a hard privacy constraint.

## D002 — Staged models

**Date:** 2026-09-11  
Shape and texture models are not assumed resident together because the documented combined requirement exceeds 16 GB VRAM. Use explicit unload/offload and disk-backed intermediates.

## D003 — Unity is a release gate

**Date:** 2026-09-11  
Unity preparation and validation are MVP requirements because the product goal is game-asset production, not merely mesh generation.

