# industrial-defect-detection

What: find surface defects in product images (visual inspection), comparing a classical
OpenCV baseline with a deep anomaly model, and serve it as a fast CPU service.
Target employers: Bosch, Siemens. Deadline: 23 Dec 2026.
Done when: per-category AUROC/F1 (baseline vs deep), ONNX on CPU with latency p50/p95 and
the hardware name, FastAPI + Docker. Optional: MQTT publish.

Data: MVTec AD (15 categories, train = good images only, test = good + defects with masks).
Licence CC BY-NC-SA 4.0 (non-commercial). Kaggle input: avdvhh/mvtec-defect-detection-dataset.
Tests use synthetic images; CI never downloads data. No data or weights in git.
Never Fraunhofer or BTU data or code.

Package: src/idd/   Tests: tests/   Configs: configs/
Commands:
- install: pip install -e ".[dev]"
- test:    pytest
- lint:    ruff check .

Milestones (tick them in the 365 Build Log):
1. MVTec data loader (train/good, test/<defect>, masks); tests on synthetic folders
2. Classical OpenCV baseline: per-category image AUROC + F1
3. Deep model: PatchCore-lite (pretrained CNN features + nearest neighbour), same table
4. ONNX export + ONNX Runtime CPU latency p50/p95, CPU name, PyTorch vs ONNX
5. FastAPI /inspect + Dockerfile + pinned requirements + Docker build in CI
6. README: results, latency, heatmap examples, limitations. Optional: MQTT publish

Current status: milestone 2 done. Baseline on 15 MVTec categories: mean AUROC 0.740 (bottle 0.977 ... screw 0.454); F1-max mostly at the "all defect" floor. Milestone 3 code done (idd.patchcore, fake-backbone tests). Next: PatchCore run on Kaggle GPU, 15 categories.
