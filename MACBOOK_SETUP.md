# MacBook setup

SeqFormer Lab supports Python 3.10–3.12. The default commands deliberately use CPU execution so
that results are reproducible across Intel and Apple Silicon Macs.

```bash
cd ~/Documents/GPT_Hist_Review/seqformer-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
./scripts/verify.sh
```

Run the demonstration:

```bash
seqformer-lab demo --output artifacts/demo --epochs 2
cat artifacts/demo/metrics.json
cat artifacts/demo/manifest.json
```

Apple's MPS backend is available through `--device mps`, but CPU remains the reproducibility
baseline. MPS and CUDA can use kernels whose exact floating-point results differ by platform.
