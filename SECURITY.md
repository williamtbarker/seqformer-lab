# Security

Please report vulnerabilities privately through GitHub's security-advisory interface.

SeqFormer Lab loads only PyTorch state dictionaries with `weights_only=True`. Treat datasets and
checkpoints from untrusted sources as untrusted files, keep dependencies updated, and inspect input
paths before processing them. The software is a research and educational tool, not a clinical,
diagnostic, surveillance, or production decision system.
