"""Verify upstream versions, architecture, checkpoint and scratch semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import distribution, version

from .constants import MANTIS_GIT_COMMIT, MANTIS_VERSION
from .protocol import Protocol, build_backbone, build_head, resolve_device, set_seed


def state_hash(module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    installed = version("mantis-tsfm")
    if installed != MANTIS_VERSION:
        raise RuntimeError(
            f"Expected mantis-tsfm {MANTIS_VERSION}, found {installed}. Reinstall from pyproject.toml."
        )
    direct_url_text = distribution("mantis-tsfm").read_text("direct_url.json")
    if direct_url_text:
        direct_url = json.loads(direct_url_text)
        installed_commit = direct_url.get("vcs_info", {}).get("commit_id")
        if installed_commit and installed_commit != MANTIS_GIT_COMMIT:
            raise RuntimeError(
                f"Expected Mantis commit {MANTIS_GIT_COMMIT}, found {installed_commit}"
            )
    device = resolve_device(args.device)
    protocol = Protocol()
    set_seed(0)
    pretrained = build_backbone("full", device, protocol)
    set_seed(0)
    scratch = build_backbone("scratch", device, protocol)
    head = build_head(pretrained, n_channels=1, n_classes=3, device=device)
    if pretrained.__class__ is not scratch.__class__:
        raise AssertionError("Pretrained and scratch architectures differ")
    if state_hash(pretrained) == state_hash(scratch):
        raise AssertionError("Scratch and pretrained weights are unexpectedly identical")
    if [layer.__class__.__name__ for layer in head] != ["LayerNorm", "Linear"]:
        raise AssertionError("Prediction head is not LayerNorm + Linear")
    print("Environment verification passed")
    print(f"mantis-tsfm={installed}")
    print(f"mantis_git_commit={MANTIS_GIT_COMMIT}")
    print(f"architecture={pretrained.__class__.__name__}")
    print(f"checkpoint={protocol.checkpoint}@{protocol.checkpoint_revision}")
    print(f"pretrained_sha256={state_hash(pretrained)}")
    print(f"scratch_sha256={state_hash(scratch)}")


if __name__ == "__main__":
    main()
