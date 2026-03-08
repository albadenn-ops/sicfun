#!/usr/bin/env python3
"""
Generate a tiny DDRE ONNX smoke model.

Model contract:
  inputs:
    - prior:       float[1, H]
    - likelihoods: float[O, H]   (currently unused in the graph, retained for provider contract parity)
  output:
    - posterior:   float[1, H]

Computation:
  posterior = sqrt(prior)

The runtime side normalizes probabilities after model output, so this is sufficient
for exercising a non-trivial ONNX path in integration tests.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import onnx
from onnx import TensorProto, checker, helper


def build_model() -> onnx.ModelProto:
    prior = helper.make_tensor_value_info("prior", TensorProto.FLOAT, [1, "H"])
    likelihoods = helper.make_tensor_value_info("likelihoods", TensorProto.FLOAT, ["O", "H"])
    posterior = helper.make_tensor_value_info("posterior", TensorProto.FLOAT, [1, "H"])

    sqrt_node = helper.make_node("Sqrt", inputs=["prior"], outputs=["posterior"], name="sqrt_prior")

    graph = helper.make_graph(
        nodes=[sqrt_node],
        name="ddre_smoke_sqrt_graph",
        inputs=[prior, likelihoods],
        outputs=[posterior],
    )

    model = helper.make_model(
        graph,
        producer_name="sicfun-ddre-smoke-generator",
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=9,
    )
    checker.check_model(model)
    return model


def write_artifact_metadata(artifact_dir: Path, model_filename: str) -> None:
    metadata_path = artifact_dir / "metadata.properties"
    lines = [
        "format.version=1",
        "artifact.kind=onnx",
        "artifact.id=ddre-smoke-sqrt",
        "artifact.source=scripts/generate-ddre-smoke-onnx.py",
        f"artifact.createdAtEpochMillis={int(time.time() * 1000)}",
        "artifact.notes=Experimental smoke artifact. Not validated for decision driving.",
        f"model.file={model_filename}",
        "model.input.prior=prior",
        "model.input.likelihoods=likelihoods",
        "model.output.posterior=posterior",
        "runtime.executionProvider=cpu",
        "runtime.cudaDevice=0",
        "runtime.intraOpThreads=",
        "runtime.interOpThreads=",
        "validation.status=experimental",
        "validation.decisionDrivingAllowed=false",
        "validation.sampleCount=",
        "validation.meanNll=",
        "validation.meanKlVsBayes=",
        "validation.blockerViolationRate=",
        "validation.failureRate=",
        "validation.p50LatencyMillis=",
        "validation.p95LatencyMillis=",
        "gate.minSamples=",
        "gate.maxMeanNll=",
        "gate.maxMeanKlVsBayes=",
        "gate.maxBlockerViolationRate=",
        "gate.maxFailureRate=",
        "gate.maxP95LatencyMillis=",
    ]
    metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DDRE smoke ONNX model")
    parser.add_argument(
        "--out",
        default="src/test/resources/sicfun/ddre/ddre-smoke-sqrt.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--artifact-dir",
        default="",
        help="Optional DDRE artifact directory. Writes metadata.properties next to the model.",
    )
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model()
    onnx.save(model, out_path.as_posix())
    print(f"wrote {out_path}")

    if args.artifact_dir:
        artifact_dir = Path(args.artifact_dir).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_model = artifact_dir / out_path.name
        artifact_model.write_bytes(out_path.read_bytes())
        write_artifact_metadata(artifact_dir, artifact_model.name)
        print(f"wrote {artifact_dir / 'metadata.properties'}")


if __name__ == "__main__":
    main()
