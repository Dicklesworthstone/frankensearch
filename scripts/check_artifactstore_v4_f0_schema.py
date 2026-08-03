#!/usr/bin/env python3
"""Validate frozen ArtifactStore v4 F0 schema truth-table vectors.

Usage: scripts/check_artifactstore_v4_f0_schema.py

This is a read-only contract check. It validates only structural JSON-Schema
rules; signature bytes, canonical serialization, nonce storage, predecessor
existence, and independent policy coverage remain implementation checks.
"""

from __future__ import annotations

import json
import sys
from json import JSONDecodeError
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schemas" / "artifactstore-v4-f0.schema.json"
VECTORS_PATH = ROOT / "schemas" / "artifactstore-v4-f0.schema-test-vectors.json"
JSON_DECODER = json.JSONDecoder()


def load_json(path: Path) -> object:
    try:
        decoded = JSON_DECODER.decode(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot load {path.name}: {error}") from error
    except JSONDecodeError as error:
        raise ValueError(f"cannot load {path.name}: {error}") from error
    return decoded


def main() -> int:
    try:
        schema = load_json(SCHEMA_PATH)
        vectors = load_json(VECTORS_PATH)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        print(f"invalid schema: {error.message}", file=sys.stderr)
        return 2
    if not isinstance(vectors, dict) or vectors.get("schema") != SCHEMA_PATH.name:
        print("vectors do not name the checked schema", file=sys.stderr)
        return 2
    cases = vectors.get("cases")
    if not isinstance(cases, list) or not cases:
        print("vectors must contain at least one case", file=sys.stderr)
        return 2

    validator = Draft202012Validator(schema)
    failures: list[str] = []
    for case in cases:
        if not isinstance(case, dict):
            failures.append("non-object case")
            continue
        name = case.get("name")
        expected = case.get("expect_valid")
        instance = case.get("instance")
        if not isinstance(name, str) or not isinstance(expected, bool) or not isinstance(instance, dict):
            failures.append("case has invalid name, expectation, or instance")
            continue
        errors = list(validator.iter_errors(instance))
        actual = not errors
        if actual != expected:
            detail = "accepted" if actual else errors[0].message
            failures.append(f"{name}: expected valid={expected}, got {detail}")

    if failures:
        print("ArtifactStore v4 F0 schema conformance failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"ArtifactStore v4 F0 schema conformance passed: {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
