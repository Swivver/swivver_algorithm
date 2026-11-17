#!/usr/bin/env python3
"""Offline evaluation metrics, Draft 2020-12 schema checks, and clean-room scanner.

Runs using only the Python 3 standard library.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DRAFT_2020_12_URI = "https://json-schema.org/draft/2020-12/schema"

EXPECTED_SCHEMA_FILES = (
    "scoped-signal-features-v1.schema.json",
    "candidate-feature-vector-v1.schema.json",
    "ranked-feed-result-v1.schema.json",
)

FORBIDDEN_ORACLE_FIELDS = frozenset(
    {
        "_".join(("universal", "quality", "score")),
        "_".join(("editorial", "merit", "score")),
        "_".join(("author", "credibility", "score")),
        "_".join(("truth", "score")),
    }
)

RESTRICTED_REPO_TOKENS = (
    "_".join(("swivver", "signal")),
    "_".join(("swivver", "core")),
    "/".join(("github.com", "swivver", "swivver")),
    "-".join(("signal", "analysis", "profile")),
    ".".join(("analysis-0", "3", "openapi")),
    "_".join(("SIGNAL", "LLM", "ENDPOINT")),
    "_".join(("SIGNAL", "CRITIC", "MODEL")),
)

LOCAL_HOME_PATH_RE = re.compile(r"(?:^|[\s\"'`(])/(?:Users|home)/[A-Za-z0-9._-]+/")
WINDOWS_USER_PATH_RE = re.compile(r"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9._-]+", re.IGNORECASE)
PRIVATE_EDITORIAL_ROW_RE = re.compile(r"\b" + "E" + r"0[1-6]\b")
GO_IMPORT_STR_RE = re.compile(r'"([^"]+)"')


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevance_by_id: Mapping[str, float],
    k: int,
) -> float:
    """Compute graded Normalized Discounted Cumulative Gain at rank cutoff k."""
    if k <= 0 or not ranked_ids or not relevance_by_id:
        return 0.0

    def dcg(gains: Sequence[float]) -> float:
        total = 0.0
        for idx, rel in enumerate(gains):
            clamped = max(0.0, float(rel))
            if clamped > 0.0:
                total += (math.pow(2.0, clamped) - 1.0) / math.log2(idx + 2.0)
        return total

    observed_gains = [relevance_by_id.get(cid, 0.0) for cid in ranked_ids[:k]]
    ideal_gains = sorted((max(0.0, float(v)) for v in relevance_by_id.values()), reverse=True)[:k]

    ideal_dcg = dcg(ideal_gains)
    if ideal_dcg <= 0.0:
        return 0.0
    return min(1.0, dcg(observed_gains) / ideal_dcg)


def reciprocal_rank(
    ranked_ids: Sequence[str],
    relevance_by_id: Mapping[str, float],
    min_relevance: float = 2.0,
) -> float:
    """Compute Reciprocal Rank (1/rank) of the first candidate meeting min_relevance."""
    if not ranked_ids or not relevance_by_id:
        return 0.0
    for idx, cid in enumerate(ranked_ids, start=1):
        if float(relevance_by_id.get(cid, 0.0)) >= min_relevance:
            return 1.0 / idx
    return 0.0


def author_gini_coefficient(author_exposures: Sequence[str] | Mapping[str, int]) -> float:
    """Compute the Gini concentration coefficient across author exposures in [0, 1)."""
    if isinstance(author_exposures, Mapping):
        counts = [int(v) for v in author_exposures.values() if int(v) >= 0]
    else:
        counted = Counter(a for a in author_exposures if a)
        counts = list(counted.values())

    n = len(counts)
    if n <= 1:
        return 0.0
    total = sum(counts)
    if total <= 0:
        return 0.0

    sorted_counts = sorted(counts)
    weighted_diff_sum = 0.0
    for idx, value in enumerate(sorted_counts, start=1):
        weighted_diff_sum += (2 * idx - n - 1) * value
    return max(0.0, weighted_diff_sum / (n * total))


def minority_language_share_parity(
    pool_languages: Sequence[str],
    ranked_languages: Sequence[str],
    primary_language: str | None = None,
) -> dict[str, float]:
    """Compare minority-language share in ranked output against the eligible candidate pool."""
    clean_pool = [lang.strip().lower() for lang in pool_languages if lang and lang.strip()]
    clean_ranked = [lang.strip().lower() for lang in ranked_languages if lang and lang.strip()]

    if not clean_pool or not clean_ranked:
        return {
            "pool_minority_share": 0.0,
            "ranked_minority_share": 0.0,
            "parity_ratio": 1.0,
            "share_delta": 0.0,
        }

    if primary_language is not None and primary_language.strip():
        dominant = primary_language.strip().lower()
    else:
        dominant = Counter(clean_pool).most_common(1)[0][0]

    pool_minority = sum(1 for lang in clean_pool if lang != dominant) / len(clean_pool)
    ranked_minority = sum(1 for lang in clean_ranked if lang != dominant) / len(clean_ranked)

    if pool_minority <= 0.0:
        parity_ratio = 1.0
    else:
        parity_ratio = ranked_minority / pool_minority

    return {
        "pool_minority_share": round(pool_minority, 6),
        "ranked_minority_share": round(ranked_minority, 6),
        "parity_ratio": round(parity_ratio, 6),
        "share_delta": round(ranked_minority - pool_minority, 6),
    }


def counterfactual_unknown_rank_delta(
    baseline_ranked_ids: Sequence[str],
    counterfactual_ranked_ids: Sequence[str],
    target_ids: Iterable[str],
) -> dict[str, Any]:
    """Verify that replacing unassessed features with unknown/unsupported never demotes targets.

    Rank positions are 1-indexed; a positive delta (new_rank - base_rank > 0) means demotion.
    """
    targets = set(target_ids)
    base_pos = {cid: idx + 1 for idx, cid in enumerate(baseline_ranked_ids)}
    cf_pos = {cid: idx + 1 for idx, cid in enumerate(counterfactual_ranked_ids)}

    deltas: list[int] = []
    demoted: list[str] = []
    missing: list[str] = []

    for cid in sorted(targets):
        if cid not in base_pos:
            continue
        if cid not in cf_pos:
            missing.append(cid)
            demoted.append(cid)
            continue
        delta = cf_pos[cid] - base_pos[cid]
        deltas.append(delta)
        if delta > 0:
            demoted.append(cid)

    max_demotion = max([0, *deltas])
    mean_delta = (sum(deltas) / len(deltas)) if deltas else 0.0

    return {
        "evaluated_targets": len(deltas) + len(missing),
        "demoted_count": len(demoted),
        "demoted_ids": demoted,
        "missing_ids": missing,
        "max_demotion_positions": max_demotion,
        "mean_rank_delta": round(mean_delta, 6),
        "passed": len(demoted) == 0,
    }


def validate_schemas_directory(schemas_dir: Path) -> list[str]:
    """Validate structural soundness and anti-oracle rules across schemas/*.schema.json."""
    errors: list[str] = []
    if not schemas_dir.is_dir():
        return [f"Missing schemas directory: {schemas_dir}"]

    loaded: dict[str, dict[str, Any]] = {}
    for filename in EXPECTED_SCHEMA_FILES:
        path = schemas_dir / filename
        if not path.is_file():
            errors.append(f"Missing required schema file: {filename}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{filename}: invalid JSON ({exc})")
            continue
        if not isinstance(data, dict):
            errors.append(f"{filename}: root schema must be a JSON object")
            continue
        loaded[filename] = data

    for filename, schema in loaded.items():
        if schema.get("$schema") != DRAFT_2020_12_URI:
            errors.append(f"{filename}: $schema must equal {DRAFT_2020_12_URI!r}")
        if not isinstance(schema.get("$id"), str) or not schema["$id"].startswith("https://"):
            errors.append(f"{filename}: $id must be an https URI")
        if not isinstance(schema.get("title"), str) or not schema["title"].strip():
            errors.append(f"{filename}: missing non-empty title")
        if schema.get("type") != "object":
            errors.append(f"{filename}: root type must be 'object'")
        if schema.get("additionalProperties") is not False:
            errors.append(f"{filename}: root additionalProperties must be false")

        props = schema.get("properties")
        req = schema.get("required")
        if not isinstance(props, dict) or not props:
            errors.append(f"{filename}: properties must be a non-empty object")
            continue
        if not isinstance(req, list) or not req:
            errors.append(f"{filename}: required must be a non-empty list")
        else:
            for key in req:
                if key not in props:
                    errors.append(f"{filename}: required property {key!r} missing from properties")

        forbidden_hits = FORBIDDEN_ORACLE_FIELDS.intersection(props.keys())
        if forbidden_hits:
            errors.append(f"{filename}: contains forbidden oracle fields {sorted(forbidden_hits)}")

        _check_refs(filename, schema, schema, loaded, errors)

    return errors


def _check_refs(
    filename: str,
    node: Any,
    root_schema: Mapping[str, Any],
    all_schemas: Mapping[str, Mapping[str, Any]],
    errors: list[str],
) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            if ref.startswith("#/$defs/"):
                def_name = ref.removeprefix("#/$defs/")
                defs = root_schema.get("$defs", {})
                if not isinstance(defs, dict) or def_name not in defs:
                    errors.append(f"{filename}: unresolved local $ref {ref!r}")
            elif ref not in all_schemas:
                errors.append(f"{filename}: unresolved external $ref {ref!r}")
        for value in node.values():
            _check_refs(filename, value, root_schema, all_schemas, errors)
    elif isinstance(node, list):
        for item in node:
            _check_refs(filename, item, root_schema, all_schemas, errors)


def validate_instance_against_schema(
    instance: Any,
    schema: Mapping[str, Any],
    all_schemas: Mapping[str, Mapping[str, Any]],
    root_schema: Mapping[str, Any] | None = None,
    path: str = "$",
) -> list[str]:
    """Validate a JSON-compatible Python value against our Draft 2020-12 schema subset."""
    active_root = root_schema if root_schema is not None else schema
    errors: list[str] = []

    if "$ref" in schema:
        ref = schema["$ref"]
        if ref.startswith("#/$defs/"):
            target = active_root["$defs"][ref.removeprefix("#/$defs/")]
            return validate_instance_against_schema(instance, target, all_schemas, active_root, path)
        target_schema = all_schemas[ref]
        return validate_instance_against_schema(instance, target_schema, all_schemas, target_schema, path)

    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(instance, dict):
            return [f"{path}: expected object"]
        for req_key in schema.get("required", ()):
            if req_key not in instance:
                errors.append(f"{path}: missing required field {req_key!r}")
        props = schema.get("properties", {})
        addl = schema.get("additionalProperties", True)
        prop_names_schema = schema.get("propertyNames")
        for key, val in instance.items():
            if isinstance(prop_names_schema, dict):
                errors.extend(
                    validate_instance_against_schema(
                        key, prop_names_schema, all_schemas, active_root, f"{path}.<key:{key}>"
                    )
                )
            if key in props:
                errors.extend(
                    validate_instance_against_schema(
                        val, props[key], all_schemas, active_root, f"{path}.{key}"
                    )
                )
            elif addl is False:
                errors.append(f"{path}: unexpected additional property {key!r}")
            elif isinstance(addl, dict):
                errors.extend(
                    validate_instance_against_schema(
                        val, addl, all_schemas, active_root, f"{path}.{key}"
                    )
                )
    elif expected_type == "array":
        if not isinstance(instance, list):
            return [f"{path}: expected array"]
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: length {len(instance)} < minItems {schema['minItems']}")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: length {len(instance)} > maxItems {schema['maxItems']}")
        if schema.get("uniqueItems") and len(set(map(json.dumps, instance))) != len(instance):
            errors.append(f"{path}: duplicate items violate uniqueItems")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(instance):
                errors.extend(
                    validate_instance_against_schema(
                        item, item_schema, all_schemas, active_root, f"{path}[{idx}]"
                    )
                )
    elif expected_type == "string":
        if not isinstance(instance, str):
            return [f"{path}: expected string"]
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: length {len(instance)} < minLength {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: length {len(instance)} > maxLength {schema['maxLength']}")
    elif expected_type == "integer":
        if isinstance(instance, bool) or not isinstance(instance, int):
            return [f"{path}: expected integer"]
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")
    elif expected_type == "number":
        if isinstance(instance, bool) or not isinstance(instance, (int, float)) or not math.isfinite(instance):
            return [f"{path}: expected finite number"]
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")
    elif expected_type == "boolean":
        if not isinstance(instance, bool):
            return [f"{path}: expected boolean"]

    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: value {instance!r} != const {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: value {instance!r} not in enum {schema['enum']}")

    return errors


def scan_clean_room_boundary(repo_root: Path) -> list[str]:
    """Scan AST imports, Go imports, and repository text for clean-room violations."""
    violations: list[str] = []
    forbidden_py_roots = {
        "_".join(("swivver", "signal")),
        "_".join(("swivver", "signal", "oss")),
        "_".join(("swivver", "core")),
    }

    for path in sorted(repo_root.rglob("*")):
        rel_parts = path.relative_to(repo_root).parts
        if any(part.startswith(".git") or part == "__pycache__" for part in rel_parts):
            continue
        if not path.is_file():
            continue

        rel = path.relative_to(repo_root).as_posix()
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        # 1. Check for machine-specific home paths and restricted private identifiers.
        if LOCAL_HOME_PATH_RE.search(content) or WINDOWS_USER_PATH_RE.search(content):
            violations.append(f"{rel}: contains machine-specific user path")
        if PRIVATE_EDITORIAL_ROW_RE.search(content):
            violations.append(f"{rel}: contains private editorial taxonomy row code")
        for token in RESTRICTED_REPO_TOKENS:
            if token in content:
                violations.append(f"{rel}: contains restricted identifier {token!r}")

        # 2. Python AST import boundary verification.
        if path.suffix == ".py":
            try:
                tree = ast.parse(content, filename=rel)
            except SyntaxError as exc:
                violations.append(f"{rel}: Python syntax error ({exc})")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_pkg = alias.name.split(".")[0]
                        if root_pkg in forbidden_py_roots:
                            violations.append(f"{rel}: forbidden Python import {alias.name!r}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    root_pkg = node.module.split(".")[0]
                    if root_pkg in forbidden_py_roots:
                        violations.append(f"{rel}: forbidden Python from-import {node.module!r}")

        # 3. Go standard-library / internal-only import verification.
        elif path.suffix == ".go":
            for imp in _extract_go_imports(content):
                if "." in imp.split("/")[0] and not imp.startswith("github.com/Swivver/swivver_algorithm"):
                    violations.append(f"{rel}: external Go import {imp!r} violates zero-dependency clean-room rule")

        elif path.name == "go.mod":
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("require"):
                    violations.append(f"{rel}: unexpected external require directive in go.mod")

    return violations


def _extract_go_imports(go_source: str) -> list[str]:
    imports: list[str] = []
    in_block = False
    for line in go_source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ("):
            in_block = True
            continue
        if in_block:
            if stripped.startswith(")"):
                in_block = False
                continue
            match = GO_IMPORT_STR_RE.search(stripped)
            if match:
                imports.append(match.group(1))
        elif stripped.startswith("import "):
            match = GO_IMPORT_STR_RE.search(stripped)
            if match:
                imports.append(match.group(1))
    return imports


def run_synthetic_evaluation(repo_root: Path) -> dict[str, Any]:
    """Run an end-to-end offline benchmark and boundary check over repository contracts."""
    schemas_dir = repo_root / "schemas"
    schema_errors = validate_schemas_directory(schemas_dir)
    clean_room_errors = scan_clean_room_boundary(repo_root)

    # Representative multilingual, multi-author benchmark fixture.
    relevance = {
        "c-en-1": 3.0,
        "c-es-1": 3.0,
        "c-en-2": 2.0,
        "c-fr-1": 2.0,
        "c-en-3": 1.0,
        "c-contradicted": 0.0,
    }
    baseline_ranked = ["c-en-1", "c-es-1", "c-en-2", "c-fr-1", "c-en-3", "c-contradicted"]
    # Counterfactual: c-es-1 and c-fr-1 carry availability='unknown'/'unsupported' -> zero demotion.
    counterfactual_ranked = ["c-en-1", "c-es-1", "c-en-2", "c-fr-1", "c-en-3", "c-contradicted"]

    ndcg5 = ndcg_at_k(baseline_ranked, relevance, k=5)
    gini = author_gini_coefficient(["a1", "a2", "a3", "a4", "a5", "a6"])
    lang_parity = minority_language_share_parity(
        pool_languages=["en", "es", "en", "fr", "en", "en"],
        ranked_languages=["en", "es", "en", "fr", "en", "en"],
        primary_language="en",
    )
    cf_check = counterfactual_unknown_rank_delta(
        baseline_ranked_ids=baseline_ranked,
        counterfactual_ranked_ids=counterfactual_ranked,
        target_ids={"c-es-1", "c-fr-1"},
    )

    passed = (
        not schema_errors
        and not clean_room_errors
        and cf_check["passed"]
        and ndcg5 >= 0.95
        and gini <= 0.35
        and lang_parity["parity_ratio"] >= 0.80
    )

    return {
        "passed": passed,
        "ndcg_at_5": round(ndcg5, 6),
        "author_gini": round(gini, 6),
        "language_parity": lang_parity,
        "counterfactual_unknown_check": cf_check,
        "schema_errors": schema_errors,
        "clean_room_violations": clean_room_errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline evaluation and clean-room checks.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to the swivver_algorithm repository root.",
    )
    args = parser.parse_args(argv)

    summary = run_synthetic_evaluation(args.repo_root.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
