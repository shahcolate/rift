"""Suite adapters: promptfoo / Inspect / lm-eval / OpenAI-evals importers.

Every test round-trips the emitted dict through a YAML file and
``load_suite`` — an adapter that emits something the real loader rejects
is broken no matter how nice its dict looks.
"""

from __future__ import annotations

import json
import textwrap

import pytest
import yaml

from rift.adapters import SuiteImportError, convert
from rift.adapters import scorers as bundled
from rift.config import load_suite


def _roundtrip(tmp_path, imported):
    out = tmp_path / f"suite_{imported.variant or 'main'}.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(imported.suite, f, sort_keys=False)
    return load_suite(str(out))


# ---------------------------------------------------------------------------
# Bundled scorers
# ---------------------------------------------------------------------------

def test_bundled_scorers_binary_contract():
    assert bundled.contains("the answer is 42", "42") == 1.0
    assert bundled.contains("nope", "42") == 0.0
    assert bundled.icontains("PARIS is the capital", "paris") == 1.0
    assert bundled.starts_with("  Paris, France", "Paris") == 1.0
    assert bundled.starts_with("in Paris", "Paris") == 0.0
    assert bundled.regex_match("total: $1,240.00", r"\$1,240") == 1.0
    assert bundled.regex_match("total: 12", r"\$\d+") == 0.0
    assert bundled.contains_any("cat dog", ["bird", "dog"]) == 1.0
    assert bundled.contains_any("cat", ["bird", "dog"]) == 0.0
    assert bundled.contains_all("cat dog", ["cat", "dog"]) == 1.0
    assert bundled.contains_all("cat", ["cat", "dog"]) == 0.0


def test_bundled_scorers_coerce_non_strings():
    # Foreign formats routinely carry numeric expected values.
    assert bundled.contains("answer: 4", 4) == 1.0
    assert bundled.icontains(42, "4") == 1.0


# ---------------------------------------------------------------------------
# promptfoo
# ---------------------------------------------------------------------------

def _write_promptfoo(tmp_path, body: str):
    p = tmp_path / "promptfooconfig.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_promptfoo_equals_maps_to_exact_match(tmp_path):
    src = _write_promptfoo(tmp_path, """
        prompts:
          - "Translate: {{word}}"
        tests:
          - vars: {word: cat}
            assert: [{type: equals, value: chat}]
          - vars: {word: dog}
            assert: [{type: equals, value: chien}]
    """)
    (imported,) = convert("promptfoo", src)
    suite = _roundtrip(tmp_path, imported)
    assert suite.scoring == "exact_match"
    assert suite.cases[0].input == "Translate: cat"
    assert suite.cases[0].expected == "chat"
    assert len(suite.cases) == 2


def test_promptfoo_contains_uses_bundled_scorer(tmp_path):
    src = _write_promptfoo(tmp_path, """
        prompts: ["Say {{x}}"]
        tests:
          - vars: {x: hi}
            assert: [{type: contains, value: hi}]
    """)
    (imported,) = convert("promptfoo", src)
    suite = _roundtrip(tmp_path, imported)
    assert suite.scoring == "custom"
    assert suite.custom_scorer == "rift.adapters.scorers:contains"


def test_promptfoo_llm_rubric_maps_to_llm_judge_rubric(tmp_path):
    src = _write_promptfoo(tmp_path, """
        prompts: ["Explain {{x}}"]
        tests:
          - vars: {x: rain}
            assert: [{type: llm-rubric, value: mentions condensation}]
    """)
    (imported,) = convert("promptfoo", src)
    suite = _roundtrip(tmp_path, imported)
    assert suite.scoring == "llm_judge"
    assert suite.cases[0].expected == {"rubric": "mentions condensation"}


def test_promptfoo_mixed_asserts_error_without_split(tmp_path):
    src = _write_promptfoo(tmp_path, """
        prompts: ["{{x}}"]
        tests:
          - vars: {x: a}
            assert: [{type: equals, value: a}]
          - vars: {x: b}
            assert: [{type: contains, value: b}]
    """)
    with pytest.raises(SuiteImportError, match="--split-by-assert"):
        convert("promptfoo", src)


def test_promptfoo_split_by_assert_emits_one_suite_per_method(tmp_path):
    src = _write_promptfoo(tmp_path, """
        prompts: ["{{x}}"]
        tests:
          - vars: {x: a}
            assert: [{type: equals, value: a}]
          - vars: {x: b}
            assert: [{type: contains, value: b}]
    """)
    results = convert("promptfoo", src, split_by_assert=True)
    assert len(results) == 2
    variants = {r.variant for r in results}
    assert variants == {"exact_match", "contains"}
    for r in results:
        assert len(r.suite["cases"]) == 1
        _roundtrip(tmp_path, r)


def test_promptfoo_test_assert_beats_default_assert(tmp_path):
    src = _write_promptfoo(tmp_path, """
        prompts: ["{{x}}"]
        defaultTest:
          assert: [{type: icontains, value: fallback}]
        tests:
          - vars: {x: a}
            assert: [{type: equals, value: exact}]
    """)
    (imported,) = convert("promptfoo", src)
    assert imported.suite["scoring"] == "exact_match"
    assert imported.suite["cases"][0]["expected"] == "exact"


def test_promptfoo_default_assert_fills_assertless_test(tmp_path):
    src = _write_promptfoo(tmp_path, """
        prompts: ["{{x}}"]
        defaultTest:
          assert: [{type: icontains, value: shared}]
        tests:
          - vars: {x: a}
    """)
    (imported,) = convert("promptfoo", src)
    assert imported.suite["custom_scorer"] == "rift.adapters.scorers:icontains"
    assert imported.suite["cases"][0]["expected"] == "shared"


def test_promptfoo_unsupported_assert_dropped_with_warning(tmp_path):
    src = _write_promptfoo(tmp_path, """
        prompts: ["{{x}}"]
        tests:
          - vars: {x: a}
            assert:
              - {type: javascript, value: "output.length > 1"}
              - {type: equals, value: a}
    """)
    (imported,) = convert("promptfoo", src)
    assert any("javascript" in w for w in imported.warnings)
    assert imported.suite["scoring"] == "exact_match"


def test_promptfoo_template_blocks_refused(tmp_path):
    src = _write_promptfoo(tmp_path, """
        prompts: ["{% for x in items %}{{x}}{% endfor %}"]
        tests:
          - vars: {items: [a]}
            assert: [{type: equals, value: a}]
    """)
    with pytest.raises(SuiteImportError, match="blocks"):
        convert("promptfoo", src)


def test_promptfoo_undefined_variable_refused(tmp_path):
    src = _write_promptfoo(tmp_path, """
        prompts: ["Say {{missing}}"]
        tests:
          - vars: {present: a}
            assert: [{type: equals, value: a}]
    """)
    with pytest.raises(SuiteImportError, match="missing"):
        convert("promptfoo", src)


def test_promptfoo_caveats_recorded_in_description(tmp_path):
    src = _write_promptfoo(tmp_path, """
        prompts: ["{{x}}"]
        tests:
          - vars: {x: a}
            assert:
              - {type: equals, value: a}
              - {type: contains, value: a}
    """)
    (imported,) = convert("promptfoo", src)
    assert "Import caveats" in imported.suite["description"]
    assert "extra assert" in imported.suite["description"]


# ---------------------------------------------------------------------------
# Inspect AI
# ---------------------------------------------------------------------------

def _write_jsonl(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_inspect_basic_samples(tmp_path):
    src = _write_jsonl(tmp_path, "samples.jsonl", [
        {"input": "2+2?", "target": "4", "metadata": {"topic": "math"}, "id": 7},
    ])
    imported = convert("inspect", src)[0]
    suite = _roundtrip(tmp_path, imported)
    assert suite.scoring == "exact_match"
    assert suite.cases[0].expected == "4"
    assert "id:7" in suite.cases[0].tags
    assert "topic:math" in suite.cases[0].tags


def test_inspect_chat_input_flattened_with_warning(tmp_path):
    src = _write_jsonl(tmp_path, "samples.jsonl", [
        {"input": [{"role": "system", "content": "Be terse."},
                   {"role": "user", "content": "Capital of France?"}],
         "target": "Paris"},
    ])
    imported = convert("inspect", src)[0]
    case = imported.suite["cases"][0]
    assert "[system]" in case["input"]
    assert "Capital of France?" in case["input"]
    assert any("flattened" in w for w in imported.warnings)


def test_inspect_choices_rendered_as_options(tmp_path):
    src = _write_jsonl(tmp_path, "samples.jsonl", [
        {"input": "Pick the prime.", "choices": ["4", "7", "9"], "target": "B"},
    ])
    imported = convert("inspect", src)[0]
    case = imported.suite["cases"][0]
    assert "B) 7" in case["input"]
    assert case["expected"] == "B"


def test_inspect_multi_target_keeps_first_with_warning(tmp_path):
    src = _write_jsonl(tmp_path, "samples.jsonl", [
        {"input": "Q", "target": ["Paris", "paris"]},
    ])
    imported = convert("inspect", src)[0]
    assert imported.suite["cases"][0]["expected"] == "Paris"
    assert any("first was kept" in w for w in imported.warnings)


def test_inspect_scoring_override(tmp_path):
    src = _write_jsonl(tmp_path, "samples.jsonl", [{"input": "Q", "target": "A"}])
    imported = convert("inspect", src, scoring="fuzzy_match")[0]
    assert imported.suite["scoring"] == "fuzzy_match"


# ---------------------------------------------------------------------------
# lm-eval
# ---------------------------------------------------------------------------

def test_lmeval_generate_until(tmp_path):
    task = tmp_path / "task.yaml"
    task.write_text(textwrap.dedent("""
        task: demo_qa
        output_type: generate_until
        doc_to_text: "Q: {{question}}\\nA:"
        doc_to_target: "{{answer}}"
    """))
    docs = _write_jsonl(tmp_path, "docs.jsonl", [
        {"question": "Largest planet?", "answer": "Jupiter"},
    ])
    imported = convert("lm-eval", task, dataset=docs)[0]
    suite = _roundtrip(tmp_path, imported)
    assert suite.name == "demo_qa"
    assert suite.cases[0].input == "Q: Largest planet?\nA:"
    assert suite.cases[0].expected == "Jupiter"


def test_lmeval_doc_field_reference_target(tmp_path):
    task = tmp_path / "task.yaml"
    task.write_text("doc_to_text: 'Q: {{q}}'\ndoc_to_target: a\n")
    docs = _write_jsonl(tmp_path, "docs.jsonl", [{"q": "?", "a": "yes"}])
    imported = convert("lm-eval", task, dataset=docs)[0]
    assert imported.suite["cases"][0]["expected"] == "yes"


def test_lmeval_multiple_choice_approximated_with_loud_warning(tmp_path):
    task = tmp_path / "task.yaml"
    task.write_text(textwrap.dedent("""
        task: mc_demo
        output_type: multiple_choice
        doc_to_text: "{{question}}"
        doc_to_target: "{{gold}}"
        doc_to_choice: choices
    """))
    docs = _write_jsonl(tmp_path, "docs.jsonl", [
        {"question": "Pick", "choices": ["a", "b", "c"], "gold": 1},
    ])
    imported = convert("lm-eval", task, dataset=docs)[0]
    case = imported.suite["cases"][0]
    assert case["expected"] == "B"
    assert "B) b" in case["input"]
    assert any("loglikelihood" in w for w in imported.warnings)
    assert "loglikelihood" in imported.suite["description"]


def test_lmeval_requires_local_dataset(tmp_path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "doc_to_text: '{{q}}'\ndoc_to_target: '{{a}}'\ndataset_path: hf/hub-name\n"
    )
    with pytest.raises(SuiteImportError, match="--dataset"):
        convert("lm-eval", task)


def test_lmeval_jinja_filter_refused(tmp_path):
    task = tmp_path / "task.yaml"
    task.write_text("doc_to_text: '{{q | upper}}'\ndoc_to_target: a\n")
    docs = _write_jsonl(tmp_path, "docs.jsonl", [{"q": "x", "a": "y"}])
    with pytest.raises(SuiteImportError, match="filters"):
        convert("lm-eval", task, dataset=docs)


def test_lmeval_doc_prefix_transparent(tmp_path):
    # lm-eval templates commonly write {{doc.field}}.
    task = tmp_path / "task.yaml"
    task.write_text("doc_to_text: 'Q: {{doc.q}}'\ndoc_to_target: '{{doc.a}}'\n")
    docs = _write_jsonl(tmp_path, "docs.jsonl", [{"q": "?", "a": "!"}])
    imported = convert("lm-eval", task, dataset=docs)[0]
    assert imported.suite["cases"][0]["input"] == "Q: ?"
    assert imported.suite["cases"][0]["expected"] == "!"


# ---------------------------------------------------------------------------
# OpenAI evals
# ---------------------------------------------------------------------------

def test_openai_evals_samples(tmp_path):
    src = _write_jsonl(tmp_path, "samples.jsonl", [
        {"input": [{"role": "system", "content": "calc"},
                   {"role": "user", "content": "3*3?"}],
         "ideal": "9"},
        {"input": "Ocean between America and Europe?",
         "ideal": ["Atlantic", "Atlantic Ocean"]},
    ])
    imported = convert("openai-evals", src)[0]
    suite = _roundtrip(tmp_path, imported)
    assert suite.scoring == "exact_match"
    assert suite.cases[0].expected == "9"
    assert suite.cases[1].expected == "Atlantic"
    assert any("first was kept" in w for w in imported.warnings)


def test_openai_evals_empty_file_errors(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(SuiteImportError, match="no samples"):
        convert("openai-evals", p)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_import_cli_writes_and_validates(tmp_path):
    from click.testing import CliRunner

    from rift.cli import main

    src = _write_jsonl(tmp_path, "samples.jsonl", [
        {"input": "2+2?", "target": "4"},
    ])
    out = tmp_path / "suites" / "imported.yaml"
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["import", str(src), "--from", "inspect", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    suite = load_suite(str(out))
    assert suite.cases[0].expected == "4"
    assert "Next step" in result.output


def test_import_cli_split_names_files_by_variant(tmp_path):
    from click.testing import CliRunner

    from rift.cli import main

    src = tmp_path / "pf.yaml"
    src.write_text(textwrap.dedent("""
        prompts: ["{{x}}"]
        tests:
          - vars: {x: a}
            assert: [{type: equals, value: a}]
          - vars: {x: b}
            assert: [{type: regex, value: "b+"}]
    """))
    out = tmp_path / "imported.yaml"
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["import", str(src), "--from", "promptfoo", "-o", str(out),
         "--split-by-assert"],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "imported_exact_match.yaml").exists()
    assert (tmp_path / "imported_regex_match.yaml").exists()


def test_import_cli_clean_error_on_mixed_asserts(tmp_path):
    from click.testing import CliRunner

    from rift.cli import main

    src = tmp_path / "pf.yaml"
    src.write_text(textwrap.dedent("""
        prompts: ["{{x}}"]
        tests:
          - vars: {x: a}
            assert: [{type: equals, value: a}]
          - vars: {x: b}
            assert: [{type: contains, value: b}]
    """))
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["import", str(src), "--from", "promptfoo", "-o", str(tmp_path / "o.yaml")],
    )
    assert result.exit_code == 2  # operational error
    assert "--split-by-assert" in result.output
    assert "Traceback" not in result.output


def test_imported_suite_runs_through_custom_scorer(tmp_path):
    """The emitted custom_scorer reference must actually resolve and score."""
    from rift.scoring import get_scorer

    scorer = get_scorer("custom",
                        custom_scorer="rift.adapters.scorers:contains")
    assert scorer.score("well, chien it is", "chien") == 1.0
    assert scorer.score("cat", "chien") == 0.0
