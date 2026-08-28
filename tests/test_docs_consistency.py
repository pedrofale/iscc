"""Standing check: does what we SAY still match what the code DOES?

Every consistency bug found in the 2026-08-27 audit had one shape — a model or a config changed,
and the things describing it did not follow:

* ``kill_mode="proliferation"`` was written for the escape-modes work and reached exactly one file,
  leaving the hero, the metastasis notebook and the PM cohort on a kill law whose own docstring says
  it "cannot serve clones whose birth rates differ threefold".
* ``tool_rctd_R`` kept telling readers it was "not yet on the realistic ductal field" for a day after
  it was migrated onto it.
* ``realistic_regime`` documented ``max_cells`` as "the config's 50k" long after the config said 8,000.
* ``02_tumor_growth`` still teaches the CELL engine's kill formula, while every tutorial runs the
  genotype engine, which does not read the parameter the notebook tunes.
* ``configs/landing.yaml``'s reproduce command omitted ``--skip-confinement``, added seventeen days
  earlier by a commit named for the problem it solves, so the documented command could not
  regenerate the asset it claimed to.

None of those are visible in a diff of the file that changed. They are only visible by reading one
artefact against another, which is what this module automates. Everything here is fast and
simulation-free; the expensive behavioural checks live in the validation harness.
"""
import json
import os
import re
import glob

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = sorted(glob.glob(os.path.join(REPO, "notebooks", "*.ipynb")))
CONFIG = os.path.join(REPO, "notebooks", "example_config.yaml")
LANDING = os.path.join(REPO, "configs", "landing.yaml")


def _cells(path, kind=None):
    nb = json.load(open(path))
    return [c for c in nb["cells"] if kind is None or c["cell_type"] == kind]


def _src(cell):
    return "".join(cell["source"])


def _name(path):
    return os.path.basename(path)


# --------------------------------------------------------------------------------------------------
# 1. Shared constants must be DERIVED from one source, not restated.
# --------------------------------------------------------------------------------------------------
def test_normals_is_not_hand_copied():
    """NORMALS was written out three times and the notebook copy had already lost "host" — the met
    compartment's own normal tissue. Nothing used it, so nothing failed."""
    import sys
    sys.path.insert(0, os.path.join(REPO, "validation"))
    sys.path.insert(0, os.path.join(REPO, "notebooks"))
    from iscc.constants import normal_names
    import realistic_regime as RR
    import base_sim as BS

    engine = tuple(normal_names)
    assert tuple(RR.NORMALS) == engine, (
        f"realistic_regime.NORMALS {RR.NORMALS} != engine {engine}; derive it, do not restate it")
    assert tuple(BS.NORMALS) == engine, (
        f"base_sim.NORMALS {BS.NORMALS} != engine {engine}; derive it, do not restate it")


def test_expression_params_hands_out_copies():
    """programs_common.expression_params returned its module constants BY REFERENCE, and
    base_sim.EXPR() edits what it is given — so calling EXPR() silently rewrote the calibrated
    cohort coupling for the rest of the process."""
    import sys
    sys.path.insert(0, os.path.join(REPO, "validation"))
    import programs_common as PC

    a = PC.expression_params()
    for block in ("activity_params", "coupling_params", "program_params"):
        assert a[block] is not getattr(PC, block.upper(), None), (
            f"expression_params()['{block}'] is the module constant itself; a caller that edits the "
            f"returned dict would rewrite it for every later caller")
    a["coupling_params"]["__canary__"] = 1
    assert "__canary__" not in PC.COUPLING_PARAMS, (
        "editing the returned coupling_params mutated programs_common.COUPLING_PARAMS")


# --------------------------------------------------------------------------------------------------
# 2. Prose numbers must match the config they describe.
# --------------------------------------------------------------------------------------------------
def test_prose_parameter_claims_match_the_config():
    """Markdown that names a parameter value must name one the config actually produces.

    Scale presets are legitimate (the analysis datasets run at "mid"), so a claim is accepted if it
    matches ANY preset — the check catches values that match NONE, which is how RCTD's "grid-26" and
    realistic_regime's "50k" survived.
    """
    import sys
    sys.path.insert(0, os.path.join(REPO, "validation"))
    import realistic_regime as RR

    cfg = yaml.safe_load(open(CONFIG))
    gen = cfg["genome_params"]
    ok_grid = {s["grid_size"] for s in RR.SCALES.values()}
    ok_glands = {s["n_glands"] for s in RR.SCALES.values()}
    ok_genes = {gen["n_segments"] * gen["segment_size"], gen["segment_size"]}

    checks = [(re.compile(r"grid[ -]?(\d{2,4})", re.I), ok_grid, "grid_size"),
              (re.compile(r"(\d{1,2})\s+(?:ducts|glands)\b", re.I), ok_glands, "n_glands"),
              (re.compile(r"(\d[\d,]{2,})\s*genes", re.I), ok_genes, "gene count")]

    bad = []
    for p in NOTEBOOKS:
        for i, c in enumerate(_cells(p, "markdown")):
            s = _src(c)
            for pat, allowed, label in checks:
                for m in pat.finditer(s):
                    v = int(m.group(1).replace(",", ""))
                    if v not in allowed:
                        bad.append(f"{_name(p)} md cell {i}: {label} '{m.group(0).strip()}' "
                                   f"is not any configured value {sorted(allowed)}")
    assert not bad, "prose claims a parameter value the config does not produce:\n  " + "\n  ".join(bad)


def _prose_of(path):
    """Comment and docstring text only — never executable lines.

    The first version of this check scanned raw source and flagged `_CFG.get("max_cells", 50000)`
    (a legitimate fallback) and a neighbouring gene count. A standing check that cries wolf gets
    switched off, so it reads PROSE and nothing else.
    """
    import ast
    src = open(path).read()
    out = [l.split("#", 1)[1] for l in src.splitlines() if l.strip().startswith("#")]
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "\n".join(out)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            if doc:
                out.append(doc)
    return "\n".join(out)


def test_max_cells_module_matches_the_config():
    """The real invariant: realistic_regime's cap IS the config's cap.

    The docstring bug ("defaults to the config's 50k" when it was 8,000) was downstream of the config
    being lowered. Asserting the mechanism is robust; regex over free prose is not, which the first
    two attempts at this check demonstrated by flagging a legitimate fallback and a gene count.
    """
    import sys
    sys.path.insert(0, os.path.join(REPO, "validation"))
    import realistic_regime as RR

    cfg = yaml.safe_load(open(CONFIG))
    assert RR.MAX_CELLS == cfg["max_cells"], (
        f"realistic_regime.MAX_CELLS is {RR.MAX_CELLS} but {os.path.basename(CONFIG)} says "
        f"{cfg['max_cells']}")


def test_prose_does_not_quote_a_stale_max_cells():
    """A stated cap must be the real cap. Scoped tightly: the number must FOLLOW max_cells within a
    short window that names no other quantity, so "max_cells x n_genes ... 6,000 genes" is not read
    as a cap claim."""
    import sys
    sys.path.insert(0, os.path.join(REPO, "validation"))
    import realistic_regime as RR

    prose = _prose_of(os.path.join(REPO, "validation", "realistic_regime.py"))
    for m in re.finditer(r"max_cells(\D{0,45}?)(\d[\d,_]*)\s*(k\b)?", prose, re.I):
        gap, raw, kilo = m.group(1), m.group(2), m.group(3)
        if re.search(r"gene|segment|grid|deme|patient", gap, re.I):
            continue                                  # the number belongs to another quantity
        val = int(raw.replace(",", "").replace("_", "")) * (1000 if kilo else 1)
        if val >= 1000 and val != RR.MAX_CELLS:
            pytest.fail(f"prose documents max_cells as {val:,} but MAX_CELLS is {RR.MAX_CELLS:,}: "
                        f"...{m.group(0).strip()}...")


# --------------------------------------------------------------------------------------------------
# 3. Reproduce commands must carry the flags that make them correct.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("flag, why", [
    ("--skip-confinement",
     "origin_confinement seals the founding acinus for ~84% of the run; without this the animation "
     "spends most of its frames on a duct where nothing moves, and the GIF palette desaturates the "
     "relapse (de3ccb7 added the flag for exactly this)"),
    ("--no-tables",
     "isccgif --compartment reads only the trajectory; the count tables are ~47 of 68 minutes and "
     "~4.7 of 5 GB of output it never opens"),
])
def test_landing_reproduce_command_is_the_one_that_works(flag, why):
    """The header of configs/landing.yaml tells the reader how to regenerate the docs hero. It has
    twice drifted from the command that actually produces the shipped asset."""
    header = "\n".join(l for l in open(LANDING).read().splitlines() if l.startswith("#"))
    assert flag in header, f"configs/landing.yaml's reproduce command is missing {flag}: {why}"


def test_overview_and_config_document_the_same_hero_command():
    """docs/overview.md and the config header describe the same regeneration. When one gains a flag
    and the other does not, one of them is wrong and the reader cannot tell which."""
    header = "\n".join(l for l in open(LANDING).read().splitlines() if l.startswith("#"))
    overview = open(os.path.join(REPO, "docs", "overview.md")).read()
    if "isccgif" not in overview:
        pytest.skip("overview.md no longer documents the hero render")
    for flag in ("--skip-confinement",):
        assert flag in overview, (
            f"configs/landing.yaml's reproduce command uses {flag} but docs/overview.md does not")
        assert flag in header


# --------------------------------------------------------------------------------------------------
# 4. Treatment: the kill law must be stated, never silently defaulted.
# --------------------------------------------------------------------------------------------------
def test_every_configured_chemotherapy_states_its_kill_mode():
    """`kill_mode` defaults to "additive" in arc.py. That default is how the hero ended up on a kill
    law the engine's own docstring rejects, five days after a better one was written, with nothing
    in the config to show a choice had been made. Requiring the key makes the choice visible.
    """
    missing = []
    for path in sorted(glob.glob(os.path.join(REPO, "configs", "*.yaml"))):
        cfg = yaml.safe_load(open(path)) or {}
        for ph in (cfg.get("schedule") or {}).get("phases", []):
            if ph.get("op") == "chemotherapy" and "kill_mode" not in ph:
                missing.append(f"{os.path.basename(path)}: chemotherapy phase has no kill_mode")
    assert not missing, (
        "a chemotherapy phase leaves the kill law to arc.py's default:\n  " + "\n  ".join(missing))


# --------------------------------------------------------------------------------------------------
# 5. Published notebooks: no internals, no dev scaffolding, no stale execution.
# --------------------------------------------------------------------------------------------------
def test_notebooks_do_not_reach_past_the_public_api():
    """A tutorial that documents the API should not call through the underscore. base_sim wraps the
    two that used to (cancer_clones, stage_palette)."""
    bad = []
    for p in NOTEBOOKS:
        for i, c in enumerate(_cells(p, "code")):
            for m in re.finditer(r"\b(?!np|pd|plt|self)\w+\.(_[a-z]\w*)\(", _src(c)):
                bad.append(f"{_name(p)} code cell {i}: {m.group(0)}")
    assert not bad, "notebook calls a private API:\n  " + "\n  ".join(bad)


def test_notebooks_carry_no_internal_scaffolding():
    """DESIGN_*.md, BACKLOG and feature tags (F8, M2, R13) are internal planning artefacts. They
    render on the public docs site."""
    tag = re.compile(r"DESIGN_[A-Za-z_]*\.md|BACKLOG|(?<![A-Za-z0-9_])[FMR]\d{1,2}(?![A-Za-z0-9_])"
                     r"|PARAMETERS\.md|SCHEMA\.md|handoffs/")
    bad = []
    for p in NOTEBOOKS:
        for i, c in enumerate(_cells(p)):
            for m in tag.finditer(_src(c)):
                bad.append(f"{_name(p)} cell {i}: '{m.group(0)}'")
    assert not bad, "notebook ships internal scaffolding:\n  " + "\n  ".join(bad)


def test_notebooks_are_fully_executed_and_error_free():
    """A notebook whose stored outputs are stale or partial is a notebook nobody can trust. Every
    code cell must have run, in order, with no error output."""
    bad = []
    for p in NOTEBOOKS:
        code = _cells(p, "code")
        counts = [c.get("execution_count") for c in code]
        if any(c.get("execution_count") is None and not c.get("outputs") for c in code):
            bad.append(f"{_name(p)}: has an unexecuted code cell")
        run = [c for c in counts if c is not None]
        if run and run != sorted(run):
            bad.append(f"{_name(p)}: executed out of order {run}")
        for i, c in enumerate(code):
            if any(o.get("output_type") == "error" for o in c.get("outputs", [])):
                bad.append(f"{_name(p)}: code cell {i} stored an error output")
    assert not bad, "notebook execution state:\n  " + "\n  ".join(bad)


def test_notebook_outputs_do_not_publish_dependency_warnings():
    """dask/numba/anndata/tqdm import warnings say nothing about the tumour and print the absolute
    path of the site-packages file they came from onto a public page.
    scripts/clean_notebook_outputs.py strips them."""
    dep = re.compile(r"site-packages/.*:\d+:\s*\w*(Warning|Error)")
    bad = []
    for p in NOTEBOOKS:
        for i, c in enumerate(_cells(p, "code")):
            for o in c.get("outputs", []):
                if dep.search("".join(o.get("text", []))):
                    bad.append(f"{_name(p)} cell {i}")
    # NB: the hint names no path. This file is published to `main`, where scripts/ (an internal-only
    # directory) does not exist — a pointer to it would be dangling for exactly the readers who hit
    # this failure there.
    assert not bad, ("dependency import warnings in published outputs; strip the warning blocks from "
                     "the stored outputs (the repo's notebook-output cleaner does this):\n  "
                     + "\n  ".join(sorted(set(bad))))


def test_cross_notebook_references_are_links():
    """A bare filename is not navigable on the rendered site."""
    bad = []
    for p in NOTEBOOKS:
        for i, c in enumerate(_cells(p)):
            s = _src(c)
            for m in re.finditer(r"([\w/]+\.ipynb)", s):
                before = s[max(0, m.start() - 30):m.start()]
                if not (before.endswith("](") or "href=" in before):
                    bad.append(f"{_name(p)} cell {i}: bare '{m.group(1)}'")
    assert not bad, "cross-notebook reference is not a link:\n  " + "\n  ".join(bad)
