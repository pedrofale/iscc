#!/usr/bin/env bash
# publish-main.sh — project dev's PUBLIC subset onto main (the curated public branch).
#
# dev is the single source of truth for the CODE (the design docs live in the iscc-markdown
# repository and the manuscript in iscc-overleaf — see WORKSPACE.md). main is a
# path-filtered projection of dev: dev's whole tree MINUS the internal-only paths below. Nothing is
# ever committed to main by hand — run this instead, so main can never drift from dev.
#
# Run from a clean `dev` checkout, in the iscc conda env (needs mkdocs + the docs deps). It builds a
# commit on your LOCAL main and does NOT push — review it, then `git push origin main` (which triggers
# the Pages deploy).
set -euo pipefail

# --- internal-only paths (kept on dev for development, never published to public main) ---
INTERNAL_FILES=(
  PARAMETERS.md SCHEMA.md docs/parameters.md docs/schema.md
  notebooks/TUTORIALS_PLAN.md WORKSPACE.md
)
INTERNAL_DIRS=( analysis scripts .claude )
# NB: DESIGN_*.md, BACKLOG.md, AUDIT.md, RESEARCH_QUESTIONS.md, handoffs/ and manuscript/ left this
# repository for iscc-markdown / iscc-overleaf. They are gone from dev, so stripping them is a
# no-op — but the leak check below still names them, deliberately: if one is ever re-added here by
# accident, publishing must fail rather than push it to the public branch.

cd "$(git rev-parse --show-toplevel)"
[ "$(git branch --show-current)" = "dev" ] || { echo "ERROR: run from the dev branch"; exit 1; }
git diff --quiet && git diff --cached --quiet || { echo "ERROR: dev working tree not clean"; exit 1; }
DEV_SHA=$(git rev-parse --short HEAD)

cleanup() { git checkout -q dev 2>/dev/null || true; }
trap cleanup EXIT

git checkout -q main
# make main's tree EXACTLY dev's tree (adds new, drops files dev removed), then strip internal paths.
# -f is required because `checkout dev -- .` stages the files, and plain `git rm` refuses to remove
# staged changes.
git rm -rfq --ignore-unmatch . >/dev/null
git checkout -q dev -- .
git rm -rfq --ignore-unmatch "${INTERNAL_FILES[@]}" "${INTERNAL_DIRS[@]}" >/dev/null
find . -name .DS_Store -not -path './.git/*' -delete
git add -A

# SAFETY: never publish if an internal path leaked into the staged tree
LEAK=$(git diff --cached --name-only | grep -Ei \
  '^(DESIGN_|BACKLOG|AUDIT|RESEARCH_QUESTIONS|PARAMETERS|SCHEMA|WORKSPACE|handoffs/|analysis/|manuscript/|scripts/|\.claude|notebooks/TUTORIALS_PLAN)|docs/(parameters|schema)\.md|\.DS_Store' || true)
[ -z "$LEAK" ] || { echo "ABORT: internal paths would be published to main:"; echo "$LEAK"; \
  git reset -q --hard HEAD; git checkout -q dev; git branch -f main origin/main; exit 1; }

# gate: the public site must build strict before we record the commit
mkdocs build --strict --site-dir "$(mktemp -d)/site" >/dev/null

if git diff --cached --quiet; then
  echo "main already matches dev's public subset — nothing to publish."; exit 0
fi
git commit -q -m "publish: sync dev -> main public subset (dev ${DEV_SHA})"
echo "Published to LOCAL main: $(git rev-parse --short HEAD) (from dev ${DEV_SHA})."
echo "Review it, then deploy with:  git push origin main"
