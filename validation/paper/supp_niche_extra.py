"""S10 Fig: the panels moved out of Fig 2 when it was cut to half a page.

  (A) the showcase's cancer cells in gland, stroma and metastasis over generations (Fig 2A's timeline);
  (B, C) escape from the gland across seeds (the old Fig 2B i and ii);
  (D) confinement and the genome (the old Fig 2D);
  (E) the showcase's clone tree (the old Fig 2E i);
  (F) whole-genome-duplication prevalence across rates (the old Fig 2F ii).

Every panel is drawn by fig2_niche.py's functions from the same caches as Fig 2 (see its docstring for
the definitions), so this script computes nothing. Drawn at 165 mm, the text width, so that its text
prints at its true size. Writes ``figures/paper1/supp_niche_extra.pdf`` and its numbers to
``<OUT>/derived/fig2/niche_extra_render_numbers.json``.

Run (with ISCC_PAPER_OUT and ISCC_PAPER_TIER set as for fig2_niche.py):
    python validation/paper/supp_niche_extra.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import fig2_niche as F2  # noqa: E402
import style as S  # noqa: E402


def main():
    S.setup()
    fig = S.figure(165, 166)
    out = {}
    S.panel_letter(fig, "A", 1, 1)
    out["A"] = F2.render_history(S.ax_mm(fig, 14, 10, 148, 26))
    S.panel_letter(fig, "B", 1, 47)
    S.panel_letter(fig, "C", 60, 47)
    out["B, C"] = F2.render_escape(S.ax_mm(fig, 14, 54, 38, 30), S.ax_mm(fig, 74, 54, 15, 30))
    S.panel_letter(fig, "D", 104, 47)
    out["D"] = F2.render_confinement([S.ax_mm(fig, 116, 54, 18, 30), S.ax_mm(fig, 146, 54, 18, 30)])
    S.panel_letter(fig, "E", 1, 110)
    out["E"] = F2.render_tree(S.ax_mm(fig, 14, 117, 70, 33), legend_mm=9)
    S.panel_letter(fig, "F", 96, 110)
    out["F"] = F2.render_wgd(S.ax_mm(fig, 110, 117, 50, 33))
    path = S.save(fig, "supp_niche_extra")
    print("wrote", path)
    F2._numbers(out, "niche_extra_render_numbers.json")


if __name__ == "__main__":
    main()
