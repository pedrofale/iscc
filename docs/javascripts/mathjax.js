// MathJax for the rendered notebooks. The tutorials are notebook HTML rather than markdown, so the
// usual markdown math extension never sees them — MathJax has to typeset the page itself. Code blocks
// are excluded so a stray "$" in code is not mistaken for math.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: "highlight|jp-InputArea|no-mathjax",
    processHtmlClass: "arithmatex|jp-RenderedMarkdown|md-typeset",
  },
};
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
