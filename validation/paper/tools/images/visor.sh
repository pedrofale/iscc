# VISOR 1.1.2.1 (bioconda): HACk builds a haplotype-resolved tumour genome, SHORtS simulates short
# reads from it (pywgsim) and aligns them (minimap2). samtools and bedtools for the per-site depth.
micromamba install -y -p /opt/conda -c conda-forge -c bioconda python=3.10 "visor=1.1.2.1" samtools bedtools
export PATH=/opt/conda/bin:$PATH
VISOR --help > /dev/null
{ echo "VISOR $(micromamba list -p /opt/conda visor | awk '$1=="visor"{print $2}')"
  for p in pywgsim minimap2 samtools; do echo "$p $(micromamba list -p /opt/conda $p | awk -v p=$p '$1==p{print $2}')"; done
} > /opt/versions.txt
micromamba clean -a -y
