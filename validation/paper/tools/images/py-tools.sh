# Python copy-number tools, one environment each in one image:
#   base         infercnvpy 0.6.1 (as the laptop's iscc-infercnv) with scanpy
#   cancerfinder Cancer-Finder (GitHub, pinned; MIT) on its own pins: python 3.9, torch 1.13.1 CPU
#   calicost     CalicoST (GitHub, pinned; BSD-3) on its environment.yml pins: python 3.10,
#                numpy 1.24.4, scipy 1.11.3. Startle (the phylogeny step, which needs CPLEX) is not
#                installed; clones and allele-specific copy number do not need it.
micromamba install -y -p /opt/conda -c conda-forge python=3.10 pip git
export PATH=/opt/conda/bin:$PATH
python -m pip install --no-cache-dir "infercnvpy==0.6.1" scanpy
micromamba create -y -p /opt/envs/cancerfinder -c conda-forge python=3.9.16 pip git
/opt/envs/cancerfinder/bin/python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
  "torch==1.13.1+cpu" "torchvision==0.14.1+cpu"
/opt/envs/cancerfinder/bin/python -m pip install --no-cache-dir "anndata==0.9.1" "numpy==1.23.0" \
  "pandas==1.5.2" "scanpy==1.9.3" zarr gdown
git clone -q https://github.com/Patchouli-M/SequencingCancerFinder.git /opt/cancerfinder
git -C /opt/cancerfinder checkout -q ec88d4d9b75d589aa3635fb1a40b0b71a78657be
# CalicoST's hmrf.py imports turtle, which loads tkinter and so needs Tk and X11 even on a headless node.
micromamba create -y -p /opt/envs/calicost -c conda-forge -c bioconda python=3.10 "numpy=1.24.4" \
  "scipy=1.11.3" pip git tk xorg-libx11
/opt/envs/calicost/bin/python -m pip install --no-cache-dir \
  "calicost @ git+https://github.com/raphael-group/CalicoST.git@c1abcae3e3657e01e547ee4529e3b9d039221453"
git clone -q https://github.com/raphael-group/CalicoST.git /opt/calicost
git -C /opt/calicost checkout -q c1abcae3e3657e01e547ee4529e3b9d039221453
{
  python -c "import infercnvpy, scanpy; print('infercnvpy', infercnvpy.__version__); print('scanpy', scanpy.__version__)"
  /opt/envs/cancerfinder/bin/python -c "import torch, scanpy; print('cancerfinder ec88d4d torch', torch.__version__, 'scanpy', scanpy.__version__)"
  /opt/envs/calicost/bin/python -c "import calicost, numpy, scipy; print('calicost c1abcae numpy', numpy.__version__, 'scipy', scipy.__version__)"
} > /opt/versions.txt
micromamba clean -a -y
