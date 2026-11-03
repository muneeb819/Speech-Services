#!/bin/sh
# faster-whisper (CTranslate2) needs CUDA 12 cuBLAS + cuDNN 9; the PyTorch wheels ship them as pip packages.
NV=$(python - <<'PY'
import glob, site
print(":".join(glob.glob(site.getsitepackages()[0] + "/nvidia/*/lib")))
PY
)
export LD_LIBRARY_PATH="${NV}:${LD_LIBRARY_PATH}"
exec "$@"
