# JUPITER Benchmark Suite: MMoCLIP

[![DOI](https://zenodo.org/badge/831410928.svg)](https://zenodo.org/badge/latestdoi/831410928) [![Static Badge](https://img.shields.io/badge/DOI%20(Suite)-10.5281%2Fzenodo.12737073-blue)](https://zenodo.org/badge/latestdoi/764615316)

This benchmark is part of the [JUPITER Benchmark Suite](https://github.com/FZJ-JSC/jubench). See the repository of the suite for some general remarks.

This repository contains the MMoCLIP benchmark. [`DESCRIPTION.md`](DESCRIPTION.md) contains details for compilation, execution, and evaluation. MMoCLIP runs [OpenCLIP](https://github.com/mlfoundations/open_clip), an open source implementation of [CLIP](https://openai.com/blog/clip/), which is a multi-modal contrastive model that can learn image and text representations from a dataset of image-text pairs.

The source code of OpenCLIP is included in the `./src/` subdirectory as a submodule from the upstream OpenCLIP repository at [github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip).