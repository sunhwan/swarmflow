# swarmflow runtime image.
#
# Build:
#   docker build -t swarmflow .
#
# Run (from a project directory containing config.yml + inputs):
#   docker run --rm --gpus all -v "$PWD":/work swarmflow --stage kinetics
#
# The host needs the NVIDIA Container Toolkit for --gpus to work; openmm
# brings its own CUDA libraries via conda-forge, so no host CUDA install
# is required, only a working nvidia driver.

FROM mambaorg/micromamba:1.5.8

USER root

# Build-time pins for git deps. Override with --build-arg <NAME>=<ref> to
# track a specific commit.
ARG SEEKR2_REF=main
ARG SEEKRTOOLS_REF=main
ARG PAPRIKA_REF=master

# Conda layer — pinned to match the host env where swarmflow was developed
# (openmm 8.1, ambertools 24, python 3.10).
RUN micromamba install -y -n base -c conda-forge \
        python=3.10 \
        openmm=8.1 seekr2_openmm_plugin \
        ambertools=24 parmed \
        rdkit networkx numpy scipy \
        matplotlib-base pyyaml tqdm \
        openff-toolkit-base openff-units \
        mdanalysis pymbar \
        git pip && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH=/opt/conda/bin:$PATH

# Pip layer — git installs that aren't on conda-forge. --no-deps because
# the deps are already satisfied by the conda layer; this avoids dragging
# in a second copy of numpy/scipy/etc.
RUN pip install --no-cache-dir --no-deps \
        "git+https://github.com/seekrcentral/seekr2@${SEEKR2_REF}" \
        "git+https://github.com/seekrcentral/seekrtools@${SEEKRTOOLS_REF}" \
        "git+https://github.com/GilsonLabUCSD/pAPRika@${PAPRIKA_REF}"

# swarmflow itself, editable so a host bind-mount of the source can be
# used during development without rebuilding.
COPY pyproject.toml /opt/swarmflow/
COPY swarmflow      /opt/swarmflow/swarmflow
RUN pip install --no-cache-dir --no-deps -e /opt/swarmflow

WORKDIR /work

ENTRYPOINT ["python", "-m", "swarmflow"]
CMD ["--help"]
