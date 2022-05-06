# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.234.0/containers/python-3-anaconda/.devcontainer/base.Dockerfile

FROM mcr.microsoft.com/vscode/devcontainers/anaconda:0-3

# install packages required by opencv: ffmpeg libsm6 libxext6
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends ffmpeg libsm6 libxext6
