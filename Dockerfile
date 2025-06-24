# This file was created by ]init[ AG 2023.
#
# see https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=cudnn8-devel-ubuntu22.0
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive
ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
RUN apt-get update && \
	apt-get install -y ca-certificates && \
	# Upgrade system
	apt-get full-upgrade -y && \
	# Install locale and timezone data, set timezone Berlin
	apt-get install -y locales tzdata && \
	echo "de_DE.UTF-8 UTF-8" > /etc/locale.gen && \
	locale-gen && \
	ln -fs /usr/share/zoneinfo/Europe/Berlin /etc/localtime && \ 
	dpkg-reconfigure --frontend noninteractive tzdata && \
	# Python & Pip
	apt-get install -y python3-pip && \
	# Curl, Git, rsync, unrar, postgresql-client and jq
	apt-get install -y curl git rsync unrar jq && \
	# Cleanup
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

# Set default locale de_DE.utf8
ENV LANG de_DE.utf8
ENV LC_ALL de_DE.utf8

# Update Pip & install Poetry
ENV POETRY_VIRTUALENVS_CREATE=false
RUN pip install --upgrade pip setuptools poetry && \
	poetry config installer.max-workers 10

# Install UV, Certifi and upgrade Python libs with CVEs
# CVEs: https://harbor.init.de/harbor/projects/62973/repositories/ki-suche%2Fcopilot/artifacts-tab
RUN pip install --upgrade --break-system-packages certifi poetry uv ${PIP_INDEX_URL}

WORKDIR /opt/mteb
COPY pyproject.toml ./

EXPOSE 8200

FROM base AS dev
ENV TARGET=dev

RUN --mount=type=cache,target=/root/.cache uv pip install --break-system-packages --editable .[gritlm,dev] --python $(which python3) ${PIP_INDEX_URL}

# FROM base AS run
# ENV TARGET=run
# COPY .env LICENSE ./
# RUN pip3 install .

# COPY start_services.sh .
# RUN chmod +x start_services.sh
# CMD ./start_services.sh
# HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl --include --request GET http://localhost:8200/health || exit 1
