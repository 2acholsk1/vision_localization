FROM pytorch/pytorch

ARG USERNAME=zacholski
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV TZ=Europe/Warsaw

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get clean && apt-get update

RUN apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git

RUN apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev

RUN rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

ENV SHELL /bin/bash

RUN apt-get autoremove -y && apt-get clean -y

USER $USERNAME
CMD ["/bin/bash"]
