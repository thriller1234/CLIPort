FROM nvidia/cudagl:11.1.1-devel-ubuntu18.04

ARG USER_NAME=cliportuser
ARG USER_PASSWORD=cliportpass
ARG USER_UID=1000
ARG USER_GID=1000

# NVIDIA CUDA GPG key追加（失敗しても続行）
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub || true

# 基本パッケージのインストール
RUN apt-get update && apt-get install -y sudo

# ユーザー作成
RUN useradd -ms /bin/bash $USER_NAME
RUN usermod -aG sudo $USER_NAME
RUN echo "$USER_NAME:$USER_PASSWORD" | chpasswd

# UID/GID の調整
RUN usermod -u $USER_UID $USER_NAME && groupmod -g $USER_GID $USER_NAME

# 作業ディレクトリ設定
WORKDIR /home/$USER_NAME

# 依存スクリプトのコピーと実行
COPY ./scripts/install_deps.sh /tmp/install_deps.sh
RUN yes "Y" | /tmp/install_deps.sh

# pip3 のインストールとアップグレード
RUN apt-get -y install python3-pip
RUN python3 -m pip install --upgrade pip

# PyTorch のインストール
RUN pip3 install \
    torch==1.9.1+cu111 \
    torchvision==0.10.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

# glxgears などの追加パッケージ
RUN apt-get update && apt-get install -y \
    mesa-utils \
    python3-setuptools \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# OpenCV のビルドに必要な依存ライブラリをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Pythonライブラリのインストール（順番に1行ずつ）
RUN pip3 install numpy==1.19.5
RUN pip3 install opencv-python==4.5.5.64
RUN pip3 install absl-py==0.7.0
RUN pip3 install gym==0.17.3
RUN pip3 install pybullet==3.0.4
RUN pip3 install matplotlib==3.3.4
RUN pip3 install meshcat==0.0.18
RUN pip3 install scipy==1.4.1
RUN pip3 install scikit-image==0.17.2
RUN pip3 install transforms3d==0.3.1
RUN pip3 install pytorch_lightning==1.0.3
RUN pip3 install tqdm==4.62.3
RUN pip3 install hydra-core==1.0.5
RUN pip3 install wandb==0.10.15
RUN pip3 install transformers==4.3.2
RUN pip3 install kornia==0.4.1
RUN pip3 install ftfy==5.8
RUN pip3 install regex==2021.4.4
RUN pip3 install imageio-ffmpeg==0.4.5

# ユーザー所有ディレクトリ作成と環境変数設定
RUN mkdir /home/$USER_NAME/cliport && chown $USER_NAME:$USER_NAME -R /home/$USER_NAME/cliport
RUN echo "export CLIPORT_ROOT=~/cliport" >> /home/$USER_NAME/.bashrc
RUN echo "export PYTHONPATH=\$PYTHONPATH:~/cliport" >> /home/$USER_NAME/.bashrc
ENV CLIPORT_ROOT=/workspace/cliport
