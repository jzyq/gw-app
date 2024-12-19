FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/ascend-infer-310b:24.0.RC1-dev-arm

USER 0

# Make project directory.
# We will put dispatcher and its dependences in this.
RUN mkdir -p /app
WORKDIR /app

# Copy projcet files,
# Note Miniconda3 installer should manual download and move into build contaxt.
COPY dispatcher/  .
COPY gw ./gw
COPY gwmodel ./gwmodel

# Build and Install ACLLite.
#
# 1. Install dependences.
RUN apt update && \
    apt install -y --no-install-recommends \
    libavcodec-dev \
    libavdevice-dev && \
    apt clean
#
# 2. Setup build env
ENV DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
ENV NPU_HOST_LIB=${DDK_PATH}/runtime/lib64/stub
#
# 3. Run build script to build ACLLite.
RUN --mount=type=bind,target=/tmp,rw \
    cd /tmp/3party/ACLLite && \
    python build_so.py
#
# It will auto install all libs into /lib.
# Now depoly acllite python packages.
#
# 4. Set envs.
ENV THIRDPART_PATH=${DDK_PATH}/thirdpart
ENV PYTHONPATH=${THIRDPART_PATH}:${PYTHONPATH}
#
# 5. Copy packages into thirdparty.
RUN --mount=type=bind,target=/tmp \
    cp -r /tmp/3party/ACLLite/python ${THIRDPART_PATH}/acllite
#
# ACLLite install ok, EOP.

# Give project directory to hw user becuase ascend require a specified user id to run it.
RUN chown -R HwHiAiUser:HwHiAiUser /app
USER 1000

# Install conda and setup projcet env.
RUN --mount=type=bind,target=/tmp \
    bash /tmp/Miniconda3-latest-Linux-aarch64.sh -p /app/miniconda -b && \
    /app/miniconda/bin/conda init

ENV PATH=/app/miniconda/bin:${PATH}
RUN conda create --name gw -y python=3.11

ENV CONDA_DEFAULT_ENV=gw
RUN echo "conda activate gw" >> ~/.bashrc

# If build too slow in mainland, use aliyun mirros when need.
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# Install dependences.
RUN pip install -r gw/requirements.txt --no-deps --no-cache-dir
RUN pip install -r gwmodel/requirements.txt --no-deps --no-cache-dir

CMD ["python", "task_dispatcher.py"]
