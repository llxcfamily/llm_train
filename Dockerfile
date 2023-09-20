FROM nvcr.io/nvidia/pytorch:23.05-py3

WORKDIR /workspace
COPY . /workspace

# necessary stuff
# for china pip source, add extra-index-url
# --extra-index-url http://mirrors.cloud.aliyuncs.com/pypi/simple/ --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip install torch==2.0.0 \
    peft==0.5.0 \
    transformers==4.31.0 \
    transformers-stream-generator \
    deepspeed==0.9.5 \
    accelerate==0.21.0 \
    gunicorn==20.1.0 \
    flask==2.1.2 \
    flask_api \
    langchain \
    fastapi==0.89.1 \
    uvicorn==0.19.0 \
    jinja2==3.1.2 \
    huggingface_hub \
    grpcio-tools==1.48.2 \
    bitsandbytes==0.38.0 \
    sentencepiece==0.1.99 \
    safetensors \
    datasets \
    texttable \
    toml  \
    numpy==1.24.4 \
    scikit-learn==1.3.0 \
    loguru==0.7.0 \
    protobuf==3.20.0 \
    pydantic==1.10.7 \
    python-dotenv==1.0.0 \
    tritonclient[all] \
    sse_starlette

ENV TRANSFORMERS_CACHE=/workspace/HF_cache \
    HUGGINGFACE_HUB_CACHE=${TRANSFORMERS_CACHE} \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64/"

RUN mkdir /.cache && \
    chmod -R g+w /.cache
