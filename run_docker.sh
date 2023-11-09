#!/bin/bash

CODE_DIR=$PWD
DOCKERFILE=${CODE_DIR}/Dockerfile
TAG=gpt_demo
NAME=gpt_demo
GPU_TYPE=A100

#
# build docker image
#docker build --tag ${TAG} --file ${DOCKERFILE} --build-arg GPU_TYPE=${GPU_TYPE}  .

# run docker container
docker run -d -it --rm --gpus all --ulimit memlock=-1 --ulimit stack=67108864  --security-opt seccomp:unconfined \
    --shm-size=4g \
    -v ${CODE_DIR}:/workspace/ \
    --name ${NAME} \
    ${TAG}

# exec docker container
docker exec -it ${NAME} /bin/bash
