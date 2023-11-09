#!/bin/bash

CODE_DIR=$PWD
DOCKERFILE=${CODE_DIR}/Dockerfile
TAG=gpt_demo
NAME=gpt_demo
GPU_TYPE=A100

# build docker image
docker build --tag ${TAG} --file ${DOCKERFILE} --build-arg GPU_TYPE=${GPU_TYPE}  .
