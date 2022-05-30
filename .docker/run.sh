#!/bin/sh

VERSION=$(cat .docker/VERSION)
docker run --port 8501:8501 --rm $VERSION
