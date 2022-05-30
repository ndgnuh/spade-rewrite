#!/bin/sh

VERSION=$(cat .docker/VERSION)
docker run --publish 8501:8501 --rm $VERSION
