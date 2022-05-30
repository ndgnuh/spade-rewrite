#!/bin/sh

VERSION=$(cat .docker/VERSION)
docker run --rm $VERSION
