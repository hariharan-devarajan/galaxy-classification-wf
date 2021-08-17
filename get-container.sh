#!/usr/bin/env bash

mkdir containers
singularity pull containers/galaxy-container_latest.sif docker://papajim/galaxy-container:works21
