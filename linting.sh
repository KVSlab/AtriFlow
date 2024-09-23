#! /usr/bin/env bash

isort src

black src

flake8 src