#!/usr/bin/env bash
git diff master -- implementation tests | ./virtual_env/bin/flake8 --diff
