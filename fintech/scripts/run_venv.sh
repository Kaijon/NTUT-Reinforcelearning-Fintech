#!/bin/bash

cmd_line="$@"
echo "$cmd_line"

source activate
$cmd_line
deactivate
