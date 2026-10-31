#!/usr/bin/env bash
# Run inside a new mount namespace. Requires mount privileges in that namespace.
set -euo pipefail
if [[ $# != 4 ]]; then
  echo 'usage: isolate.sh BUNDLE MODEL_FILENAME DEVICE ELEMENTS' >&2
  exit 2
fi
bundle=$1
model=$2
device=$3
elements=$4
if [[ "$model" == */* || "$model" != *.om ]]; then
  echo 'MODEL_FILENAME must be a basename ending in .om' >&2
  exit 2
fi
mount --make-rprivate /
mount -t tmpfs tmpfs /tmp
mkdir /tmp/deployment
mount --bind "$bundle" /tmp/deployment
mount -o remount,bind,ro /tmp/deployment
mount -t tmpfs tmpfs /home
mount -t tmpfs tmpfs /root
mount -t tmpfs tmpfs /var/tmp
mkdir -p /tmp/empty-home /tmp/empty-cache
export HOME=/tmp/empty-home
export XDG_CACHE_HOME=/tmp/empty-cache
unset ASCEND_CUSTOM_OPP_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
cd /tmp/deployment
[[ ! -e "$bundle" ]]
[[ -z "$(ls -A /home)" && -z "$(ls -A /root)" && -z "$(ls -A /var/tmp)" ]]
echo '[ISOLATION] private_mounts=1 original_bundle_hidden=1 home_hidden=1 root_hidden=1 fresh_tmp=1 fresh_var_tmp=1 custom_opp_unset=1 deployment_readonly=1'
exec ./model_exec "$model" "$device" "$elements"
