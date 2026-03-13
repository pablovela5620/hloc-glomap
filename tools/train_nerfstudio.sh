#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
data_dir="$(realpath "${DATA_DIR:-data}")"
max_num_iterations="${MAX_NUM_ITERATIONS:-7000}"

case "$mode" in
  glomap)
    exp_name="${EXP_NAME:-glomap-splat}"
    colmap_path="$(realpath "${GLOMAP_DIR:-${COLMAP_DIR:-${data_dir}/glomap-aliked-n16-aliked+lightglue/sparse/0}}")"
    vis="${NS_VIS:-viewer}"
    ;;
  colmap)
    exp_name="${EXP_NAME:-colmap-splat}"
    colmap_path="$(realpath "${COLMAP_DIR:-${data_dir}/colmap-aliked-n16-aliked+lightglue/sparse/0}")"
    vis="${NS_VIS:-viewer+tensorboard}"
    ;;
  *)
    echo "Usage: $0 {glomap|colmap}" >&2
    exit 2
    ;;
esac

exec ns-train splatfacto \
  --experiment-name "$exp_name" \
  --pipeline.model.use-bilateral-grid True \
  --vis "$vis" \
  --viewer.quit-on-train-completion True \
  --output-dir "$data_dir/outputs" \
  --max-num-iterations "$max_num_iterations" \
  colmap \
  --data "$data_dir" \
  --colmap-path "$colmap_path"
