#!/bin/bash
#SBATCH --job-name=causal_models
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-65

MODEL_TYPES=("diffusion" "causalflow")
OUTPUT_DIRS=("output/adult/med/diffusion" "output/adult/med/causalflow")

# Delay start to stagger jobs: 10s × array index
SLEEP_TIME=$((SLURM_ARRAY_TASK_ID * 10))
echo "Sleeping ${SLEEP_TIME}s before starting..."
sleep $SLEEP_TIME

if [ $SLURM_ARRAY_TASK_ID -lt 33 ]; then
    MODEL=${MODEL_TYPES[0]}
    OUTDIR=${OUTPUT_DIRS[0]}
    WORLD_INDEX=$SLURM_ARRAY_TASK_ID
else
    MODEL=${MODEL_TYPES[1]}
    OUTDIR=${OUTPUT_DIRS[1]}
    WORLD_INDEX=$((SLURM_ARRAY_TASK_ID - 33))
fi

python fit_causal_models_gen_eval_counterfactuals.py \
    --model-type $MODEL \
    --output-dir $OUTDIR \
    --world-indexes "$WORLD_INDEX"
