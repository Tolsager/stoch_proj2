#!/bin/bash
#BSUB -J dis[0-3]
#BSUB -q hpc
#BSUB -W 1:00
#BSUB -n 1
#BSUB -R "rusage[mem=200MB]"
#BSUB -o dis_%J.out
#BSUB -e dis_%J.err

conda activate stoch
python sim_dis.py $LSB_JOBINDEX
