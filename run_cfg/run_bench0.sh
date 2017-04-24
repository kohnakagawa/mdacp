#!/bin/sh
#QSUB -queue F18acc
#QSUB -node 4
#QSUB -mpi 8
#QSUB -omp 8
#QSUB -place distribute
#QSUB -over false
#PBS -l walltime=00:30:00
#PBS -N mdacp-gpu-bench-8gpus

cd $PBS_O_WORKDIR
. /etc/profile.d/modules.sh

mpijob ./mdacp input_gpu.cfg
