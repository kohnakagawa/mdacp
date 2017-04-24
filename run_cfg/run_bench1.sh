#!/bin/sh
#QSUB -queue F72acc
#QSUB -node 32
#QSUB -mpi 64
#QSUB -omp 8
#QSUB -place distribute
#QSUB -over false
#PBS -l walltime=00:30:00
#PBS -N mdacp-gpu-bench-64gpus

cd $PBS_O_WORKDIR
. /etc/profile.d/modules.sh

mpijob ./mdacp input_gpu.cfg
