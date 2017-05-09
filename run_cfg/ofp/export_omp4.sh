#!/bin/sh

## clear settings
unset I_MPI_DEBUG
unset I_MPI_PIN_ORDER
unset I_MPI_PIN_DOMAIN
unset KMP_AFFINITY
unset I_MPI_PIN_PROCESSOR_EXCLUDE_LIST
unset I_MPI_FABRICS_LIST
unset I_MPI_FABRICS
unset HFI_NO_CPUAFFINITY

## verbose
# export I_MPI_DEBUG=4

## set affinity
export I_MPI_PIN_ORDER=compact
export I_MPI_PIN_DOMAIN=auto:compact
export KMP_AFFINITY=granularity=fine

## do not bind core 0,1
export I_MPI_PIN_PROCESSOR_EXCLUDE_LIST=0,1,68,69,136,137,204,205

## communication settings
export I_MPI_FABRICS_LIST=tmi
export I_MPI_FABRICS=shm:tmi
# export HFI_NO_CPUAFFINITY=1

