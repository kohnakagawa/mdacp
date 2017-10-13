# MDACP (Molecular Dynamics code for Avogadro Challenge Project)

## Summary
MDACP (Molecular Dynamics code for Avogadro Challenge Project) is an
efficient implementations of classical molecular dynamics (MD) method
for the Lennard-Jones particle systems.

This fork of MDACP is designated to support (1) the force calculation 
with GPU and MIC (2) pairlist construction using SIMD instructions. 

The latest information of original MDACP is available at
http://mdacp.sourceforge.net/

## How to compile
### CPU only with AVX2

```sh
$ mkdir build
$ cd build
$ cmake ../ -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc -DUSE_AVX2=true
$ make
```

### GPU & CPU with AVX2 @ ISSP System B

```sh
$ source env/sekirei.sh
$ mkdir build
$ cd build
$ cmake ../ -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=gcc -DUSE_GPU_CUDA=true -DUSE_AVX2=true
$ make
```

### GPU & CPU with AVX2 @ Reedbush-L

```sh
$ source env/reedbush-l.sh
$ mkdir build
$ cd build
$ cmake ../ -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc -DUSE_GPU_CUDA=true -DUSE_AVX2=true
$ make
```

### MIC @ Oakforest-PACS

```sh
$ mkdir build
$ cd build
$ cmake ../ -DCMAKE_CXX_COMPILER=mpiicpc -DCMAKE_C_COMPILER=mpiicc -DUSE_AVX512=true
$ make
```

## Run

```
usage: mpiexec -np n ./mdacp [options] ...
options:
  -i, --in                      input file name (optional [default: input.cfg])
  -g, --num_gpus_per_node       number of gpus per one node (optional [default: # of GPUs available in one node])
  -p, --num_of_procs_per_gpu    number of processes per one gpu (mandatory when compiling with CUDA support)
  -?, --help                    print this message
```

### CPU only @ ISSP System B

```sh
$ mpijob ./mdacp -i input.cfg
```

### CPU + GPU @ ISSP System B

* Total number of MPI processes is 16.
* 8 MPI processes are assigned to each GPU.

```sh
$ mpijob ./mdacp -i input.cfg -p 8 -g 2
```

## List of Developers
- *Hiroshi Watanabe <hwatanabe@issp.u-tokyo.ac.jp>
    - Institute for Solid State Physics, University of Tokyo (Corresponding Author)

- Masaru Suzuki
    - Department of Applied Quantum Physics and Nuclear Engineering, Faculty of Engineering, Kyushu University

- Nobuyasu Ito
    - Department of Applied Physics, School of Engineering, The University of Tokyo
