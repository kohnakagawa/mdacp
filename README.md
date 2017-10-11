# MDACP (Molecular Dynamics code for Avogadro Challenge Project)

## Summary
MDACP (Molecular Dynamics code for Avogadro Challenge Project) is an
efficient implementations of classical molecular dynamics (MD) method
for the Lennard-Jones particle systems.

## Usage
### CPU only with AVX2

```sh
$ mkdir build
$ cd build
$ cmake ../ -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc -DUSE_AVX2=true
$ make
```

### GPU & CPU with AVX2 @ sekirei

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

### MIC

```sh
$ mkdir build
$ cd build
$ cmake ../ -DCMAKE_CXX_COMPILER=mpiicpc -DCMAKE_C_COMPILER=mpiicc -DUSE_AVX512=true
$ make
```

The latest information of MDACP is available at
http://mdacp.sourceforge.net/

## List of Developers
- *Hiroshi Watanabe <hwatanabe@issp.u-tokyo.ac.jp>
    - Institute for Solid State Physics, University of Tokyo (Corresponding Author)

- Masaru Suzuki
    - Department of Applied Quantum Physics and Nuclear Engineering, Faculty of Engineering, Kyushu University

- Nobuyasu Ito
    - Department of Applied Physics, School of Engineering, The University of Tokyo
