# MDACP (Molecular Dynamics code for Avogadro Challenge Project)

## Summary
MDACP (Molecular Dynamics code for Avogadro Challenge Project) is an
efficient implementations of classical molecular dynamics (MD) method
for the Lennard-Jones particle systems.

## Usage
### CPU only

```sh
$ mkdir build
$ cd build
$ cmake ../ -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc -DUSE_GPU_CUDA=false
$ make
```

### CUDA support

```sh
$ mkdir build
$ cd build
$ cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DUSE_GPU_CUDA=true
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

