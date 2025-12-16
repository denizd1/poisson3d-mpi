# 3D Poisson Equation Solver with MPI & Jacobi Iteration
A parallel and serial implementation of a 3D Poisson equation solver using finite differences and Jacobi iteration, with MPI-based domain decomposition and halo exchange for HPC environments.

This project implements a numerical solution of the **3D Poisson equation** using the **Finite Difference Method (FDM)** and **Jacobi Iteration**. It includes both a serial C implementation and a parallel MPI implementation suitable for HPC environments.

## Overview

The Poisson equation solved is:

$$
\nabla^2 u(x,y,z) = f(x,y,z)
$$

This project uses:

- **Discretization:** Finite Difference Method on a 3D Cartesian grid (e.g., $$(100\times 100\times 100\))$$.
- **Solver:** Jacobi iterative method.
- **Parallelization:** Domain decomposition with MPI using ghost (halo) layers for boundary exchange.

## Project Structure

- `jacobi3dserial.c` — Serial implementation (C)
- `jacobi3dmpi.c` — MPI-parallel implementation (C + MPI)
- `run_parallel_job.sbatch` — Example SLURM batch script for cluster runs

## Compilation & Usage

### Prerequisites

- GCC (serial build)
- OpenMPI or MPICH (MPI build)
- SLURM (optional, for cluster execution)

### 1) Serial Implementation

Compile and run:

```bash
gcc jacobi3dserial.c -o serial_poisson -lm
./serial_poisson
```

### 2) Parallel Implementation (MPI)

Compile:

```bash
mpicc jacobi3dmpi.c -o mpi_poisson -lm
```

Run locally (example with 4 processes):

```bash
mpirun -np 4 ./mpi_poisson
```

Run on a SLURM cluster (example):

```bash
sbatch run_parallel_job.sbatch
```

## Performance & Scaling (Example)

Performance is typically evaluated using:

- Speedup: $$(S(p)=T(1)/T(p)$$)
- Efficiency: $$(E(p)=S(p)/p\)$$

Example benchmark data (grid: $$(100^3$$), tolerance: $$(10^{-6}$$)):

| Processes | Time (s) |
|---:|---:|
| Serial | 505.33 |
| 2 | 449.83 |
| 4 | 119.05 |
| 8 | 92.53 |
| 12 | 82.53 |
| 14 | 337.90 |

Notes:
- Best performance was observed around **8–12 processes**.
- Beyond that, **communication overhead** may dominate for this problem size.

## Algorithm Details (MPI Version)

1. **Grid initialization / decomposition:** The 3D domain is split into subdomains; `MPI_Cart_create` can be used to set a 3D Cartesian topology.
2. **Halo exchange:** Neighbor boundary layers are exchanged via **non-blocking** communication (`MPI_Isend`, `MPI_Irecv`) to overlap communication and computation.
3. **Convergence:** A global stopping criterion is computed using `MPI_Allreduce` (e.g., global max error).

## References

- Selvadurai, A. P. S. (2000). *Poisson's equation*. In **Partial Differential Equations in Mechanics 2**. Springer.
- Langer, U., & Neumüller, M. (2018). *Direct and Iterative Solvers*. In **Computational Acoustics**. Springer.
