#!/bin/sh

#SBATCH -o p2.log-%j
#SBATCH -n 10
#SBATCH -N 10

source /etc/profile
module load julia-1.0
julia p2.jl
