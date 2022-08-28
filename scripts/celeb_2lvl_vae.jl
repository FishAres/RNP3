using DrWatson
@quickactivate "RNP3"
ENV["GKSwstype"] = "nul"

using PyCall, Pkg
ENV["PYTHON"] = "/mmfs1/gscratch/rao/aresf/miniconda3/bin/python"
Pkg.build("PyCall")

using MLDatasets
using Flux, Zygote, CUDA
using IterTools: partition, iterated
using Flux: batch, unsqueeze, flatten
using Flux.Data: DataLoader
using Distributions
using StatsBase: sample
using Random: shuffle
using ParameterSchedulers

# todo - refactor so scripts share same code
include(srcdir("double_H_vae_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, :img_size => (28, 28), :π => 32,
    :esz => 32, :add_offset => true, :fa_out => identity, :f_z => elu,
    :asz => 6, :glimpse_len => 4, :seqlen => 5, :λ => 1.0f-3, :δL => Float32(1 / 4),
    :scale_offset => 2.8f0, :scale_offset_sense => 3.2f0,
    :λf => 0.167f0, :D => Normal(0.0f0, 1.0f0),
)

## =====

device!(2)

dev = gpu

##=====

datasets = pyimport("torchvision.datasets")
dataset = datasets.CelebA(root="/mmfs1/gscratch/rao/aresf/datasets/celeba", download=true)