module CustomAwid

using LinearAlgebra
using Random   # Optymalizacja KM2 - rand! losuje maskę dropoutu w miejscu
using LoopVectorization   # Optymalizacja KM2 - @turbo wektoryzuje pętlę splotu

export Operator, Loss, Chain, Tensor, tensor, chain
export GraphNode, graph, forward!, backward!, optimize!, zerograd!

export Dense, dense, Conv, conv, MaxPool, maxpool, Flatten, flatten, Dropout, dropout
export ReLU, relu, Sigmoid, sigmoid, SoftMax, softmax
export BinaryCrossEntropy, bce, LogitCrossEntropy, logitcrossentropy

include("core.jl")
include("operators.jl")

end # module CustomAwid