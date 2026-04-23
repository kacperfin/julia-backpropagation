module CustomAwid

using LinearAlgebra

export Operator, Loss, Chain, Tensor, tensor, chain
export GraphNode, graph, forward!, backward!, optimize!, zerograd!

export Dense, dense, Conv, conv, MaxPool, maxpool, Flatten, flatten, Dropout, dropout
export ReLU, relu, Sigmoid, sigmoid
export BinaryCrossEntropy, bce, LogitCrossEntropy, logitcrossentropy

include("core.jl")
include("operators.jl")

end # module CustomAwid