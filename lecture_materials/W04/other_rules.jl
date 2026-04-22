σ(x) = BroadcastedOperator(σ, x)
forward(::BroadcastedOperator{typeof(σ)}, x) = return 1.0 ./ (1.0 .+ exp.(-x))
backward(node::BroadcastedOperator{typeof(σ)}, x, g) = let
    y = node.output
    𝟏 = ones(length(y))
    J = diagm(y .* (1.0 .- y))
    tuple(J' * g)
end

Base.Broadcast.broadcasted(^, x::GraphNode, y::GraphNode) = BroadcastedOperator(^, x, y)
forward(::BroadcastedOperator{typeof(^)}, x, y) = return x .^ y
backward(node::BroadcastedOperator{typeof(^)}, x, y, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(y .* x .^ (y .- 1.0))
    Jy = diagm(log.(abs.(x)) .* x .^ y)
    tuple(Jx' * g, Jy' * g)
end

Base.Broadcast.broadcasted(exp, x::GraphNode) = BroadcastedOperator(exp, x)
forward(::BroadcastedOperator{typeof(exp)}, x) = return exp.(x)
backward(node::BroadcastedOperator{typeof(exp)}, x, g) = let
    y = node.output
    J = diagm(y)
    tuple(J' * g)
end

Base.Broadcast.broadcasted(log, x::GraphNode) = BroadcastedOperator(log, x)
forward(::BroadcastedOperator{typeof(log)}, x) = return log.(x)
backward(::BroadcastedOperator{typeof(log)}, x, g) = tuple(diagm(1.0 ./ x)' * g)