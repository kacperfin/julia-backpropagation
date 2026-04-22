#= Przykład użycia
function train!(model, batches, dataset)
  x = zeros(28, 28, 1)
  y = zeros(10)
  L = 0.0
  for batch in eachcol(batches)
    for sample in batch
      x .= dataset.features[:, :, sample]
      y .= 0.0; y[dataset.targets[sample] + 1] = 1.0;
          
      zerograd!(model)
      forward!(model, input  => x, target => y)
      backward!(model)
      accumulate!(opt, model)
          
      L += model[end].data[1]
    end
	  optimize!(opt, model)
  end
  return L / length(batches)
end
=#

mutable struct GradientDescent
  ∇ :: Dict{GraphWeight, Array{Float64}}
  η :: Float64
  s :: Int64
  GradientDescent(η) = new(Dict(), η, 0)
end

function accumulate!(opt, graph)
  for node in graph
    if node isa GraphWeight
      if node in keys(opt.∇)
        opt.∇[node] .+= node.grad
      else
        opt.∇[node]   = node.grad
      end
    end
  end
  opt.s += 1
  return nothing
end

function step!(θ, α, ∇f)
  θ .-= α .* ∇f
  ∇f .= 0.0
  return nothing
end

function optimize!(opt, graph)
  for node in graph
    if node isa GraphWeight
      step!(node.data, opt.η / opt.s, opt.∇[node])
    end
  end
  opt.s = 0
  return nothing
end
