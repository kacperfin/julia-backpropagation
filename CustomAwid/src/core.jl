using LinearAlgebra

abstract type Operator end # abstrakcyjny typ Operator
const Chain = Vector{Operator} # alias Chain do Vector{Operator}
abstract type Loss end # abstrakcyjny typ Loss

# Tensor
struct Tensor{N}
  # struktura Tensor z parametrem N, który jest przekazywany do NTuple. Czyli Tensor N
  # składa się z atrybutu outsize, który jest typu NTuple o długości N i każdy z jego
  # elementów ma typ Int64 
  outsize :: NTuple{N, Int64}
end
  tensor(sz...) = Tensor(sz)()
# ... to tzw. splat operator, jak *args w Pythonie. Można wtedy pisać tensor(2, 3, 4), co spowoduje spakowanie
# tych wartości do krotki: sz = (2, 3, 4) i uruchomienie konstruktora Tensor(sz). Krotka sz zostanie przekształcona
# na NTuple{N, Int64} automatycznie, aby dopasować się do definicji atrybutu "outsize" Tensora.

function (x::Tensor{N})() where N
  data = zeros(x.outsize...)
  return GraphNode(data)
end

function chain(operators)
# chain to funkcja tworząca wektor operatorów. Wejściem jest wektor składający się z pojedynczych
# operatorów lub krotek operatorów - w tym drugim przypadku są one odpakowywane
  function flatten(x::Tuple)
    y = Vector{Operator}()
    for v in x
      if v isa Tuple
        push!(y, v...)
      else
        push!(y, v)
      end
      end
    return y
  end

  result = Vector{Operator}()
  for operator in flatten(operators)
    push!(result, operator)
  end
  return result
end

function (chain::Chain)(x)
# (chain::Chain) oznacza, że jest to funkcja nieprzypisana do nazwy, a do każdego z obiektów
# typu Chain. x to argument.
  node = x # wartość początkowa
  for op in chain
    node = op(node) # dla każdej operacji w chainie (wektorze operacji) wykonaj tę operację na wartości node.
  end
  return node # zwróć uzyskaną wartość
end
mutable struct GraphNode{OP, N}
  args :: NTuple{N, GraphNode}
  grad
  data
end

const GraphWeight = GraphNode{:weight, 0} # Węzeł bez argumentów gdzie OP = :weight
const GraphTensor = GraphNode{:tensor, 0} # Węzeł bez argumentów gdzie OP = :tensor

# Konstruktor dla danych jak wagi i tensory (obrazy), czyli nieposiadających argumentów (rodziców):
function GraphNode(data::T, trainable=false) where T
  if trainable # wagi są trainable
    return GraphNode{:weight, 0}((), zero(data), data)
  else # tensory nie są trainable
    return GraphNode{:tensor, 0}((), zero(data), data)
  end
end

# Konstruktor dla operatorów-symboli, czyli operatorów, które zaczynają się od :, np. :mul.
# Wymaga podania argumentów! (args)
function GraphNode(op::Symbol, args::Tuple, data::T) where T
  N = length(args)
  grad = similar(data)
  return GraphNode{op, N}(args, grad, data)
end

# Zwracanie wektora GraphNode'ów w kolejności:
function graph(node)
  function visit!(node::GraphNode, visited, ordered)
    if node in visited
    else # jeśli node nie był odwiedzony
      push!(visited, node) # dodaj node do odwiedzonych
      for arg in node.args # dla każdego argumentu w node (jeżeli są)
        visit!(arg, visited, ordered)
        # Odwiedź argumenty ZANIM dodasz obecny node do listy.
        # Gwarantuje to odpowiednią kolejność w grafie. 
      end
      push!(ordered, node)
    end
    return nothing
  end
  ordered = Vector{GraphNode}() # utwórz wektor gdzie każdy element to GraphNode
  visited = Set{GraphNode}() # utwórz wektor z odwiedzonymi GraphNode'ami
  visit!(node, visited, ordered) # odwiedź node i modyfikuj visited oraz ordered
  return ordered # zwróć uporządkowany wektor GraphNode'ów
end

# Funkcja do wyzerowywania gradientu w całym wektorze GraphNode'ów:
function zerograd!(order :: Vector{GraphNode})
  for node in order
    node.grad .= 0
  end
end

# Funkcje, które nic nie robią, ale zapobiegają błędowi kompilatora.
# primal! liczy wartość przejścia w przód - ale nie trzeba jej liczyć dla GraphTensor
# i GraphWeight, ponieważ te wartości są ustalone.
# adjoint! liczy gradient.
function primal!(tensor::GraphTensor) end
function primal!(weight::GraphWeight) end
function adjoint!(::GraphTensor) end
function adjoint!(::GraphWeight) end

# Funkcja forward! przyjmuje posortowany wektor GraphNode'ów oraz pary
# i inicjalizuje data w GraphNode'ach faktycznymi wartościami, a następnie
# przechodzi po Vector{GraphNode} i liczy wartości za pomocą funkcji określonych
# w sekcji Operatory.
function forward!(order::Vector{GraphNode}, pairs...; train=true)
  for pair in pairs
    tensor,data = pair
    tensor.data .= data
  end

  for node in order
    primal!(node, train)
  end
end

# Funkcja backward! liczy gradient.
function backward!(order::Vector{GraphNode})
	seed = last(order)
	seed.grad .= 1

  for node in reverse(order) # w przeciwnym kierunku niż forward!
    adjoint!(node)
  end
end

# Algorytm spadku gradientu
function optimize!(graph, η)
  for node in graph
	if node isa GraphWeight
      node.data .-= η * node.grad
    end
  end
end

# Wyświetlanie
import Base: show
show(io::IO, x::GraphNode{OP, N}) where {OP,N} =
  print(io, "layer ", OP, " with ", N, " arg(s)")
show(io::IO, x::GraphWeight) = print(io, "weight")
show(io::IO, x::GraphTensor) = print(io, "tensor")