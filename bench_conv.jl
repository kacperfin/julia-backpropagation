# bench_conv.jl - izolowany pomiar splotu dla obu warstw sieci, w dwóch wersjach:
#   naiwny  - jądra z KM1 (_primal_conv!/_adjoint_conv!) = wzorzec odniesienia
#   turbo   - pętla bez warunków brzegowych + @turbo (wersja w sieci)
# Wersję @turbo porównujemy z naiwną (czy liczy to samo) i benchmarkujemy.
# Uruchomienie:  julia bench_conv.jl

include("CustomAwid/src/CustomAwid.jl")
using .CustomAwid
using BenchmarkTools
using Random
const CA = CustomAwid

Random.seed!(0)

struct Layer
  name :: String
  W_in :: Int; H_in :: Int; C_in :: Int; C_out :: Int
  F :: Int; pad :: Int; B :: Int
end

const LAYERS = [
  Layer("conv1 (28x28x1 -> 6)",  28, 28, 1,  6, 3, 1, 20),
  Layer("conv2 (14x14x6 -> 16)", 14, 14, 6, 16, 3, 1, 20),
]

function make_arrays(L::Layer)
  W_out = L.W_in + 2*L.pad - L.F + 1
  H_out = L.H_in + 2*L.pad - L.F + 1
  x  = randn(Float32, L.W_in, L.H_in, L.C_in, L.B)
  W  = randn(Float32, L.F, L.F, L.C_in, L.C_out) .* 0.1f0
  dy = randn(Float32, W_out, H_out, L.C_out, L.B)
  return x, W, dy
end

# Oracle: naiwny splot policzony niezależnie (wzorzec poprawności).
function oracle(L::Layer, x, W, dy)
  W_out = L.W_in + 2*L.pad - L.F + 1
  H_out = L.H_in + 2*L.pad - L.F + 1
  y  = zeros(Float32, W_out, H_out, L.C_out, L.B); CA._primal_conv!(y, W, x)
  Wg = zeros(Float32, size(W)); xg = zeros(Float32, size(x))
  CA._adjoint_conv!(dy, Wg, W, xg, x)
  return y, Wg, xg
end

# Buduje węzeł conv (@turbo), ustawia naszą wagę, robi forward+backward.
function run_turbo(L::Layer, x, W, dy)
  xt = CA.GraphNode(copy(x), false)
  cnode = conv((L.F, L.F), L.C_in => L.C_out, pad=L.pad)(xt)
  cnode.args[1].data .= W
  CA.primal!(cnode)
  cnode.grad .= dy
  for a in cnode.args; a isa CA.GraphNode && (a.grad .= 0); end
  CA.adjoint!(cnode)
  return cnode, xt
end

println("=========================================================")
println(" POMIAR SPLOTU — naiwny (oracle) vs @turbo (produkcyjny)")
println("=========================================================")

for L in LAYERS
  println("\n--- $(L.name) ---")
  x, W, dy = make_arrays(L)
  y_ref, Wg_ref, xg_ref = oracle(L, x, W, dy)
  Wg = zeros(Float32, size(W)); xg = zeros(Float32, size(x))

  print("  [naive] primal! : "); @btime CA._primal_conv!($(zeros(Float32, size(y_ref))), $W, $x)
  print("  [naive] adjoint!: "); @btime CA._adjoint_conv!($dy, $Wg, $W, $xg, $x)

  cnode, xt = run_turbo(L, x, W, dy)
  oky = isapprox(cnode.data, y_ref; rtol=1f-3)
  okW = isapprox(cnode.args[1].grad, Wg_ref; rtol=1f-3)
  okx = isapprox(xt.grad, xg_ref; rtol=1f-3)
  println("  [turbo] poprawność vs oracle: fwd=$oky dW=$okW dx=$okx")
  print("  [turbo] primal! : "); @btime CA.primal!($cnode)
  print("  [turbo] adjoint!: "); @btime CA.adjoint!($cnode)
end
