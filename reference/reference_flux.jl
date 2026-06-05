# reference_flux.jl - punkt odniesienia w Flux (ta sama Julia co nasza biblioteka).
# Sieć i hiperparametry jak AWID-2026-CNN.ipynb (prowadzącego) / nasz CustomAwid:
#   Conv(1=>6,3x3,pad=1,bias=false) -> MaxPool(2)
#   Conv(6=>16,3x3,pad=1,bias=false) -> MaxPool(2)
#   flatten -> Dense(784=>84,relu) -> Dropout(0.4) -> Dense(84=>10)
# 3 epoki, η=0.01, Descent (SGD), batch=20, logitcrossentropy.
#
# Flux to Julia, więc porównujemy NIE tylko czas/dokładność, ale i ALOKACJE/krok
# (z PyTorch się nie dało - inny runtime). Mierzymy po rozgrzewce (warmup), bo
# pierwszy przebieg to kompilacja JIT (jak po naszej stronie).
#
# Uruchomienie:
#   julia reference/reference_flux.jl           # ziarno 0
#   julia reference/reference_flux.jl 0 1 2 3    # kilka ziaren -> średnia ± odch.

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using Flux
using MLDatasets
using Random
using Statistics
using LinearAlgebra   # BLAS.get_num_threads

# Pinujemy wątki BLAS, żeby porównanie czasu z naszą biblioteką było uczciwe.
# Flux wymusza BLAS=Sys.CPU_THREADS (ignoruje OPENBLAS_NUM_THREADS), a nasz
# train.jl domyślnie używa tylu, ile rdzeni fizycznych (6) - ustawiamy to samo.
BLAS.set_num_threads(6)

const epochs = 3
const η = 0.01
const batch_size = 20

# Ziarna z argumentów (domyślnie pojedyncze 0):
const seeds = isempty(ARGS) ? [0] : parse.(Int, ARGS)

# --- Dane (FashionMNIST), wczytane raz, jak w naszym train.jl -----------------
train_data = MLDatasets.FashionMNIST(split=:train)
test_data  = MLDatasets.FashionMNIST(split=:test)
train_x = reshape(train_data.features, 28, 28, 1, :)
test_x  = reshape(test_data.features,  28, 28, 1, :)
train_y = Flux.onehotbatch(train_data.targets, 0:9)
test_y  = Flux.onehotbatch(test_data.targets,  0:9)

# --- Budowa sieci (Chain jak u prowadzącego) ---------------------------------
# Inicjalizacja DOMYŚLNA Fluxa (Glorot) - my mamy He. Dla czasu/pamięci bez
# znaczenia; ewentualna różnica dokładności bierze się z innego startu, nie z
# jakości implementacji (tak samo jak przy PyTorch).
function build_model()
  Chain(
    Conv((3, 3), 1 => 6, pad=1, bias=false),
    MaxPool((2, 2)),
    Conv((3, 3), 6 => 16, pad=1, bias=false),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(784 => 84, relu),
    Dropout(0.4),
    Dense(84 => 10),
  )
end

# Dokładność testowa (model w trybie testowym = dropout wyłączony).
function eval_acc(net)
  Flux.testmode!(net)
  correct = 0; total = 0; N = size(test_x, 4)
  for s in 1:1000:N
    idx = s:min(s + 999, N)
    pred  = Flux.onecold(net(test_x[:, :, :, idx]))
    truth = Flux.onecold(test_y[:, idx])
    correct += sum(pred .== truth); total += length(truth)
  end
  Flux.trainmode!(net)
  return correct / total
end

# Jedno uczenie dla danego ziarna -> (dokładność testowa, czas 3 epok).
function train_once(seed::Int)
  Random.seed!(seed)
  net = build_model()
  opt_state = Flux.setup(Descent(η), net)
  loader = Flux.DataLoader((train_x, train_y); batchsize=batch_size, shuffle=true, partial=false)

  Flux.trainmode!(net)
  t = @elapsed for _ in 1:epochs
    for (x, y) in loader
      grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), net)
      Flux.update!(opt_state, net, grads[1])
    end
  end
  return eval_acc(net), t
end

# Alokacje JEDNEGO kroku uczenia (forward+backward+update) - jak nasz bench.jl.
function alloc_per_step()
  Random.seed!(0)
  net = build_model(); Flux.trainmode!(net)
  opt_state = Flux.setup(Descent(η), net)
  x = train_x[:, :, :, 1:batch_size]; y = train_y[:, 1:batch_size]
  step!() = (g = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), net); Flux.update!(opt_state, net, g[1]))
  step!()                       # rozgrzewka (kompilacja)
  return @allocated step!()     # bajty jednego kroku po rozgrzewce
end

# --- Rozgrzewka: skompiluj wszystkie ścieżki ZANIM cokolwiek mierzymy ---------
let
  net = build_model(); Flux.trainmode!(net)
  opt_state = Flux.setup(Descent(η), net)
  x = train_x[:, :, :, 1:batch_size]; y = train_y[:, 1:batch_size]
  g = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), net)
  Flux.update!(opt_state, net, g[1])
  eval_acc(net)
end

# --- Pętla po ziarnach -------------------------------------------------------
println("[ref Flux] wątki BLAS=$(BLAS.get_num_threads()), Julia=$(Threads.nthreads()), ",
        "epoki=$epochs, η=$η, batch=$batch_size, ziarna=$seeds\n")

accs = Float64[]
for s in seeds
  acc, t = train_once(s)
  push!(accs, acc * 100)
  println("ziarno $s: dokładność = ", round(acc * 100, digits=2), "%   (czas 3 epok: ", round(t, digits=1), " s)")
end

if length(accs) > 1
  println("\nDokładność: średnia = ", round(mean(accs), digits=2),
          "% ± ", round(std(accs), digits=2), " pp  (n=$(length(accs)) ziaren)")
end

println("Alokacje / krok uczenia: ", round(alloc_per_step() / 1024, digits=1), " KiB")
