# train.jl - trenuje sieć na FashionMNIST przez 3 epoki i sprawdza dokładność testową.
# Bez Flux/CairoMakie - własne batchowanie, one-hot i onecold (tylko MLDatasets).
# Uruchomienie:
#   julia train.jl            # pojedyncze ziarno (0), z detalem epok
#   julia train.jl 0 1 2 3    # kilka ziaren -> średnia ± odchylenie dokładności

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"   # nie pytaj o pobranie zbioru danych

include("CustomAwid/src/CustomAwid.jl")
using .CustomAwid
using MLDatasets
using Random
using Statistics   # mean, std

# --- Parametry (stałe, narzucone treścią projektu) --------------------------
const epochs = 3
const η = 0.01
const batch_size = 20

# Ziarna z argumentów (domyślnie pojedyncze 0):
const seeds = isempty(ARGS) ? [0] : parse.(Int, ARGS)
const verbose = length(seeds) == 1   # detal epok tylko dla jednego ziarna

# --- Wczytanie danych (raz, niezależne od ziarna) ---------------------------
println("[x] Wczytywanie FashionMNIST...")
train_data = MLDatasets.FashionMNIST(split=:train)
test_data  = MLDatasets.FashionMNIST(split=:test)
train_x = reshape(train_data.features, 28, 28, 1, :)
test_x  = reshape(test_data.features,  28, 28, 1, :)

function onehot(targets)
  yhot = zeros(Float32, 10, length(targets))
  for (i, t) in enumerate(targets)
    yhot[t + 1, i] = 1.0f0
  end
  return yhot
end
train_yhot = onehot(train_data.targets)
test_yhot  = onehot(test_data.targets)

onecold(M) = [argmax(view(M, :, i)) - 1 for i in 1:size(M, 2)]

# --- Jedno uczenie dla danego ziarna -> (dokładność testowa, czas) ----------
function train_once(seed::Int)
  Random.seed!(seed)

  net = chain((
    conv((3, 3), 1 => 6, pad=1, bias=false),
    maxpool((2, 2)),
    conv((3, 3), 6 => 16, pad=1, bias=false),
    maxpool((2, 2)),
    flatten(),
    dense(784 => 84, relu),
    dropout(0.4),
    dense(84 => 10)
  ))
  input  = tensor(28, 28, 1, batch_size)
  target = tensor(10, batch_size)
  output = net(input)
  loss   = logitcrossentropy(output, target)
  model  = graph(loss)

  function eval_acc(X, Yhot)
    correct = 0; total = 0; N = size(X, 4)
    for start in 1:batch_size:(N - batch_size + 1)
      idx = start:(start + batch_size - 1)
      forward!(model, input => @view(X[:, :, :, idx]), target => @view(Yhot[:, idx]); train=false)
      pred  = onecold(output.data)
      truth = onecold(collect(@view Yhot[:, idx]))
      correct += sum(pred .== truth); total += length(truth)
    end
    return correct / total
  end

  N = size(train_x, 4)
  t = @elapsed for epoch in 1:epochs
    idxs = shuffle(1:N)
    for start in 1:batch_size:(N - batch_size + 1)
      idx = idxs[start:(start + batch_size - 1)]
      zerograd!(model)
      forward!(model, input => train_x[:, :, :, idx], target => train_yhot[:, idx])
      backward!(model)
      optimize!(model, η)
    end
    verbose && println("  epoka $epoch: ", round(eval_acc(test_x, test_yhot) * 100, digits=2), "%")
  end

  return eval_acc(test_x, test_yhot), t
end

# --- Pętla po ziarnach ------------------------------------------------------
println("[x] Uczenie ($epochs epoki, η=$η, batch=$batch_size), ziarna: $seeds\n")
accs = Float64[]
for s in seeds
  acc, t = train_once(s)
  push!(accs, acc * 100)
  println("ziarno $s: dokładność = ", round(acc * 100, digits=2), "%   (czas: ", round(t, digits=1), " s)")
end

if length(accs) > 1
  println("\nDokładność: średnia = ", round(mean(accs), digits=2),
          "% ± ", round(std(accs), digits=2), " pp  (n=$(length(accs)) ziaren)")
end
