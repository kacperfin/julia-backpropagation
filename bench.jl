# bench.jl - benchmark jednego kroku uczenia (czas i alokacje) bieżącej biblioteki.
# include ładuje to, co aktualnie jest w katalogu, więc before/after robimy tak:
#   git checkout 0c63b7f -- CustomAwid/src   # KM1 (przed optymalizacjami)
#   julia bench.jl                           # -> liczby KM1
#   git checkout HEAD -- CustomAwid/src      # KM2 (po optymalizacjach)
#   julia bench.jl                           # -> liczby KM2
# Metodyka (@btime, @code_warntype) z wykładu W08.
# Uruchomienie:  julia bench.jl

include("CustomAwid/src/CustomAwid.jl")
using .CustomAwid

using BenchmarkTools   # @btime — pomiar czasu i alokacji (z wykładu W08)
using InteractiveUtils # @code_warntype — wykrywanie niestabilności typów (W08)
using Random           # losowy batch (powtarzalny dzięki seedowi)

Random.seed!(0)

const batch_size = 20
const η = 0.01

# --- Ta sama architektura co w AD.ipynb (KM1) -------------------------------
# Nie pobieramy FashionMNIST — używamy LOSOWEGO batcha, żeby odizolować
# wydajność samej biblioteki (a nie ładowania danych).
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

input  = tensor(28, 28, 1, batch_size)  # węzeł wejściowy (obraz)
target = tensor(10, batch_size)          # węzeł celu (etykieta one-hot)
output = net(input)                      # logity (przedostatni węzeł)
loss   = logitcrossentropy(output, target)  # strata (ostatni węzeł)
model  = graph(loss)                     # posortowany graf obliczeniowy

# --- Losowy batch: obraz + etykiety one-hot ---------------------------------
x = randn(Float32, 28, 28, 1, batch_size)   # losowy "obraz"
y = zeros(Float32, 10, batch_size)          # etykiety one-hot
for b in 1:batch_size
    y[rand(1:10), b] = 1.0f0                 # jedna klasa = 1 w każdej kolumnie
end

# --- Jeden pełny krok uczenia (te same 4 kroki co w AD.ipynb) ----------------
function train_step!(model, input, target, x, y, η)
    zerograd!(model)                              # 1. wyzeruj gradienty
    forward!(model, input => x, target => y)      # 2. w przód
    backward!(model)                              # 3. w tył (gradienty)
    optimize!(model, η)                           # 4. popraw wagi
end

# --- Rozgrzewka: pierwszy przebieg KOMPILUJE kod (nie mierzymy go) -----------
train_step!(model, input, target, x, y, η)

# --- Pomiary -----------------------------------------------------------------
# $ przed zmienną mówi BenchmarkTools, żeby ją wstawić jako stałą (bez narzutu
# zmiennych globalnych) — standard z wykładu W08.
println("=========================================================")
println(" BENCHMARK kroku uczenia (batch_size = $batch_size)")
println("=========================================================")

print("forward!     : ");  @btime forward!($model, $input => $x, $target => $y)
print("backward!    : ");  @btime backward!($model)
print("krok uczenia : ");  @btime train_step!($model, $input, $target, $x, $y, $η)

# Szacunkowy czas 1 epoki (60 000 obrazów / batch_size kroków):
t = @belapsed train_step!($model, $input, $target, $x, $y, $η)
steps_per_epoch = 60_000 ÷ batch_size
println("\nSzac. czas 1 epoki: ", round(t * steps_per_epoch, digits=1), " s",
        "  (limit: < 300 s/epokę)")
println("Szac. czas 3 epok : ", round(t * steps_per_epoch * 3, digits=1), " s",
        "  (limit: < 900 s = 15 min)")
