# Add

# Jak w pliku wzorcowym
function primal!(z::GraphNode{:add, 2})
  x, y = z.args
  z.data .= x.data .+ y.data
  return nothing
end

# Pochodna z dodawania to 1, więc błąd z "z" jest przekazywany dalej
function adjoint!(z::GraphNode{:add, 2})
  x, y = z.args
  
  if length(size(x.data)) == 2 && length(size(y.data)) == 1
    # obsługa batchy: jeśli dodajemy wektor do batcha, to:
      x.grad .+= z.grad
      # do y (wektor) dodajemy gradient po zsumowaniu batchy
      y.grad .+= vec(sum(z.grad, dims=2))
  else
      # jeżeli nie mamy do czynienia z batchem, wykonaj klasyczne przeniesienie błędu
      # (zgodne z rozwiązaniem wzorcowym)
      x.grad .+= z.grad
      y.grad .+= z.grad
  end
  return nothing
end

# Sum

function primal!(y::GraphNode{:sum, 1})
  # Przecinek po x, ponieważ y.args jest krotką.
  # x, wyciąga pierwszą wartość w krotce.
  x, = y.args
  y.data = sum(x.data) # Zsumowanie podanego argumentu
  return nothing
end

function adjoint!(y::GraphNode{:sum, 1})
  # Podobnie jak w przypadku :add - gradient x jest taki sam jak y,
  # ponieważ pochodna dla każdego elementu w wektorze x.data wynosi 1
  x, = y.args
  x.grad += y.grad
  return nothing
end

# Mul

function primal!(y::GraphNode{:mul, 2})
  W, x = y.args
  y.data = W.data * x.data # wynik to mnożenie dwóch macierzy lub wektorów
  return nothing
end

function adjoint!(y::GraphNode{:mul, 2})
  W, x = y.args
  W.grad += y.grad * x.data' # grad W to grad y razy wartość drugiego czynnika mnożenia (x.data)
  # transpozycje wynikają z kolejności w mnożeniu macierzowym
  x.grad += W.data' * y.grad
  return nothing
end

# Dot

function primal!(z::GraphNode{:dot, 2})
  # Iloczyn skalarny dwóch wektorów
  x, y = z.args
  z.data = dot(x.data, y.data)
  return nothing
end

function adjoint!(z::GraphNode{:dot, 2})
  x, y = z.args
  # podobne do :mul, ale bez transpozycji macierzowych
  # z.grad to pojedyncza liczba, nie wektor
  x.grad += y.data .* z.grad
  y.grad += x.data .* z.grad 
  return nothing
end

# Dense

struct Dense <: Operator # Dense jest jednym z Operatorów
  insize  :: Int64 # l. neuronów wejściowych
  outsize :: Int64 # l. neuronów wyjściowych
end

dense(pair :: Pair{Int64, Int64}) =
  Dense(first(pair), last(pair)) # umożliwia tworzenie warstwy Dense za pomocą pary: np. dense(2 => 16) wykonuje Dense(2, 16)
dense(pair :: Pair{Int64, Int64}, activation) =
  tuple(dense(pair), activation()) # umożliwia tworzenie warstwy wraz z funkcją aktywacji, np. dense(2 => 16, relu)

function (y::Dense)(x)
  # y = W * x + b
  n   = y.insize # l. neuronów wejściowych
  m   = y.outsize # l. neuronów wyjściowych
  B   = size(x.data, 2)  # 2. wymiar podanych danych
  W   = GraphNode(randn(m, n) .* 0.01, true) # warstwa wag
  b   = GraphNode(randn(m) .* 0.01, true) # warstwa biasów (też :weight)
  mul = GraphNode(:mul, (W, x), zeros(m, B)) # mnożenie Wx
  add = GraphNode(:add, (mul, b), zeros(m, B)) # dodawanie Wx + b
  return add
end

# Sigmoid (funkcja aktywacji)

struct Sigmoid <: Operator end 
sigmoid() = Sigmoid() 

function (y::Sigmoid)(x)
  return GraphNode(:sigmoid, (x,), zeros(size(x.data)))
end

function primal!(y::GraphNode{:sigmoid, 1})
  # sigmoid(x) = 1 / (1 + exp(-x))
  # zwraca wartości od 0 do 1
  x, = y.args
  y.data .= 1 ./ (1 .+ exp.(-x.data))
  return nothing
end

function adjoint!(y::GraphNode{:sigmoid, 1})
  x, = y.args
  x.grad .+= exp.(-x.data) ./ (1 .+ exp.(-x.data)) .^ 2 .* y.grad
  return nothing
end

# ReLU (funkcja aktywacji)

struct ReLU <: Operator end 
relu() = ReLU() 

function (y::ReLU)(x)
  return GraphNode(:relu, (x,), zeros(size(x.data)))
end

function primal!(y::GraphNode{:relu, 1})
  # max(0, x) w formie macierzowej
  x, = y.args
  y.data .= max.(0, x.data)
  return nothing
end

function adjoint!(y::GraphNode{:relu, 1})
  # (x.data .> 0) zwróci 1, gdy dana liczba w x.data jest większa od 0. Wtedy pochodna wynosi 1
  # i błąd jest bezpośrednio propagowany. Jeżeli 0, to błąd wynosi 0.
  x, = y.args
  x.grad .+= (x.data .> 0) .* y.grad 
  return nothing
end

# BinaryCrossEntropy (BCE)

struct BinaryCrossEntropy <: Loss end # BinaryCrossEntropy należy do Loss
bce(output, target) = BinaryCrossEntropy()(output, target) # skrócenie zapisu

function (E::BinaryCrossEntropy)(x, y)
  return GraphNode(:bce, (x, y), zeros(1))
end

function primal!(z::GraphNode{:bce, 2})
  # L = -(y log(x)) + (1 - y) * log(1 - x)
  # x - output, y - target
  x, y = z.args
  z.data = -(y.data .* log.(x.data) + (1 .- y.data) .* log.(1 .- x.data))
  return nothing
end

function adjoint!(z::GraphNode{:bce, 2})
  # pochodna liczona po x (output). Po y (target) nie potrzeba, ponieważ target
  # się nie zmienia
  x, y = z.args
  x.grad -= y.data ./ x.data .* z.grad
  x.grad += (1 .- y.data) ./ (1 .- x.data) .* z.grad
  return nothing
end

# --------------------------Nowe--------------------------
# Flatten

struct Flatten <: Operator end
flatten() = Flatten()

function (y::Flatten)(x)
  features = prod(size(x.data)[1:end-1]) # wylicza liczbę komórek poza ostatnim wymiarem (batch)
  B = size(x.data)[end] # B = batchsize
  return GraphNode(:flatten, (x,), zeros(features, B))
end

function primal!(y::GraphNode{:flatten, 1})
  # przypisanie danych w innym formacie
  x, = y.args
  B = size(x.data)[end]
  # batchsize zachowane, długość pierwszego wymiaru dopasowana do utworzonej struktury
  y.data .= reshape(x.data, :, B) 
  return nothing
end

function adjoint!(y::GraphNode{:flatten, 1})
  x, = y.args
  # gradient dodawany do macierzy w formacie przed flatten
  x.grad .+= reshape(y.grad, size(x.grad))
  return nothing
end

# Conv

struct Conv <: Operator
  filter_size :: NTuple{2, Int64}
  in_channels :: Int64
  out_channels :: Int64
  pad :: Int64
end

function conv(filter_size::NTuple{2, Int64}, channels::Pair{Int64, Int64}; pad=0, bias=false)
  return Conv(filter_size, first(channels), last(channels), pad)
end

function (y::Conv)(x)
  W_in, H_in, _, B = size(x.data) # szerokość, wysokość, _, i batchsize
  # padding dodany po obu stronach - lewej i prawej, odjąć rozmiar filtra
  W_out = W_in + 2*y.pad - y.filter_size[1] + 1 # szer. wyjściowa
  H_out = H_in + 2*y.pad - y.filter_size[2] + 1 # wys. wyjściowa
  
  # Warstwa z wagami w kształcie: szer. filtra, wys. filtra, kanały wejściowe, kanały wyjściowe
  W_shape = (y.filter_size[1], y.filter_size[2], y.in_channels, y.out_channels)
  W = GraphNode(randn(Float32, W_shape) .* 0.1, true) # losowo inicjalizowane wagi filtra
  
  return GraphNode(:conv, (W, x), zeros(Float32, W_out, H_out, y.out_channels, B))
end

function primal!(y::GraphNode{:conv, 2})
  W, x = y.args
  _primal_conv!(y.data, W.data, x.data)
  return nothing
end

function _primal_conv!(y_data::AbstractArray{T1, 4}, W_data::AbstractArray{T2, 4}, x_data::AbstractArray{T3, 4}) where {T1, T2, T3}
  F_w, F_h, C_in, C_out = size(W_data) # wymiary filtra
  W_in, H_in, _, B = size(x_data) # wymiary oryginalnego obrazu
  W_out, H_out, _, _ = size(y_data) # wymiary wyjścia

  y_data .= 0 # dane wyjścia = 0

  pad_w = (W_out - W_in + F_w - 1) ÷ 2
  pad_h = (H_out - H_in + F_h - 1) ÷ 2

  # inbounds wyłącza sprawdzenie, czy indeksy wychodzą poza pętlę
  @inbounds for b in 1:B # dla każdego obrazka w batchu
    for c_out in 1:C_out # dla każdego filtra (kanału wyjściowego)
      for c_in in 1:C_in # dla każdego kanału wejściowego
        for j in 1:H_out # dla każdego piksela obrazka wyjściowego
          for i in 1:W_out # -||-
             sum_val = zero(T1) # zero
             for dj in 1:F_h # dla pikseli filtra
                for di in 1:F_w # -||-
                    xi = i + di - 1 - pad_w # współrzędne obrazka oryginalnego 
                    xj = j + dj - 1 - pad_h # -||-
                    if 1 <= xi <= W_in && 1 <= xj <= H_in # jeżeli współrzędne nie wychodzą poza obrazek
                        # piksel jest pikselem oryginalnym razy waga filtra
                        sum_val += x_data[xi, xj, c_in, b] * W_data[di, dj, c_in, c_out]
                    end
                end
             end
             y_data[i, j, c_out, b] += sum_val # zapisanie wyniku w warstwie wyjściowej
          end
        end
      end
    end
  end
end

function adjoint!(y::GraphNode{:conv, 2})
  W, x = y.args
  _adjoint_conv!(y.grad, W.grad, W.data, x.grad, x.data)
  return nothing
end

function _adjoint_conv!(y_grad::AbstractArray{T1, 4}, W_grad::AbstractArray{T2, 4}, W_data::AbstractArray{T2, 4}, x_grad::AbstractArray{T3, 4}, x_data::AbstractArray{T3, 4}) where {T1, T2, T3}
  F_w, F_h, C_in, C_out = size(W_data)
  W_in, H_in, _, B = size(x_data)
  W_out, H_out, _, _ = size(y_grad)

  pad_w = (W_out - W_in + F_w - 1) ÷ 2
  pad_h = (H_out - H_in + F_h - 1) ÷ 2

  @inbounds for b in 1:B
    for c_out in 1:C_out
      for c_in in 1:C_in
        for j in 1:H_out
          for i in 1:W_out
             dy = y_grad[i, j, c_out, b] # dy to wartość pochodnej dla każdego piksela obrazka wyjściowego
             if dy != zero(T1)
                 for dj in 1:F_h
                    for di in 1:F_w
                        xi = i + di - 1 - pad_w
                        xj = j + dj - 1 - pad_h
                        if 1 <= xi <= W_in && 1 <= xj <= H_in
                            # klasyczna pochodna z mnożenia
                            W_grad[di, dj, c_in, c_out] += x_data[xi, xj, c_in, b] * dy
                            x_grad[xi, xj, c_in, b] += W_data[di, dj, c_in, c_out] * dy
                        end
                    end
                 end
             end
          end
        end
      end
    end
  end
end

# MaxPool

struct MaxPool <: Operator
  window_size :: NTuple{2, Int64}
end

maxpool(window_size::NTuple{2, Int64}) = MaxPool(window_size)

function (m::MaxPool)(x)
  # MaxPool zmniejsza rozdzielczość
  W_in, H_in, C, B = size(x.data)
  W_out = W_in ÷ m.window_size[1] # przez ile podzielić
  H_out = H_in ÷ m.window_size[2] # -||-
  
  return GraphNode(:maxpool, (x,), zeros(Float32, W_out, H_out, C, B))
end

function primal!(y::GraphNode{:maxpool, 1})
  x, = y.args
  _primal_maxpool!(y.data, x.data)
  return nothing
end

function _primal_maxpool!(y_data::AbstractArray{T1, 4}, x_data::AbstractArray{T2, 4}) where {T1, T2}
  W_out, H_out, C, B = size(y_data)
  w_w = size(x_data, 1) ÷ W_out
  w_h = size(x_data, 2) ÷ H_out

  @inbounds for b in 1:B # dla każdego obrazka w batchu
    for c in 1:C # dla każdego kanału
      for j in 1:H_out # dla każdego piksela wyjściowego
        for i in 1:W_out # -||-
          max_val = typemin(T2) # zwraca najmniejszą możliwą wartość dla danego typu danych
          for dj in 1:w_h # dla każdej komórki okna (np. 2x2)
              for di in 1:w_w # -||-
                val = x_data[(i-1)*w_w + di, (j-1)*w_h + dj, c, b]
                  # wybierz max. wartość w oknie
                  if val > max_val
                      max_val = val
                  end
              end
          end
          # przypisz wartość
          y_data[i, j, c, b] = max_val
        end
      end
    end
  end
end

function adjoint!(y::GraphNode{:maxpool, 1})
  x, = y.args
  _adjoint_maxpool!(y.grad, y.data, x.grad, x.data)
  return nothing
end

function _adjoint_maxpool!(y_grad::AbstractArray{T1, 4}, y_data::AbstractArray{T1, 4}, x_grad::AbstractArray{T2, 4}, x_data::AbstractArray{T2, 4}) where {T1, T2}
  W_out, H_out, C, B = size(y_data)
  w_w = size(x_data, 1) ÷ W_out
  w_h = size(x_data, 2) ÷ H_out

  @inbounds for b in 1:B
    for c in 1:C
      for j in 1:H_out
        for i in 1:W_out
          max_val = y_data[i, j, c, b]
          dy = y_grad[i, j, c, b]
          
          for dj in 1:w_h
            for di in 1:w_w
              xi = (i-1)*w_w + di
              xj = (j-1)*w_h + dj
              # pochodna przepisywana z kolejnej warstwy tylko dla max. piksela
              if x_data[xi, xj, c, b] == max_val
                x_grad[xi, xj, c, b] += dy
              end
            end
          end
        end
      end
    end
  end
end

# Dropout

struct Dropout <: Operator
  p :: Float64
end

dropout(p::Float64) = Dropout(p)

function (y::Dropout)(x)
  p_node = GraphNode([y.p], false) # prawdopodobieństwo wyłączenia
  mask_node = GraphNode(zeros(size(x.data)), false) # pusta maska do przechowywania wyłączonych komórek
  return GraphNode(:dropout, (p_node, mask_node, x), zeros(size(x.data)))
end

function primal!(y::GraphNode{:dropout, 3}, train::Bool)
  p_node, mask_node, x = y.args
  if train
      p = p_node.data[1] # pobierz prawdopodobieństwo
      mask_node.data .= rand(size(x.data)...) .> p # zamiana maski na binarną 
      # Przemnożenie przez maskę (wyłączenie niektórych pikseli) oraz przeskalowanie pozostałych
      # o odwrotność prawdopodobieństwa zachowania neuronu:
      y.data .= (x.data .* mask_node.data) ./ (1 - p)
  else
      y.data .= x.data # w przypadku trybu ewaluacji, wszystkie neurony pozostają włączone
  end
  return nothing
end

function adjoint!(y::GraphNode{:dropout, 3})
  p_node, mask_node, x = y.args
  p = p_node.data[1]
  # Gradient propagowany tylko dla włączonych neuronów:
  x.grad .+= y.grad .* (mask_node.data ./ (1 - p)) 
  return nothing
end

# Softmax

struct SoftMax <: Operator end 
softmax() = SoftMax() 

function (y::SoftMax)(x)
  return GraphNode(:softmax, (x,), zeros(size(x.data)))
end

function primal!(y::GraphNode{:softmax, 1})
  # Softmax sprawia, że wszystkie wartości sumują się do 1
  # Wzór: e ^ (x_i) / sum(e ^ (x_j))
  x, = y.args
  xmax = maximum(x.data, dims=1) # znajduje największą wartość
  exp_x = exp.(x.data .- xmax) # odejmuje ją od każdej wartości (aby e ^ (x_i) nie było bardzo duże)
  y.data .= exp_x ./ sum(exp_x, dims=1)
  return nothing
end

function adjoint!(y::GraphNode{:softmax, 1})
  # dx_i = y_i * (dy_i - sigma(y_j * dy_j))
  x, = y.args
  sum_ydy = sum(y.data .* y.grad, dims=1) # sigma(y_j * dy_j)
  x.grad .+= y.data .* (y.grad .- sum_ydy)
  return nothing
end

# LogitCrossEntropy

struct LogitCrossEntropy <: Loss end
logitcrossentropy(output, target) = LogitCrossEntropy()(output, target)

function (E::LogitCrossEntropy)(x, y)
  return GraphNode(:crossentropy, (x, y), zeros(1)) # zwraca tylko jedną liczbę
end

function primal!(z::GraphNode{:crossentropy, 2})
  # Loss = -sigma(y * log(wektor prawdopodobieństw z Softmaxa))
  x, y = z.args
  # x - output
  # y - target (w postaci one hot batch)
  B = size(x.data, 2) # B - batchsize
  xmax = maximum(x.data, dims=1) 
  lse = xmax .+ log.(sum(exp.(x.data .- xmax), dims=1))
  log_probs = x.data .- lse
  
  z.data[1] = -sum(y.data .* log_probs) / B
  return nothing
end

primal!(node::GraphNode, train::Bool) = primal!(node)

function adjoint!(z::GraphNode{:crossentropy, 2})
  x, y = z.args
  B = size(x.data, 2)
  xmax = maximum(x.data, dims=1)
  probs = exp.(x.data .- xmax) ./ sum(exp.(x.data .- xmax), dims=1) # odtworzenie Softmaxa
  
  # softmax(output) - target / batch * pochodna już obecna w Lossie
  # (najczęściej loss jest końcem, więc 1):
  x.grad .+= ((probs .- y.data) ./ B) .* z.grad[1] 
  return nothing
end