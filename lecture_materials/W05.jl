using BenchmarkTools
import MacroTools: postwalk, @capture

macro unroll(loop)
  @assert(loop.head == :for, "Only for-loops allowed")
  res = @capture(loop, for symbol_ = start_ : stop_ body__ end)
  @assert(res == true, "Couldn't match a for-loop")

  if (stop - start + 1) > 100
    println("[!] Too many iterations, keeping as-is")
    return esc(loop)
  else
    println("[*] Unrolling ", (stop - start + 1), " iterations")
  end

  code = Expr(:block)
  for it in start:stop
    for st in body
      expr = postwalk(e -> e == symbol ? it : e, st)
      push!(code.args, expr)
    end
  end
  return esc(code)
end

function foo()
  x = 0
  @unroll for i in 1:100
    x += i
  end
  return x
end

function bar()
  x = 0
  for i in 1:100
    x += i
  end
  return x
end

@btime foo()
@btime bar()

x = 0
@unroll for i=100:105
  x += 2i
  println(x, " ", i)
end
@show x

#########################3
J(::typeof(asin), x) = (asin(x), (Δ) -> tuple(Δ * inv(sqrt(1.0 - x^2))))
J(::typeof(sin),  x) = (sin(x),  (Δ) -> tuple(Δ * cos(x)))
J(::typeof(+), x, y) = (x + y,   (Δ) -> tuple(Δ, Δ))
J(::typeof(*), x, y) = (x * y,   (Δ) -> tuple(Δ * y, Δ * x))
J(::typeof(/), x, y) = (x / y,   (Δ) -> tuple(Δ * 1 / y, -Δ * a / y / y))
J(::typeof(^), x, y) = (x ^ 2,   (Δ) -> tuple(Δ * y * x ^ (y-1), nothing))

@show Meta.lower(Main, :(asin(0.2 + sin(x))))
# primal trace
x = 3;
a, B_a = J(sin, x);
b, B_b = J(+, 0.2, a);
c, B_c = J(asin, b)

@show c ≈ asin(0.2 + sin(x))

# adjoint trace
c̄,   = 1.0    # ∂c/∂c
b̄,   = B_c(c̄) # ∂c/∂b
_, ā = B_b(b̄) # ∂c/∂a
x̄,   = B_a(ā) # ∂c/∂x = ∂f/∂x

@show x̄ ≈ -1.0531613736418153

@show Meta.lower(Main, :(a / (a + b * b)))
# primal trace
a = 1.3
b = 2.5
y_1, B_1 = J(*, b, b)
y_2, B_2 = J(+, a, y_1)
y_3, B_3 = J(/, a, y_2)

@show y_3 ≈ a / (a + b^2)

# adjoint trace
ȳ_3 = 1.0
ā_1, ȳ_2 = B_3(ȳ_3)
ā_2, ȳ_1 = B_2(ȳ_2)
b̄_1, b̄_2 = B_1(ȳ_1)
ā = ā_1 + ā_2
b̄ = b̄_1 + b̄_1

@show ā ≈ 1 / y_2 - a / y_2^2
@show b̄ ≈ 2b * ȳ_2
