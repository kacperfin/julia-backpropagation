#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Optimization of a Julia Reverse-Mode Automatic Differentiation Library for CNN Training],
  abstract: [
  We present the optimization of `CustomAwid`, a from-scratch reverse-mode
  automatic differentiation (AD) library written in Julia. The library is used
  to train a small convolutional neural network (CNN) on Fashion-MNIST
  @xiao2017fashionmnist.
  Starting from a working baseline implementation, we apply a sequence of
  measurable optimizations: parametrizing the computation-graph node on its
  array type for type stability, converting operators to in-place array
  primitives, replacing the six-deep convolution loops with a branch-free,
  SIMD-vectorized variant, and running the entire network in single precision.
  Each step is verified with micro-benchmarks and a fixed numerical oracle to
  check that the results stay correct. The combined effect reduces per-step
  training time by $2.9 times$ (the forward pass alone by $17 times$) and heap
  allocations per step by $99%$, while test accuracy stays within the required
  range ($86.4 plus.minus 0.46%$ over four seeds). We also compare the final
  library against reference implementations with the same architecture. The
  final implementation remains competitive for a compact from-scratch library
  and allocates substantially less memory per step than Flux.
],
  authors: (
    (
      name: "Kacper Aleksander",
      department: [Faculty of Electrical Engineering],
      organization: [Warsaw University of Technology],
      location: [Warsaw, Poland],
    ),
    (
      name: "Michał Zdulski",
      department: [Faculty of Electrical Engineering],
      organization: [Warsaw University of Technology],
      location: [Warsaw, Poland],
    ),
  ),
  index-terms: ("Automatic differentiation", "Julia", "Performance optimization", "Convolutional neural networks", "SIMD vectorization"),
  figure-supplement: [Fig.],
)

#show raw.where(block: true): it => block(
  width: 100%,
  inset: (x: 5pt, y: 4pt),
  fill: rgb("#f6f6f6"),
  stroke: rgb("#d9d9d9"),
  radius: 2pt,
  text(size: 8.5pt, it),
)

= Introduction

Reverse-mode automatic differentiation (AD) is a standard tool used in
neural-network training @baydin2018ad: it computes the gradient of a scalar loss
with respect to every parameter in a single backward pass, regardless of the
number of parameters. This property is what makes it preferable to symbolic
differentiation, whose expressions may grow explosively, and to numerical
differentiation, which requires one extra forward evaluation per parameter.

Our library, `CustomAwid`, implements reverse-mode AD from scratch in Julia using
an explicit computation graph of nodes. It is used to train a fixed CNN
classifying Fashion-MNIST @xiao2017fashionmnist. The network is

```
conv(3x3, 1->6, pad=1) -> maxpool(2x2)
conv(3x3, 6->16, pad=1) -> maxpool(2x2)
flatten -> dense(784->84, relu)
dropout(0.4) -> dense(84->10) -> logit-cross-entropy
```

with fixed hyper-parameters: 3 epochs, learning rate $eta = 0.01$, plain SGD,
batch size 20, and 67,708 parameters. The baseline implementation was
intentionally simple and unoptimized; the goal of this work is to optimize this
working implementation for computational complexity, memory traffic, and code
efficiency without changing the architecture, the hyper-parameters, or the target
accuracy.

We measure each optimization separately. We use `BenchmarkTools.jl` for timing and
allocation counts, `@code_warntype` for type-stability inspection, and a fixed
naive convolution kept in the code base as a numerical oracle against which every
faster variant is checked. This also makes it easier to report cases where an
optimization improves code quality but does not improve runtime.

= Library Design

The graph is built from `GraphNode` objects. Each node stores its parent nodes
(`args`), its forward value (`data`), and its accumulated gradient (`grad`). The
operation is encoded as a `Symbol` type parameter, so that the forward (`primal!`)
and backward (`adjoint!`) methods are selected by Julia's multiple dispatch:

#block(breakable: false)[
```julia
mutable struct GraphNode{OP, N}
  args :: NTuple{N, GraphNode}
  grad
  data
end
```
]

A network is a chain of layers; calling it builds the graph. Training repeats four
steps per batch: `zerograd!`, `forward!` (topologically ordered `primal!` calls),
`backward!` (reverse-ordered `adjoint!` calls seeded with
$partial L / partial L = 1$, equivalently $bar(L) = 1$), and `optimize!` (SGD
update). Heavy layers --- convolution and max-pooling --- delegate to inner kernel
functions such as `_primal_conv!(y, W, x)`; this function-barrier pattern is
important for the first optimization.

= Optimizations

We describe each optimization and its measured effect. Timings are for one full
training step on a $28 times 28 times 1 times 20$ batch on a single CPU thread.

== Type stability

In the original code the `data` and `grad` fields are untyped, effectively `Any`,
so Julia cannot specialize the graph-level operators.

#pagebreak()

We parametrize the node on its array type `T`:

#block(breakable: false)[
```julia
mutable struct GraphNode{OP, N, T}
  args :: NTuple{N, GraphNode}
  grad :: T
  data :: T
end
```
]

`@code_warntype` confirms that the destination of each broadcast changes from
`Any` to a concrete type. However, the measured end-to-end speed-up is essentially
zero (Table~I). This result suggests that most expensive work was already behind
function barriers: convolution and pooling receive concretely typed kernel
arguments, so Julia can still compile specialized methods for the hot loops. The
dynamic dispatch happened once per kernel call, not inside the loops. We keep the
change because it makes the graph representation cleaner, but it is not the main
source of speed-up.

== In-place operators

Many operators allocated a fresh array every pass, e.g. `y.data = W.data * x.data`
and `W.grad += y.grad * x.data'` in the dense layer. We convert these to in-place
primitives. Matrix multiplication becomes `mul!`, whose five-argument form
$C arrow.l alpha A B + beta C$ accumulates the gradient without a temporary. In
the example below, `xT` and `WT` denote transposed views:

#block(breakable: false)[
```julia
mul!(y.data, W.data, x.data)
mul!(W.grad, y.grad, xT, true, true)
mul!(x.grad, WT, y.grad, true, true)
```
]

The dropout mask is sampled in place with `rand!`, and the SGD update uses a fused
broadcast (`.-=` with `.*`). The time per step is unchanged: the step is
compute-bound on convolution, so garbage-collection latency is minimal within a
single step. However, heap allocations per step fall from $1.84$~MiB to $13$~KiB
($-99%$, Table~I). Over a full epoch ($3000$ steps) this is the difference between
allocating approximately 5.4~GiB and approximately 38~MiB, sharply reducing GC
pressure.

== Convolution: branch-free loops with SIMD

The naive convolution is a six-deep loop nest with an interior bounds check for
padding. This branch prevents SIMD vectorization. We first copy the input into a
zero-padded buffer, allocated once and stored as an extra graph node, exactly as
the dropout mask is. This removes the branch; the now branch-free inner loop is
vectorized with `LoopVectorization.@turbo` @elrod2022loopvectorization. The
forward kernel becomes a clean reduction into a local scalar followed by a write
to a unique output cell, which is an ideal SIMD pattern.

In isolation (Table~II), this accelerates the forward convolution by $19 times$
and $30 times$ for the two layers. End-to-end (Table~I), the forward pass drops
from $10.33$~ms to $0.60$~ms ($17 times$) and the full step from $17.29$~ms to
$6.13$~ms ($2.8 times$). The backward convolution, however, is not vectorized:
it is a scatter that accumulates into shared filter-gradient cells and into
overlapping input-gradient cells, so neighbouring iterations write to the same
memory location. This creates a data hazard that SIMD cannot reorder safely. After
this step, the backward pass dominates the remaining cost.

== Single precision

The initial prototype left most arrays at the Julia default, `Float64`, even
though single precision is standard for neural-network training. We make the
entire graph `Float32`: the input tensor, all weights, every operator buffer, and
the gradients. This halves the memory footprint of every array, including weights,
activations, padded convolution buffers, and gradient tensors. It also lets the
dense layer's `mul!` dispatch to the single-precision BLAS kernel, while the
`@turbo` convolution packs twice as many lanes per SIMD register.

Per-step allocations fall from $13$ to $8.6$~KiB and the step time edges down to
$5.9$~ms (Table~I). Wall-clock training time is essentially unchanged, since the
small network is dominated by the scalar backward convolution rather than by BLAS
or memory bandwidth. Single precision is sufficient here: over four seeds the test
accuracy is $86.4 plus.minus 0.46%$, statistically indistinguishable from the
double-precision result, with every seed above the $85%$ target.

== He initialization

Finally, we replace the constant weight scaling with He initialization,
$sqrt(2 / "fan-in")$, as recommended for ReLU networks @he2015delving. We apply
it uniformly to all trainable weights: the convolution filters and the dense
layers. The original $cal(N)(0,1) dot 0.01$ scaling was too small for the dense
path, which under-trained within the three-epoch budget. This is the only change
that targets accuracy rather than speed: it raises the test accuracy from
approximately $83%$ to approximately $86%$, lowers the seed-to-seed variance, and
reaches the target threshold one epoch earlier.

= Methodology

All timings are wall-clock measurements collected on the same 6-core CPU machine.
Each benchmark was run after a warm-up pass to avoid including Julia compilation
time, and all compared implementations were executed one at a time to avoid CPU
contention. Library micro-benchmarks use
`BenchmarkTools.@btime` on a fixed random batch, isolating the library from data
loading. Convolution kernels are benchmarked separately and checked for numerical
equality, with relative tolerance $10^(-3)$, against the retained naive kernels.
This provides a direct correctness check for both the forward output and the
gradients. End-to-end accuracy is measured by training on Fashion-MNIST for the
fixed 3 epochs. To avoid over-reading a single stochastic run, we report the mean
and standard deviation over four random seeds. Because a cold run compiles kernels
mid-loop and perturbs the result, every measured program performs a warm-up pass
before timing. We compare against Flux and PyTorch implementations with identical architecture
and hyper-parameters, both on CPU and run one at a time to avoid contention.
Flux runs in the same Julia environment, so it allows us to compare time,
accuracy, and allocations per step. PyTorch uses a different runtime, so we do
not report Julia allocation counts for it.

= Results

#figure(
  caption: [Per training step (batch 20, one CPU thread) across optimization stages. Time is largely unchanged until the convolution is vectorized; allocations collapse with in-place operators.],
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    inset: (x: 6pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },
    table.header[Stage][fwd\ (ms)][bwd\ (ms)][step\ (ms)][alloc\ /step],
    [baseline prototype], [10.33], [7.17], [17.29], [1.84 MiB],
    [\+ type stability], [10.33], [7.23], [17.28], [1.84 MiB],
    [\+ in-place ops], [10.17], [7.21], [17.11], [12.95 KiB],
    [\+ `@turbo` conv], [0.60], [5.56], [6.13], [12.97 KiB],
    [\+ `Float32`], [*0.58*], [*5.46*], [*5.91*], [*8.58 KiB*],
  )
) <tab:stages>

#figure(
  caption: [Isolated convolution kernels (ms, dense-gradient worst case). `@turbo` vectorizes the forward pass; the backward pass is a scatter and is left as a scalar loop.],
  table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    inset: (x: 6pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },
    table.header[Kernel][naive][`@turbo`],
    [conv1 forward], [1.42], [*0.076*],
    [conv1 backward], [2.85], [2.64],
    [conv2 forward], [5.73], [*0.191*],
    [conv2 backward], [11.39], [10.60],
  )
) <tab:conv>

#figure(
  caption: [Final library compared with same-architecture Flux and PyTorch references (CPU, 4 seeds, 3 epochs). Each library uses its default BLAS thread count.],
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    inset: (x: 6pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },
    table.header[Metric][CustomAwid][Flux][PyTorch],
    [Test accuracy (%)], [*86.4 ± 0.46*], [85.2 ± 0.48], [84.0 ± 0.13],
    [Time / 3 epochs (s)], [≈ 88], [≈ 63], [*≈ 22.9*],
    [Alloc. / step], [*8.6 KiB*], [2145 KiB], [---],
    [BLAS threads], [6], [12], [6],
  )
) <tab:refs>

Table~I summarizes the progression. Type stability and in-place operators leave
the step time essentially unchanged, but the latter removes $99%$ of allocations.
The vectorized convolution then cuts the step time by $2.8 times$, and single
precision trims it further to $5.9$~ms ($2.9 times$ over baseline) while halving
every array. The final configuration satisfies the required accuracy, and the
performance optimizations did not noticeably hurt accuracy in our runs.

Table~III compares the final library with Flux and PyTorch. All three reach a
comparable test accuracy; the small differences ($lt.eq 2$ percentage points) do
not mean that one implementation is mathematically better. The architecture,
optimizer, dataset, and training budget are the same, but initialization and
backend details differ. Over only three epochs, these differences are visible. Our
library attains the highest mean accuracy in this experiment, which is mainly a
sanity check that it learns correctly.

The comparison with Flux is the more controlled framework comparison, because
Flux also runs in Julia and therefore removes some of the runtime differences
that affect the PyTorch comparison. Two observations follow. First, on memory our
in-place graph allocates $8.6$~KiB per step against Flux's $2145$~KiB, a
$approx 250 times$ reduction. Both implementations train in single precision, so
this is a like-for-like comparison. The difference comes from our explicit
preallocated graph design, where the main intermediate buffers are reused rather
than repeatedly materialized.

Second, Flux is faster in wall-clock time ($approx 63$ vs. $88$~s), because it
computes convolution through optimized matrix-multiplication kernels, whereas our
convolution is the single-threaded `@turbo` loop. Giving our
library more BLAS threads makes it slower, approximately $115$~s at twelve
threads, since its only BLAS calls are the small dense matrix products, for which
extra threads add overhead. We therefore report it at its best default. PyTorch, on a different runtime, is faster still ($approx 3.4 times$) through
optimized convolution kernels and threading. Across both references, the
optimized library stays within the same order of magnitude on time while using
much less memory than Flux, which is a reasonable result for a compact
from-scratch implementation.

= Discussion

Two results are notable. First, the type-stability optimization measured no
speed-up. This negative result is still useful because it explains where the
actual bottleneck was. Second, `@turbo` helped the forward convolution much more
than the backward convolution. The forward pass writes each output cell once,
while the backward pass accumulates into shared gradient cells. Because of these
overlapping writes, the backward loop is harder to vectorize safely.

= Conclusion

By applying a disciplined, measurement-driven sequence of optimizations --- type
stability, in-place BLAS-backed operators, SIMD-vectorized convolution, and single
precision --- we reduced per-step training time by $2.9 times$ and per-step
allocations by $99%$ in a from-scratch Julia AD library, while preserving the
required test accuracy ($86.4 plus.minus 0.46%$) and verifying every change
against a numerical oracle. Benchmarked against Flux and PyTorch references with the same architecture and
training setup, the optimized library reaches comparable accuracy and stays
within the same order of magnitude on time, while allocating $approx 250 times$
less memory per step than Flux.

#bibliography("refs.bib")

