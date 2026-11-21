# FPTM leakage detection on thermometer-coded traces.
# Usage: julia -t 2 fptm_detect.jl SRC WORKDIR LABELS(y|ynull) CLAUSES T S L LF EPOCHS SEED TAG
include(joinpath(ARGS[1], "src/Tsetlin.jl"))
using .Tsetlin: TMClassifier, InputBatch, train!, predict, accuracy, compile
using Random, Printf

W = ARGS[2]; lab = ARGS[3]
CL, T, S, L, LF, EP, SEED = parse.(Int, ARGS[4:10]); TAG = ARGS[11]
Random.seed!(SEED)

meta = read(joinpath(W, "meta.json"), String)
geti(k) = parse(Int, match(Regex("\"$k\": (\\d+)"), meta)[1])
D, K, nbits, nch = geti("D"), geti("K"), geti("nbits"), geti("nchunks")
nsz(k) = parse(Int, match(Regex("\"$k\": (\\d+)"), split(meta, "\"n\"")[2])[1])

function loadsplit(name)
    n = nsz(name)
    X = Matrix{UInt64}(undef, nch, n); read!(joinpath(W, "$(name)_bits.bin"), X)
    y = Vector{UInt8}(undef, n); read!(joinpath(W, "$(name)_$(lab).bin"), y)
    return X, Bool.(y)
end
Xtr, ytr = loadsplit("train"); Xva, yva = loadsplit("val"); Xte, yte = loadsplit("test")

tm = TMClassifier(nbits, ytr, CL, T, S, L, LF, states_num=256, include_limit=128)
best = (0.0, 0, compile(tm))
t0 = time()
for e in 1:EP
    train!(tm, Xtr, ytr, shuffle=true)
    acc = accuracy(predict(tm, Xva), yva)
    if acc > best[1]; global best = (acc, e, compile(tm)); end
    @printf("epoch %d val %.4f (%.0fs)\n", e, acc, time() - t0); flush(stdout)
end
acc_va, ep_best, tmb = best

# One-sided exact binomial test on the untouched test split, H0: accuracy = 0.5
pred = predict(tmb, Xte); k = sum(pred .== yte); n = length(yte)
function binom_log10sf(k, n)       # log10 P(X >= k), X ~ Bin(n, 1/2)
    lp = [(-n * log(2.0)) + (lgam(n + 1) - lgam(i + 1) - lgam(n - i + 1)) for i in k:n]
    m = maximum(lp); return (m + log(sum(exp.(lp .- m)))) / log(10)
end
lgamv = cumsum([0.0; log.(1:n)])                 # lgamv[j+1] = log(j!)
lgam(x) = lgamv[x]
lp10 = binom_log10sf(k, n)
@printf("RESULT tag=%s labels=%s best_epoch=%d val=%.4f test=%.4f (k=%d/n=%d) log10p=%.2f\n", TAG, lab, ep_best, acc_va, k / n, k, n, lp10)

# Localization: included literals per input bit, over both clause polarities and both literal kinds
c = tmb.clauses
lits = (c.positive_included_literals, c.positive_included_literals_inverted, c.negative_included_literals, c.negative_included_literals_inverted)
cnt = zeros(Int, nbits)
for M in lits, j in axes(M, 2)
    bv = BitVector(undef, nbits); copyto!(bv.chunks, view(M, :, j)); cnt .+= bv
end
S_inc = vec(sum(reshape(cnt, D, K), dims=2))   # bit b -> sample mod1(b, D)

# Vote-weighted variant: mean vote of each clause on test traces times its literal count at t
function clause_vote(x, l, li)
    f = 0
    @inbounds for i in eachindex(l); f += count_ones((((l[i] ⊻ li[i]) & x[i]) ⊻ l[i])); end
    return max(0, LF - f)
end
S_vote = zeros(Float64, D)
for (P, Pi) in ((c.positive_included_literals, c.positive_included_literals_inverted), (c.negative_included_literals, c.negative_included_literals_inverted))
    for j in axes(P, 2)
        l = view(P, :, j); li = view(Pi, :, j)
        v = sum(clause_vote(view(Xte, :, i), l, li) for i in 1:n) / n
        v == 0 && continue
        b1 = BitVector(undef, nbits); copyto!(b1.chunks, l); b2 = BitVector(undef, nbits); copyto!(b2.chunks, li)
        S_vote .+= v .* vec(sum(reshape(b1 .| b2, D, K), dims=2))
    end
end
open(joinpath(W, "S_$(TAG).bin"), "w") do io; write(io, Float64.(S_inc)); write(io, S_vote); end
@printf("literals total %d, samples touched %d\n", sum(cnt), count(>(0), S_inc))
