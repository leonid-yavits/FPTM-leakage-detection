# Train FPTM on a localization split (epoch selected on validation, as in fptm_detect.jl) and dump every clause:
# polarity, mean vote on test traces, and its included literals as (sample, threshold k, negated).
# Usage: julia -t 2 dump_clauses.jl WORKDIR SEED EPOCHS OUTFILE
const ROOT = get(ENV, "FPTM_ROOT", dirname(@__DIR__))
include(joinpath(get(ENV, "TSETLIN_JL", joinpath(ROOT, "src", "Tsetlin.jl")), "src", "Tsetlin.jl"))
using .Tsetlin: TMClassifier, train!, predict, accuracy, compile
using Random, Printf
W = ARGS[1]; SD, EP = parse(Int, ARGS[2]), parse(Int, ARGS[3]); OUT = ARGS[4]
m = read(joinpath(W, "meta.json"), String); gi(k) = parse(Int, match(Regex("\"$k\": (\\d+)"), m)[1])
D, K, nbits, nch = gi("D"), gi("K"), gi("nbits"), gi("nchunks")
ns(k) = parse(Int, match(Regex("\"$k\": (\\d+)"), split(m, "\"n\"")[2])[1])
function ld(name); n = ns(name); X = Matrix{UInt64}(undef, nch, n); read!(joinpath(W, "$(name)_bits.bin"), X); y = Vector{UInt8}(undef, n); read!(joinpath(W, "$(name)_y.bin"), y); X, Bool.(y); end
Xtr, ytr = ld("train"); Xva, yva = ld("val"); Xte, yte = ld("test")
Random.seed!(SD); tm = TMClassifier(nbits, ytr, 200, 100, 100, 16, 8, states_num=256, include_limit=128)
best = (0.0, compile(tm))
for e in 1:EP; train!(tm, Xtr, ytr, shuffle=true); a = accuracy(predict(tm, Xva), yva); a > best[1] && (global best = (a, compile(tm))); end
tmb = best[2]; c = tmb.clauses
@printf("test acc %.4f\n", accuracy(predict(tmb, Xte), yte))
function vote(x, l, li); f = 0; @inbounds for i in eachindex(l); f += count_ones((((l[i] ⊻ li[i]) & x[i]) ⊻ l[i])); end; max(0, 8 - f); end
open(OUT, "w") do io
    for (pol, P, Pi) in (("pos", c.positive_included_literals, c.positive_included_literals_inverted), ("neg", c.negative_included_literals, c.negative_included_literals_inverted))
        for j in axes(P, 2)
            l = view(P, :, j); li = view(Pi, :, j)
            v = sum(vote(view(Xte, :, i), l, li) for i in axes(Xte, 2)) / size(Xte, 2)
            b1 = BitVector(undef, nbits); copyto!(b1.chunks, l); b2 = BitVector(undef, nbits); copyto!(b2.chunks, li)
            lits = String[]
            for b in findall(b1); push!(lits, "$(mod1(b, D)-1):$(div(b - 1, D) + 1):0"); end   # sample (0-based):k:negated
            for b in findall(b2); push!(lits, "$(mod1(b, D)-1):$(div(b - 1, D) + 1):1"); end
            println(io, pol, "\t", j, "\t", @sprintf("%.4f", v), "\t", join(lits, ","))
        end
    end
end
