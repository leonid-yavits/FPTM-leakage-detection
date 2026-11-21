# Train FPTM exactly as fptm_detect.jl (epoch selected on validation) and dump the model and its test predictions,
# for the GainP inference model (gainp_fptm.py).
# Usage: julia -t 2 dump_model.jl SRC WORKDIR EPOCHS SEED OUTPREFIX
# Writes OUTPREFIX_clauses.tsv (polarity, clause, comma-separated literals bit:negated, 0-based bit index)
#        OUTPREFIX_pred.tsv    (test trace, positive vote sum, negative vote sum, prediction, label)
include(joinpath(ARGS[1], "src/Tsetlin.jl"))
using .Tsetlin: TMClassifier, train!, predict, accuracy, compile
using Random, Printf
W = ARGS[2]; EP, SEED = parse(Int, ARGS[3]), parse(Int, ARGS[4]); OUT = ARGS[5]
Random.seed!(SEED)
meta = read(joinpath(W, "meta.json"), String)
geti(k) = parse(Int, match(Regex("\"$k\": (\\d+)"), meta)[1])
D, K, nbits, nch = geti("D"), geti("K"), geti("nbits"), geti("nchunks")
nsz(k) = parse(Int, match(Regex("\"$k\": (\\d+)"), split(meta, "\"n\"")[2])[1])
function loadsplit(name)
    n = nsz(name); X = Matrix{UInt64}(undef, nch, n); read!(joinpath(W, "$(name)_bits.bin"), X)
    y = Vector{UInt8}(undef, n); read!(joinpath(W, "$(name)_y.bin"), y); return X, Bool.(y)
end
Xtr, ytr = loadsplit("train"); Xva, yva = loadsplit("val"); Xte, yte = loadsplit("test")
tm = TMClassifier(nbits, ytr, 200, 100, 100, 16, 8, states_num=256, include_limit=128)
best = (0.0, compile(tm))
for e in 1:EP
    train!(tm, Xtr, ytr, shuffle=true); a = accuracy(predict(tm, Xva), yva)
    a > best[1] && (global best = (a, compile(tm)))
end
tmb = best[2]; c = tmb.clauses
pred = predict(tmb, Xte)
@printf("test acc %.4f\n", accuracy(pred, yte))
function vote(x, l, li); f = 0; @inbounds for i in eachindex(l); f += count_ones((((l[i] ⊻ li[i]) & x[i]) ⊻ l[i])); end; max(0, 8 - f); end
sets = (("pos", c.positive_included_literals, c.positive_included_literals_inverted), ("neg", c.negative_included_literals, c.negative_included_literals_inverted))
open(OUT * "_clauses.tsv", "w") do io
    for (pol, P, Pi) in sets, j in axes(P, 2)
        b1 = BitVector(undef, nbits); copyto!(b1.chunks, view(P, :, j)); b2 = BitVector(undef, nbits); copyto!(b2.chunks, view(Pi, :, j))
        lits = vcat(["$(b - 1):0" for b in findall(b1)], ["$(b - 1):1" for b in findall(b2)])
        println(io, pol, "\t", j, "\t", join(lits, ","))
    end
end
open(OUT * "_pred.tsv", "w") do io
    for i in axes(Xte, 2)
        x = view(Xte, :, i)
        vp = sum(vote(x, view(sets[1][2], :, j), view(sets[1][3], :, j)) for j in axes(sets[1][2], 2))
        vn = sum(vote(x, view(sets[2][2], :, j), view(sets[2][3], :, j)) for j in axes(sets[2][2], 2))
        println(io, i - 1, "\t", vp, "\t", vn, "\t", Int(pred[i]), "\t", Int(yte[i]))
    end
end
