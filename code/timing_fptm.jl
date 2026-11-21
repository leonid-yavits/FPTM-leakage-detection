const ROOT = get(ENV, "FPTM_ROOT", dirname(@__DIR__))
include(joinpath(get(ENV, "TSETLIN_JL", joinpath(ROOT, "src", "Tsetlin.jl")), "src", "Tsetlin.jl"))
using .Tsetlin: TMClassifier, train!, predict
using Random
W = ARGS[1]; m = read(joinpath(W, "meta.json"), String)
gi(k) = parse(Int, match(Regex("\"$k\": (\\d+)"), m)[1]); nbits, nch = gi("nbits"), gi("nchunks")
ns(k) = parse(Int, match(Regex("\"$k\": (\\d+)"), split(m, "\"n\"")[2])[1])
function ld(name); n = ns(name); X = Matrix{UInt64}(undef, nch, n); read!(joinpath(W, "$(name)_bits.bin"), X); y = Vector{UInt8}(undef, n); read!(joinpath(W, "$(name)_y.bin"), y); X, Bool.(y); end
Xtr, ytr = ld("train"); Xte, yte = ld("test")
for rep in 1:2  # first repetition includes compilation; report the second
    Random.seed!(1); t0 = time()
    tm = TMClassifier(nbits, ytr, 200, 100, 100, 16, 8, states_num=256, include_limit=128)
    for e in 1:10; train!(tm, Xtr, ytr, shuffle=true); end
    t1 = time(); p = predict(tm, Xte); t2 = time()
    rep == 2 && println("{\"train_10ep_s\": $(round(t1-t0, digits=1)), \"predict_s\": $(round(t2-t1, digits=2)), \"n_test\": $(length(yte)), \"per_trace_us\": $(round((t2-t1)/length(yte)*1e6, digits=1))}")
end
