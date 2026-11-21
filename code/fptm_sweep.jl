include(joinpath(ARGS[1], "src/Tsetlin.jl"))
using .Tsetlin: TMClassifier, train!, predict
using Random, Printf
W = ARGS[2]; CL, T, S, L, LF, EP = parse.(Int, ARGS[3:8])
meta = read(joinpath(W, "meta.json"), String)
geti(k) = parse(Int, match(Regex("\"$k\": (\\d+)"), meta)[1]); nbits, nch = geti("nbits"), geti("nchunks")
nsz(k) = parse(Int, match(Regex("\"$k\": (\\d+)"), split(meta, "\"n\"")[2])[1])
function ld(name); n = nsz(name); X = Matrix{UInt64}(undef, nch, n); read!(joinpath(W, "$(name)_bits.bin"), X)
    y = Vector{UInt8}(undef, n); read!(joinpath(W, "$(name)_y.bin"), y); X, Bool.(y); end
Xtr, ytr = ld("train"); Xte, yte = ld("test")
function sf(k, n); lg = cumsum([0.0; log.(1:n)]); lp = [-n*log(2.0) + lg[n+1] - lg[i+1] - lg[n-i+1] for i in k:n]
    m = maximum(lp); exp(m) * sum(exp.(lp .- m)); end
for line in eachline(joinpath(W, "sweep.txt"))
    N, r, h, m = parse.(Int, split(line))
    idx = Vector{Int32}(undef, h + m); read!(joinpath(W, "sw_$(N)_$(r).idx"), idx); a = idx[1:h] .+ 1; b = idx[h+1:end] .+ 1
    Random.seed!(1000 + r)
    tm = TMClassifier(nbits, ytr[a], CL, T, S, L, LF, states_num=256, include_limit=128)
    for e in 1:EP; train!(tm, Xtr[:, a], ytr[a], shuffle=true); end
    k = sum(predict(tm, Xte[:, b]) .== yte[b]); p = sf(k, m)
    @printf("SWEEP N=%d r=%d acc=%.4f p=%.3e detect=%d\n", N, r, k / m, p, p < 1e-5); flush(stdout)
end
