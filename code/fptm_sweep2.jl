# FPTM side of the traces-to-detection sweep. Fixed epochs, accuracy of the final epoch on the test half.
# Usage: julia -t 2 fptm_sweep2.jl SRC SWEEPDIR CLAUSES T S L LF EPOCHS OUTFILE
include(joinpath(ARGS[1], "src/Tsetlin.jl"))
using .Tsetlin: TMClassifier, train!, predict
using Random, Printf
FW = ARGS[2]; CL, T, S, L, LF, EP = parse.(Int, ARGS[3:8]); OUT = ARGS[9]
meta = read(joinpath(FW, "meta.json"), String)
geti(k) = parse(Int, match(Regex("\"$k\": (\\d+)"), meta)[1]); nbits, nch = geti("nbits"), geti("nchunks")
function log10sf(k, n)
    lg = cumsum([0.0; log.(1:n)]); lp = [-n * log(2.0) + lg[n+1] - lg[i+1] - lg[n-i+1] for i in k:n]
    m = maximum(lp); (m + log(sum(exp.(lp .- m)))) / log(10)
end
function ld(N, r, part, n)
    X = Matrix{UInt64}(undef, nch, n); read!(joinpath(FW, "$(N)_$(r)_$(part)_bits.bin"), X)
    y = Vector{UInt8}(undef, n); read!(joinpath(FW, "$(N)_$(r)_$(part)_y.bin"), y); X, Bool.(y)
end
done = Set{String}()
isfile(OUT) && for l in eachline(OUT); m = match(r"\"N\": (\d+), \"r\": (\d+)", l); m !== nothing && push!(done, "$(m[1])_$(m[2])"); end
open(OUT, "a") do io
    for line in eachline(joinpath(FW, "list.txt"))
        N, r, ntr, nte = parse.(Int, split(line))
        "$(N)_$(r)" in done && continue
        Xtr, ytr = ld(N, r, "train", ntr); Xte, yte = ld(N, r, "test", nte)
        Random.seed!(1000 + r); t0 = time()
        tm = TMClassifier(nbits, ytr, CL, T, S, L, LF, states_num=256, include_limit=128)
        for e in 1:EP; train!(tm, Xtr, ytr, shuffle=true); end
        k = sum(predict(tm, Xte) .== yte); lp = log10sf(k, nte)
        s = @sprintf("{\"N\": %d, \"r\": %d, \"fptm\": %.3f, \"fptm_acc\": %.4f, \"sec\": %.1f}", N, r, lp, k / nte, time() - t0)
        println(io, s); flush(io); println(s); flush(stdout)
    end
end
