# Train FPTM on one sweep subset (train half), test on the test half, and write the localization score S(t).
# Usage: julia -t 2 fptm_loc_sweepfile.jl SWEEPDIR N R EPOCHS SEED OUTFILE
const ROOT = get(ENV, "FPTM_ROOT", dirname(@__DIR__))
include(joinpath(get(ENV, "TSETLIN_JL", joinpath(ROOT, "src", "Tsetlin.jl")), "src", "Tsetlin.jl"))
using .Tsetlin: TMClassifier, train!, predict, compile
using Random, Printf
FW = ARGS[1]; N, R, EP, SD = parse.(Int, ARGS[2:5]); OUT = ARGS[6]
m = read(joinpath(FW, "meta.json"), String); gi(k) = parse(Int, match(Regex("\"$k\": (\\d+)"), m)[1])
D, K, nbits, nch = gi("D"), gi("K"), gi("nbits"), gi("nchunks")
ntr, nte = 0, 0
for l in eachline(joinpath(FW, "list.txt")); a = parse.(Int, split(l)); (a[1] == N && a[2] == R) && (global ntr = a[3]; global nte = a[4]); end
function ld(part, n); X = Matrix{UInt64}(undef, nch, n); read!(joinpath(FW, "$(N)_$(R)_$(part)_bits.bin"), X); y = Vector{UInt8}(undef, n); read!(joinpath(FW, "$(N)_$(R)_$(part)_y.bin"), y); X, Bool.(y); end
Xtr, ytr = ld("train", ntr); Xte, yte = ld("test", nte)
Random.seed!(SD); tm = TMClassifier(nbits, ytr, 200, 100, 100, 16, 8, states_num=256, include_limit=128)
for e in 1:EP; train!(tm, Xtr, ytr, shuffle=true); end
k = sum(predict(tm, Xte) .== yte)
lg = cumsum([0.0; log.(1:nte)]); lp = [-nte * log(2.0) + lg[nte+1] - lg[i+1] - lg[nte-i+1] for i in k:nte]; mx = maximum(lp)
@printf("LOC N=%d r=%d seed=%d acc=%.4f log10p=%.2f\n", N, R, SD, k / nte, (mx + log(sum(exp.(lp .- mx)))) / log(10))
c = tm.clauses; cnt = zeros(Int, nbits)
for M in (c.positive_included_literals, c.positive_included_literals_inverted, c.negative_included_literals, c.negative_included_literals_inverted), j in axes(M, 2)
    bv = BitVector(undef, nbits); copyto!(bv.chunks, view(M, :, j)); cnt .+= bv
end
write(OUT, Float64.(vec(sum(reshape(cnt, D, K), dims=2))))
