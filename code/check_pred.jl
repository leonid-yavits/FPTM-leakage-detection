const ROOT = get(ENV, "FPTM_ROOT", dirname(@__DIR__))
include(joinpath(get(ENV, "TSETLIN_JL", joinpath(ROOT, "src", "Tsetlin.jl")), "src", "Tsetlin.jl"))
using .Tsetlin: TMClassifier, train!, predict
using Random
W = "$ROOT/work/ref_K3"; nbits, nch = 38880, 608
function ld(name, n); X = Matrix{UInt64}(undef, nch, n); read!("$W/$(name)_bits.bin", X); y = Vector{UInt8}(undef, n); read!("$W/$(name)_y.bin", y); X, Bool.(y); end
Xtr, ytr = ld("train", 6256); Xte, yte = ld("test", 3130)
P = Vector{Vector{Bool}}()
for sd in (4, 5)
    Random.seed!(sd); tm = TMClassifier(nbits, ytr, 200, 100, 100, 16, 8, states_num=256, include_limit=128)
    for e in 1:9; train!(tm, Xtr, ytr, shuffle=true); end
    p = predict(tm, Xte); push!(P, p); println("seed $sd correct = ", sum(p .== yte))
end
println("prediction disagreements between seeds: ", sum(P[1] .!= P[2]))
