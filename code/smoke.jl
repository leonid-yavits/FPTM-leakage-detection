include(joinpath(ARGS[1], "src/Tsetlin.jl"))
using .Tsetlin: TMClassifier, InputBatch, train!, predict, accuracy, booleanize
using Random
Random.seed!(1)
D=200; n=4000
X=[randn(Float32,D) for _ in 1:n]; y=rand(Bool,n)
for i in 1:n; y[i] && (X[i][77]+=0.8f0); end
xb=InputBatch([booleanize(x, -0.5, 0.0, 0.5) for x in X])
tm=TMClassifier(3D, y, 20, 20, 50, 30, 10, states_num=256, include_limit=128)
train!(tm, xb[:,1:3000], y[1:3000], xb[:,3001:end], y[3001:end], 10, verbose=0)
println("acc ", accuracy(predict(tm, xb[:,3001:end]), y[3001:end]))
