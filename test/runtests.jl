using BayesOpt

function foo(a, b)
    return sin(a) + b
end

xmax, ymax, opt = optimize(foo, 1.0:5.0, -2:4; iter=5, restarts=2)

#plot(gp)
