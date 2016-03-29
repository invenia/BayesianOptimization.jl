"""
Provides a really simple implementation of Bayesian Optimization
for black box function. The current implementation is loosely based
off of [this](https://github.com/fmfn/BayesianOptimization) python library.

TODO: This could probably be setup to implement the
[MathProgBase](https://github.com/JuliaOpt/MathProgBase.jl)
SolverInterface, but for now I'm going to leave this as is.
"""
module BayesianOptimization

using Logging
using Distributions
using GaussianProcesses
using NLopt

export BayesOpt, optimize!, optimize

# Logging.configure(output=STDOUT, level=DEBUG)

"""
A type for storing our search history, bounds and model.
"""
type BayesOpt
    """
    A function which takes the parameters
    as input and returns a value which is to
    be maximized."""
    f::Function

    """
    The history of parameters tried as an N x M matrix
    where N is the number of parameters and M is the number
    observations.
    """
    X::Array

    """ The history of returned values to function `f` """
    y::Array

    """ The best set of parameters tried """
    xmax::Array

    """ The best result found """
    ymax::Float64

    """
    A collection of bounds for each parameter to restrict the search.
    Each bound can be an abstract array, but is typically a Range{T},
    """
    bounds::Array{AbstractArray}

    """
    The internal model of the dataset which can predict the next value.

    Currently, only a [GaussianProcess](https://github.com/STOR-i/GaussianProcesses.jl)
    is used.
    """
    model::Any
end

"""
The default BayesOpt constructor

Args:
- `f`: the function to optimize
- `bounds`: a collection of bounds (1 per parameter) to direct the search.

Returns:
- `BayesOpt`: a new `BayesOpt` instance.
"""
function BayesOpt{T<:AbstractArray}(f::Function, bounds::T...; noise=-1e8)
    # Initialize our parameters matrix
    X = Array{Float64}(length(bounds), 1)

    # Initialize our results array
    y = Array{Float64}(1)

    # Set our initial observations
    # requiring use to run the function once
    xmax = [rand(b) for b in bounds]
    ymax = f(xmax...)

    X[:,1] = xmax
    y[1] = ymax

    # return our new BayesOpt instance with these reasonable defaults
    # NOTE: see https://github.com/STOR-i/GaussianProcesses.jl for our
    # Gaussian Process arguments.
    BayesOpt(
        f, X, y, xmax, ymax, collect(bounds),
        GP(X[:,1:1], y[1:1], MeanZero(), SE(0.0, 0.0), noise)
    )
end

"""
Performs the optimization loop on a `BayesOpt` instance for the
specified number of iterations with the provided number of restarts.

Args:
- `opt`: the `BayesOpt` type to optimize
- `iter`: the number of iterations to run the optimization loop
- `restarts`: the number of restarts to allow in the acquisition function.
- `noise` : log of the observation noise, default no observation noise

Returns:
- `Array{Real}(nparams, 1)`: the best params found for `f`
- `Real`: the best result found.
"""
function optimize!(opt::BayesOpt, iter=100, restarts=10; noise=-1e8)
    # The last index we looked at is equal to the current number of
    # observation looked at.
    last_index = length(opt.y) + 1

    # Allocate `iter` number of observation to both `X` and `y`
    opt.X = hcat(opt.X, Array{Float64}(length(opt.bounds), iter))
    append!(opt.y, Array{Float64}(iter))

    # Begin our optimization loop
    for i in last_index:length(opt.y)
        Logging.debug("iteration=$i, x=$(opt.xmax), y=$(opt.ymax)")

        # Select the next params
        new_x = acquire_max(opt.model, opt.ymax, restarts, opt.bounds)

        # Run the next params
        new_y = opt.f(new_x...)

        Logging.debug("new x=$new_x, y=$new_y")

        # update X and y with next params and run
        opt.X[:,i] = new_x
        opt.y[i] = new_y

        # update the observations in the internal model (ie: the GP).
        # NOTE: This update process is specific to Gaussian Processes,
        # but could be refactored out.
        # opt.model.x = opt.X[:,1:i+1]
        # opt.model.y = opt.y[1:i+1]
        # GaussianProcesses.update_mll!(opt.model)

        # Just rebuild the model cause apparently the above complains if we update with more data.
        opt.model = GP(opt.X[:,1:i], opt.y[1:i], MeanZero(), SE(0.0, 0.0), noise)

        # Update the max x and y if our new
        # result is better.
        if new_y > opt.ymax
            opt.ymax = new_y
            opt.xmax = new_x
        end
    end

    Logging.info("Optimization completed: x=$(opt.xmax), y=$(opt.ymax)")

    # return our best parameters and result respectively.
    return opt.xmax, opt.ymax
end


"""
A helper function which handles creating the `BayesOpt` instance and
running `optimize!`

Args:
- `f`: the function to optimize
- `bounds`: a collection of bounds (1 per parameter) to direct the search.
- `iter`: the number of iterations to run the optimization loop
- `restarts`: the number of restarts to allow in the acquisition function.
- `noise` : log of the observation noise, default no observation noise

Returns:
- `Array{Real}(nparams, 1)`: the best params found for `f`
- `Real`: the best result found.
- `BayesOpt`: a new `BayesOpt` instance.
"""
function optimize{T<:AbstractArray}(f::Function, bounds::T...; iter=100, restarts=10, noise=-1e8)
    # Create out BayesOpt instance
    opt = BayesOpt(f, bounds...; noise=noise)

    # call `optimize!` for it.
    optimize!(opt, iter, restarts; noise=noise)

    # return the x and y maximums like `optimize!` along with the
    # BayesOpt instance.
    return opt.xmax, opt.ymax, opt
end


"""
Acquire function wraps a tradition [acquisition function](http://arxiv.org/pdf/1012.2599.pdf)
which selects the next x value(s) that will produce the minimum
value of y balancing exploration vs exploitation.

NOTE: While currently `expected_improvement` is the only option, future versions will support
different acquisition function described in the linked paper.

Args:
- `model`: the model which the acquisition function (in this case EI) needs to use.
- `

Returns:

"""
function acquire_max(model, ymax, restarts, bounds)
    # Selective some random parameters to check
    x_max = [rand(b) for b in bounds]

    # Create an NLopt.Opt instance using LBFGS
    # See https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
    # for details.
    opt = Opt(:LD_LBFGS, length(bounds))

    # Set the upper and lower bounds
    lower_bounds!(opt, Float64[minimum(b) for b in bounds])
    upper_bounds!(opt, Float64[maximum(b) for b in bounds])

    # TODO: Set the step size if a bound is a Range{T} with a step

    # set the objective to minimize the acquisition function.
    min_objective!(opt, expected_improvement(model, ymax))

    # initial max found is 0
    ei_max = 0

    # Optimize the acquisition with different
    # random start parameters for some number of restarts
    # to avoid being stuck in a local minima.
    for i in 1:restarts
        # Sample some points at random.
        x_try = Float64[rand(b) for b in bounds]

        #Find the minimum of minus the acquisition function
        (minf,minx,ret) = NLopt.optimize(opt, x_try)

        #Store it if better than previous minimum(maximum).
        if -minf >= ei_max
            x_max = minx
            ei_max = -minf
        end
    end

    # Return the next best expected set of parameters.
    return x_max
end


"""
The expected improvement acquisition function which balances
exporation vs exploitation by selecting the next x that is not only
the most likely to be improve the current best result, but also
to account for the expected magnitude.

NOTE: Since, all function used with NLopt only take some
data and a gradient (which in this case will always be empty
cause we are using a derivative free optimize method) we use a closure
to generate this function with the model and max y value.

Args:
- `model`: any model that returns the mean and variance when predict is
        called on some sample data.
- 'ymax': the current maximum value

Returns:
- `Function`: The `ei` function to pass to `NLopt.optimize`.
"""
function expected_improvement(model, ymax)
    """
    Inner closure fuction that performs the EI.

    Args:
    - `x`: the observation to pass to the models predict.
    - `grad`: the gradient, which should be an empty array since we are using a
            derivative free optimization method, but is required by NLopt.
    """
    function ei(x, grad)
        # Get the mean and variance from the existing observations
        # by calling predict on a gaussion process.
        mean, var = predict(model, reshape(x, length(x), 1))

        if var == 0
            # If the variance is 0 just return 0
            return 0
        else
            # otherwise calculate the ei and return that.
            Z = (mean - ymax)/sqrt(var)
            res = -((mean - ymax) * cdf(Normal(), Z) + sqrt(var) * pdf(Normal(), Z))[1]
            return res
        end
    end
    return ei
end

end # module

