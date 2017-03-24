using Distributions
using Convex
using SCS
using Wavelets
using Base.Test

# Use SCS solver
solver = SCSSolver(verbose=0)
set_default_solver(solver);

########################
# upper triangle

n_from_corner_length(v)::Int  = sqrt(2*v + 1/4) + 1/2
corner_length(n)::Int = (n^2 - n)/2

function make_corner(v)
    n = sqrt(2*length(v) + 1/4) + 1/2
    k = 0
    [ i < j ? (k+=1; v[k]) : 0 for i=1:n, j=1:n]
end

function make_upper_triangle(v, d)
    corner = make_corner(v)
    diagonal = diagm(d)
    UpperTriangular(corner + diagonal)
end

@testset "creating upper-triangle matrix" begin
    @test n_from_corner_length(6) == 4
    @test n_from_corner_length(10) == 5
    @test corner_length(5) == 10
    @test n_from_corner_length(corner_length(20)) == 20
    @test size(make_upper_triangle([1,2,3], [5,5,5])) == (3,3)
    @test size(make_upper_triangle([1,2,3,1,2,3,1,2,3,4], [5,5,5,5,5])) == (5,5)
end

########################
# optimize

function lik(H, r)
    # m = fill(0, length(r))
    # logpdf(MvNormal(m, H), r)
    S = (r - mean(r))*(r - mean(r))'
    logdet(H) - trace(S * H)
end

@testset "likelihood" begin
    srand(123)
    @test lik(cov(randn(10,3)), randn(3)) == -2.885949899751278
end

function formula(C, A, G, r, H_prev)
    # C = make_upper_triangle(c, d)
    # print(C)
    C'*C + A'*r*r'*A + G'*H_prev*G
end

upper(n) = reduce(vcat, [[fill(0, i)' Variable(n - i)'] for i in 0:(n-1)])

function optimizer(r, H_prev)
    n = length(r)
    # C = upper(n)
    # c = Variable(corner_length(n))
    # d = Variable(n) # positive with 1,1,1?
    # A = Variable(n, n) # first el positive with 1,1,1?
    # G = Variable(n, n) # first el positive with 1,1,1?
    G = zeros(n^2)

    optimize(f, )

    problem = maximize(lik(formula(C, A, G, r, H_prev), r))
    solve!(problem)
end
