using Distributions
using Convex
using SCS
using Wavelets
using Base.Test
using JuMP
using Clp

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

function lik(H, r, r_prev)
    # make m via VARMA
    m = fill(0, length(r))
    # m = r_prev
    -1/2 * (r - m)' * H^-1 * (r - m) - log(sqrt(2*pi*det(H)))
    # logpdf(MvNormal(m, H), r)
end

@testset "likelihood" begin
    srand(123)
    @test lik(cov(randn(10,3)), randn(3), [0,0,0]) == -2.885949899751278
end

function formula(c, d, A, G, r, H_prev) 
    C = make_upper_triangle(c, d)
    print(C)
    C'*C + A'*r*r'*A + G'*H_prev*G # K = 1 for now!!
end

# upper(n) = reduce(vcat, [[fill(0, i)' Variable(n - i)'] for i in 0:(n-1)])

# do something with upper triangular nonsense
# bring formula into optimizer, not as udf

function optimizer(r, H_prev)
    n = length(r)
    m = Model(solver = ClpSolver())
    @variable(m, d[1:n] >= 0) # diagonal of C is positive
    @variable(m, c[1:corner_length(n)]) # upper corner of C
    @variable(m, A[1:n, 1:n])
    @variable(m, G[1:n, 1:n])
    # add positive contraints on 1,1 of A and G? 
    @objective(m, Max, formula(c, d, A, G, r, H_prev))
    solve(m)
end
