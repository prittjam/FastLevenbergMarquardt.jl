using FastLevenbergMarquardt

using Statistics: mean
using Aqua
using Printf
using StaticArrays
using Test


@testset "FastLevenbergMarquardt.jl" begin
    Aqua.test_all(
        FastLevenbergMarquardt,
        deps_compat=(ignore=[:LinearAlgebra, :SparseArrays, :SuiteSparse],)
    )
end


@testset "NIST" begin
    include("nist.jl")

    blre(x, y) = clamp(minimum(-log10.(abs.(x .- y)./abs.(y))), 0, 11)

    @testset "Default Solver" begin
        @printf("\nNIST - Default Solver\n\n")
        @printf("%17s lre      rss    conv  iter  nfev  njev\n", "")
        resv = []

        for P in Problems
            r = zeros(P.m)
            J = zeros(P.m, P.n)

            for (i, b0) in enumerate(P.b0)
                res1 = lmsolve!(P.fun!, P.jac!, copy(b0), P.m, P.data)
                res2 = lmsolve!(P.fun!, P.jac!, copy(b0), r, J, P.data)
                @test res1[1:end-2] == res2[1:end-2]

                b, F, converged, iter, nfev, njev, LM, solver = res1
                res3 = lmsolve!(P.fun!, P.jac!, (LM.x .= b0; LM), P.data, solver=solver)
                @test res3[1:end-2] == res1[1:end-2]

                lre = blre(b, P.b)
                @test F ≈ P.rss rtol = 1e-10 atol = 1e-20
                @test lre > 5
                @test converged > 0
                push!(resv, (P.name, i, lre, F, converged, iter, nfev, njev))
                @printf("%12s %d   %4.1f  %.4e  %2d  %4d  %4d  %4d\n",
                    i == 1 ? P.name * " -" : "", i, lre, F, converged, iter, nfev, njev)
            end
            for (i, b0) in enumerate(P.b0)
                b0 = SVector{P.n}(b0)
                b, F, converged, iter, nfev, njev, = lmsolve!(P.fun!, P.jac!, b0, P.m, P.data)

                lre = blre(b, P.b)
                @test F ≈ P.rss rtol = 1e-10 atol = 1e-20
                @test lre > 5
                @test converged > 0
            end
        end

        @printf("\n")
        meanlrev = mean(r -> r[3], resv)
        minlrev  = minimum(r -> r[3], resv)
        @printf("mean log relative error = %4.1f, min lre = %4.1f\n", meanlrev, minlrev)
        @printf("\n")
    end

    @testset "Dense Cholesky" begin
        @printf("\nNIST - Dense Cholesky\n\n")
        @printf("%17s lre      rss    conv  iter  nfev  njev\n", "")
        resv = []

        for P in Problems
            r = zeros(P.m)
            J = zeros(P.m, P.n)

            for (i, b0) in enumerate(P.b0)
                b, F, converged, iter, nfev, njev, = lmsolve!(P.fun!, P.jac!, copy(b0), r, J, P.data, solver=:cholesky)

                lre = blre(b, P.b)
                @test F ≈ P.rss rtol = 1e-10 atol = 1e-20
                @test lre > 5
                @test converged > 0

                push!(resv, (P.name, i, lre, F, converged, iter, nfev, njev))
                @printf("%12s %d   %4.1f  %.4e  %2d  %4d  %4d  %4d\n",
                    i == 1 ? P.name * " -" : "", i, lre, F, converged, iter, nfev, njev)
            end
        end

        @printf("\n")
        meanlrev = mean(r -> r[3], resv)
        minlrev  = minimum(r -> r[3], resv)
        @printf("mean log relative error = %4.1f, min lre = %4.1f\n", meanlrev, minlrev)
        @printf("\n")
    end

    @testset "Dense QR" begin
        @printf("\nNIST - Dense QR\n\n")
        @printf("%17s lre      rss    conv  iter  nfev  njev\n", "")
        resv = []

        problems = []
        for P in Problems
            r = zeros(P.m)
            J = zeros(P.m, P.n)

            for (i, b0) in enumerate(P.b0)
                b, F, converged, iter, nfev, njev, = lmsolve!(P.fun!, P.jac!, copy(b0), r, J, P.data, solver=:qr)

                lre = blre(b, P.b)
                @test F ≈ P.rss rtol = 1e-10 atol = 1e-20
                @test lre > 5
                @test converged > 0
                push!(resv, (P.name, i, lre, F, converged, iter, nfev, njev))
                @printf("%12s %d   %4.1f  %.4e  %2d  %4d  %4d  %4d\n",
                    i == 1 ? P.name * " -" : "", i, lre, F, converged, iter, nfev, njev)
            end
        end

        @printf("\n")
        meanlrev = mean(r -> r[3], resv)
        minlrev  = minimum(r -> r[3], resv)
        @printf("mean log relative error = %4.1f, min lre = %4.1f\n", meanlrev, minlrev)
        @printf("\n")
    end
end


@testset "MGH" begin
    include("mgh.jl")

    flre(x, y) = maximum(-log10.(abs.(x .- y)./ifelse.(y .== 0, 1, abs.(y))))

    @testset "Default Solver" begin
        @printf("\nMGH - Default Solver\n\n")
        resv = []

        for P in TestProblems
            r = zeros(P.m)
            J = zeros(P.m, P.n)

            for i in (1, 10, 100)
                x0 = i.*P.x0

                res1 = lmsolve!(P.fun!, P.jac!, copy(x0), P.m, P.data)
                res2 = lmsolve!(P.fun!, P.jac!, copy(x0), r, J, P.data)
                @test res1[1:end-2] == res2[1:end-2]

                x, F, converged, iter, nfev, njev, LM, solver = res1
                res3 = lmsolve!(P.fun!, P.jac!, (LM.x .= x0; LM), P.data, solver=solver)
                @test res3[1:end-2] == res1[1:end-2]

                lre = flre(F, P.fx)
                push!(resv, (P.name, i, lre, F, converged, iter, nfev, njev))
            end
        end

        n = length(resv)
        nsv = count(r -> r[3] > 4, resv)
        @test nsv > 130
        @printf("%3d/%3d with ||f(x)||_2 to at least 4 digits\n", nsv, n)
        @printf("\n")

        # Test Float32 and BigFloat
        P = RosenbrockN(12)

        for T in (Float32, Float64, BigFloat)
            x0 = Array{T}(P.x0)
            x, F, converged, iter, nfev, njev, = lmsolve!(P.fun!, P.jac!, x0)
            @test F ≈ 0 atol=1e-30
            @test x ≈ ones(T, length(x0)) rtol=1e-16
            @test F isa T
            @test x isa typeof(x0)

            x0 = SVector{12, T}(P.x0)
            fun = (x, data) -> SVector{12}(P.fun!(zeros(T, 12), x, data))
            jac = (x, data) -> SMatrix{12, 12}(P.jac!(zeros(T, 12, 12), x, data))

            x, F, converged, iter, nfev, njev = lmsolve(fun, jac, x0)
            @test F ≈ 0 atol=1e-30
            @test x ≈ ones(T, length(x0)) rtol=1e-16
            @test F isa T
            @test x isa typeof(x0)
        end
    end

    @testset "Dense Cholesky" begin
        @printf("\nMGH - Dense Cholesky\n\n")
        resv = []

        for P in TestProblems
            r = zeros(P.m)
            J = zeros(P.m, P.n)

            for i in (1, 10, 100)
                res = lmsolve!(P.fun!, P.jac!, i.*P.x0, r, J, P.data, solver=:cholesky)
                x, F, converged, iter, nfev, njev, LM, solver = res
                lre = flre(F, P.fx)
                push!(resv, (P.name, i, lre, F, converged, iter, nfev, njev))
            end
        end

        n = length(resv)
        nsv = count(r -> r[3] > 4, resv)
        @test nsv > 130
        @printf("%3d/%3d with ||f(x)||_2 to at least 4 digits\n", nsv, n)
        @printf("\n")

        # Test Float32 and BigFloat
        P = RosenbrockN(12)

        for T in (Float32, BigFloat)
            x0 = Array{T}(P.x0)
            res = lmsolve!(P.fun!, P.jac!, x0, solver=:cholesky)
            x, F, converged, iter, nfev, njev, LM, solver = res
            @test F ≈ 0 atol=1e-30
            @test x ≈ ones(T, length(x0)) rtol=1e-16
            @test F isa T
            @test x isa typeof(x0)
        end
    end

    @testset "Dense QR" begin
        @printf("\nMGH - Dense QR\n\n")
        resv = []

        for P in TestProblems
            r = zeros(P.m)
            J = zeros(P.m, P.n)

            for i in (1, 10, 100)
                res = lmsolve!(P.fun!, P.jac!, i.*P.x0, r, J, P.data, solver=:qr)
                x, F, converged, iter, nfev, njev, LM, solver = res
                lre = flre(F, P.fx)
                push!(resv, (P.name, i, lre, F, converged, iter, nfev, njev))
            end
        end

        n = length(resv)
        nsv = count(r -> r[3] > 4, resv)
        @test nsv > 130
        @printf("%3d/%3d with ||f(x)||_2 to at least 4 digits\n", nsv, n)
        @printf("\n")

        # Test Float32 and BigFloat
        P = RosenbrockN(12)

        for T in (Float32, BigFloat)
            x0 = Array{T}(P.x0)
            if T === BigFloat
                @test_throws ArgumentError lmsolve!(P.fun!, P.jac!, x0, solver=:qr)
            else
                res = lmsolve!(P.fun!, P.jac!, x0, solver=:qr)
                x, F, converged, iter, nfev, njev, LM, solver = res
                @test F ≈ 0 atol=1e-30
                @test x ≈ ones(T, length(x0)) rtol=1e-16
                @test F isa T
                @test x isa typeof(x0)
            end
        end
    end
end


# A negligible trial step that did NOT reduce the objective is not convergence in
# `x`: it reports `4` (no further reduction possible), not `2`.
@testset "rejected negligible step" begin
    # f(x) = x - 3 with a wrong-sign Jacobian: no step along the LM direction
    # descends, so every trial is rejected and the damping grows until the step
    # is below xtol, at the start point.
    f(x, _) = SA[x[1] - 3.0]
    j(x, _) = SA[-1.0;;]
    x, F, converged, = lmsolve(f, j, SA[0.0])
    @test converged == 4
    @test x == SA[0.0] && F == 9.0

    f!(r, x, _) = (r[1] = x[1] - 3.0; r)
    j!(J, x, _) = (J[1, 1] = -1.0; J)
    x, F, converged, = lmsolve!(f!, j!, [0.0], 1)
    @test converged == 4
    @test x == [0.0] && F == 9.0

    # A domain wall: the objective is NaN past x > 0.5, so the minimizer x = 3
    # is unreachable and every trial past the wall is rejected.
    g(x, _) = SA[x[1] > 0.5 ? NaN : x[1] - 3.0]
    jg(x, _) = SA[1.0;;]
    x, F, converged, = lmsolve(g, jg, SA[0.0])
    @test converged == 4
    @test x[1] <= 0.5

    # A well-posed problem still converges with a positive code and reaches
    # its minimizer.
    h(x, _) = SA[x[1] - 3.0, 2 * (x[2] + 1.0)]
    jh(x, _) = SA[1.0 0.0; 0.0 2.0]
    x, F, converged, = lmsolve(h, jh, SA[0.0, 0.0])
    @test converged > 0
    @test x ≈ SA[3.0, -1.0]
end
