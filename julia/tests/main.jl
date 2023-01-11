
cd(@__DIR__)

if !@isdefined(Armon)
    include("../Armon.jl")
end

using .Armon
using Test

include("reference_data/reference_functions.jl")


if isinteractive()
    menu = """
    Tests available:
     - all            All tests below
     - short          Equivalent to 'code, stability, domains, convergence, conservation, kernels'
     - code           Code quality
     - stability      Type stability
     - domains        Domain 2D indexing
     - kernels        Compilation and correctness of indexing in generic kernels (CPU & GPU)
     - convergence    Convergence to the refenrence solutions
     - conservation   Check that the energy and mass for each are kept constant throughout a lot of cycles.
     - GPU            Equivalence of the GPU backends (CUDA & ROCm) with the CPU
     - performance    Checks for any regression in performance
     - async          Checks that separating the domain and treating the boundary conditions asynchronously 
                      doesn't introduce any variations in the result.
     - MPI            Equivalence with the single domain case and asynchronous communications

    Separate multiple test sets with a comma.

    Choice: """
    printstyled(stdout, menu; color=:light_green)
    raw_main_options = readline()
else
    raw_main_options = isempty(ARGS) ? "all" : join(ARGS, ", ")
end

main_options = split(raw_main_options, ',') .|> strip .|> lowercase
filter!(!isempty, main_options)
main_options = main_options .|> Symbol |> union

if :all in main_options
    main_options = [:quality, :stability, :domains, :convergence, :conservation, :kernels, 
                    :gpu, :performance, :async, :mpi]
elseif :short in main_options
    main_options = [:quality, :stability, :domains, :convergence, :conservation, :kernels]
end


function do_tests(tests_to_do)
    println("Testing: ", join(tests_to_do, ", "))

    ts = @testset "Armon tests" begin
        for test in tests_to_do
            if     test == :quality        include("code_quality.jl")
            elseif test == :stability      include("type_stability.jl")
            elseif test == :domains        include("domains.jl")
            elseif test == :convergence    include("convergence.jl")
            elseif test == :conservation   include("conservation.jl")
            elseif test == :kernels        include("kernels.jl")
            elseif test == :gpu            include("gpu.jl")
            elseif test == :performance    include("performance.jl")
            elseif test == :async          include("async.jl")
            elseif test == :mpi            include("mpi.jl")
            else
                error("Unknown test set: $test")
            end

            # TODO : suceptibility test comparing a result with different rounding modes
            # TODO : test lagrangian only mode
        end
    end

    # TODO: in Julia 1.8, there is one more option: 'showtimings' which display the time for each test
    if isinteractive()
        Test.print_test_results(ts)
    end
end


!isempty(main_options) && do_tests(main_options)
