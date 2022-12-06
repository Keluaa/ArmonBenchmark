
if !@isdefined(Armon)
    include("../armon_2D_MPI_async.jl")
end

using .Armon
using Test


if isinteractive()
    menu = 
"""
Tests available:
 - all            All tests below
 - short          Equivalent to 'code, stability, convergence, kernels'
 - code           Code quality
 - stability      Type stability
 - kernels        Compilation and correctness of indexing in generic kernels (CPU & GPU)
 - convergence    Convergence to the refenrence solutions
 - GPU            Equivalence of the GPU backends (CUDA & ROCm) with the CPU
 - performance    Checks for any regression in performance
 - mpi            Equivalence with the single domain case and asynchronous communications

Separate test multiple sets with a comma.

Additionally, putting 'verbose' in the list of tests will display the entire tree of tests results.

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
    main_options = [:quality, :stability, :convergence, :kernels, :gpu, :performance, :mpi]
elseif :short in main_options
    main_options = [:quality, :stability, :convergence, :kernels]
end

if :verbose in main_options
    verbose = true
    deleteat!(main_options, findall(x -> x == :verbose, main_options))
else
    verbose = false
end


function override_verbosity(ts::Test.DefaultTestSet, verbose::Bool)
    ts.verbose = verbose
    for sub_ts in ts.results
        if sub_ts isa Test.DefaultTestSet
            override_verbosity(ts, verbose)
        end
    end
    return ts
end


function do_tests(tests_to_do; verbose=true)
    println("Testing: ", join(tests_to_do, ", "))

    tmp_print = Test.TESTSET_PRINT_ENABLE[]

    ts = @testset "Armon tests" begin
        for test in tests_to_do
            if     test == :quality        include("code_quality.jl")
            elseif test == :stability      include("type_stability.jl")
            elseif test == :convergence    include("convergence.jl")
            elseif test == :kernels        include("kernels.jl")
            elseif test == :gpu            include("gpu.jl")
            elseif test == :performance    include("performance.jl")
            elseif test == :mpi            include("mpi.jl")
            else
                error("Unknown test set: $test")
            end

            # TODO : test domains (and remove the file in the 'julia' dir)
            # TODO : suceptibility test comparing a result with different rounding modes
            # TODO : test lagrangian only mode
        end

        # Disable printing at the end so that we can print the results as we like without hiding any
        # errors.
        Test.TESTSET_PRINT_ENABLE[] = false
    end

    Test.TESTSET_PRINT_ENABLE[] = tmp_print

    # TODO: in Julia 1.8, there is one more option: 'showtimings' which display the time for each test
    override_verbosity(ts, verbose)
    Test.print_test_results(ts)
end


!isempty(main_options) && do_tests(main_options; verbose)
