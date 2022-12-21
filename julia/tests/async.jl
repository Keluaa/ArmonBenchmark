
no_threading = Threads.nthreads() == 1

no_threading && @warn "Async tests require more than one thread to work."


@testset "Async" begin
    # TODO
end
