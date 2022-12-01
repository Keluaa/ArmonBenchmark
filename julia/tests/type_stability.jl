
import .Armon: limiter, NoLimiter, MinmodLimiter, SuperbeeLimiter

@testset "Type stability" begin
    @testset "Limiters" begin
        for type in (Float32, Float64)
            x = type(0.456)
            @test type == typeof(@inferred limiter(x, NoLimiter()))
            @test type == typeof(@inferred limiter(x, MinmodLimiter()))
            @test type == typeof(@inferred limiter(x, SuperbeeLimiter()))
        end
    end
end
