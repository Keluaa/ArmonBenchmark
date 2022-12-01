
@testset "Code quality" begin
    # TODO : better code quality testing using Aqua.jl: https://github.com/JuliaTesting/Aqua.jl
    @test isempty(Test.detect_ambiguities(Armon))
    @test isempty(Test.detect_unbound_args(Armon))
end
