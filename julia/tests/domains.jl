
import .Armon: shift, expand, prepend, inflate, shift_dir, inflate_dir, X_axis, Y_axis
import .Armon: DomainRange, StepsRanges, steps_ranges, update_axis_parameters, connect_ranges


@testset "Domains" begin
    @testset "Utilities $name" for (name, r, n) in (
                                        ("UnitRange", 1:10, 5), 
                                        ("OrdinalRange", 1:50:1100, 17), 
                                        ("Reverse UnitRange", 10:-1:1, 5),
                                        ("Reverse OrdinalRange", 1100:-50:1, 17))
        @testset "shift" begin
            rs  = shift(r,  n)
            rsn = shift(r, -n)

            @test length(rs)  == length(r)
            @test length(rsn) == length(r)
            @test first(rs)   == first(r) + n * step(r)
            @test last(rs)    == last(r)  + n * step(r)
            @test first(rsn)  == first(r) - n * step(r)
            @test last(rsn)   == last(r)  - n * step(r)
        end

        @testset "expand" begin
            rs  = expand(r,  n)
            rsn = expand(r, -n)

            @test length(rs)  == length(r) + n
            @test length(rsn) == max(length(r) - n, 0)
            @test first(rs)   == first(r)
            @test last(rs)    == last(r) + n * step(r)
            @test first(rsn)  == first(r)
            @test last(rsn)   == last(r) - n * step(r)

            @test length(expand(r,   - length(r))) == 0
            @test length(expand(r, 1 - length(r))) == 1
        end

        @testset "prepend" begin
            rs  = prepend(r,  n)
            rsn = prepend(r, -n)

            @test length(rs)  == length(r) + n
            @test length(rsn) == max(length(r) - n, 0)
            @test first(rs)   == first(r) - n * step(r)
            @test last(rs)    == last(r)
            @test first(rsn)  == first(r) + n * step(r)
            @test last(rsn)   == last(r)

            @test length(prepend(r,   - length(r))) == 0
            @test length(prepend(r, 1 - length(r))) == 1
        end

        @testset "inflate" begin
            rs  = inflate(r,  n)
            rsn = inflate(r, -n)

            @test length(rs)  == length(r) + 2n
            @test length(rsn) == max(length(r) - 2n, 0)
            @test first(rs)   == first(r) - n * step(r)
            @test last(rs)    == last(r)  + n * step(r)
            @test first(rsn)  == first(r) + n * step(r)
            if length(rsn) > 0
                @test last(rsn)   == last(r)  - n * step(r)
            end

            @test length(inflate(r, -length(r))) == 0
        end
    end

    @testset "DomainRange" begin
        main_r  = 50:30:600
        inner_r = 5:25
        dr = DomainRange(main_r, inner_r)

        @test first(dr) == first(main_r) + first(inner_r) - 1
        @test last(dr)  == last(main_r)  + last(inner_r)  - 1

        @test first(dr) in dr
        @test last(dr)  in dr
        @test !(0 in dr)
        @test !(typemax(Int64) in dr)
        @test !((first(dr) - 1) in dr)
        @test !((last(dr)  + 1) in dr)
        @test (main_r[3] + first(inner_r)) in dr
        @test !(main_r[3] in dr)

        n = 5
        sxdr = shift_dir(dr, X_axis, n)
        @test length(sxdr) == length(dr)
        @test first(sxdr)  == first(dr) + n
        @test last(sxdr)   == last(dr)  + n

        sydr = shift_dir(dr, Y_axis, n)
        @test length(sydr) == length(dr)
        @test first(sydr)  == first(dr) + n * step(main_r)
        @test last(sydr)   == last(dr)  + n * step(main_r)

        ixdr = inflate_dir(dr, X_axis, n)
        @test length(ixdr) == length(dr) + 2n * length(main_r)
        @test first(ixdr)  == first(dr) - n
        @test last(ixdr)   == last(dr)  + n

        iydr = inflate_dir(dr, Y_axis, n)
        @test length(iydr) == length(dr) + 2n * length(inner_r)
        @test first(iydr)  == first(dr) - n * step(main_r)
        @test last(iydr)   == last(dr)  + n * step(main_r)
    end

    @testset "StepsRanges $(nx)×$(ny) $ng ghosts" for (nx, ny, ng) in (
                                                            (13, 7, 4), (13, 7, 3), (7, 7, 5),
                                                            (7, 13, 4), (7, 13, 3), (7, 7, 5),
                                                            (40, 40, 5))
        p = ArmonParameters(;
            scheme = :GAD, projection = :euler_2nd,
            nghost = ng, nx, ny,
            use_MPI = false
        )

        @testset "$axis" for axis in (X_axis, Y_axis)
            update_axis_parameters(p, axis)
            steps = steps_ranges(p)

            splitted_range = Base.Fix2(getfield, axis == X_axis ? :row : :col)

            if isempty(steps.inner_EOS)
                merged_EOS_range = splitted_range(steps.outer_lb_EOS)
            else
                merged_EOS_range = connect_ranges(
                    splitted_range(steps.outer_lb_EOS),
                    splitted_range(steps.inner_EOS))
            end

            merged_EOS_range = connect_ranges(
                merged_EOS_range,
                splitted_range(steps.outer_rt_EOS))

            @test splitted_range(steps.EOS) == merged_EOS_range

            if isempty(steps.inner_fluxes)
                merged_fluxes_range = splitted_range(steps.outer_lb_fluxes)
            else
                merged_fluxes_range = connect_ranges(
                    splitted_range(steps.outer_lb_fluxes),
                    splitted_range(steps.inner_fluxes))
            end

            merged_fluxes_range = connect_ranges(
                merged_fluxes_range,
                splitted_range(steps.outer_rt_fluxes))

            @test splitted_range(steps.fluxes) == merged_fluxes_range
        end
    end
end
