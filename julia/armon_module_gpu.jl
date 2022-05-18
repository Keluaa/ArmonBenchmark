module Armon

using Printf
using Polyester
using KernelAbstractions
using KernelAbstractions.Extras: @unroll

use_ROCM = haskey(ENV, "USE_ROCM_GPU") && ENV["USE_ROCM_GPU"] == "true"

if use_ROCM
	using AMDGPU
	using ROCKernels
	println("Using ROCM GPU")
else
	using CUDA
	using CUDAKernels
	CUDA.allowscalar(false)
	println("Using CUDA GPU")
end


export ArmonParameters, armon

#
# Parameters
# 

mutable struct ArmonParameters{Flt_T}
	# Floating point type (Float32 or Float64)
	data_type::Type

	# Test problem type, riemann solver and solver scheme
	test::Symbol
	riemann::Symbol
	scheme::Symbol
	
	# Solver parameters
	iterations::Int
	
	# Domain parameters
	nghost::Int
	nbcell::Int
	cfl::Flt_T
	Dt::Flt_T
	ideb::Int
	ifin::Int
	euler_projection::Bool
	cst_dt::Bool

	# Bounds
	maxtime::Flt_T
	maxcycle::Int
	
	# Output
	silent::Int
	write_output::Bool
	measure_time::Bool

	# Performance
	use_ccall::Bool
	use_threading::Bool
	use_simd::Bool
	interleaving::Bool
	use_gpu::Bool
end


# Constructor for ArmonParameters
function ArmonParameters(; ieee_bits = 64,
						   test = :Sod, riemann = :acoustic, scheme = :GAD_minmod,
						   iterations = 4, 
						   nghost = 2, nbcell = 100, cfl = 0.6, Dt = 0., euler_projection = false, cst_dt = false,
						   maxtime = 0, maxcycle = 500_000,
						   silent = 0, write_output = true, measure_time = true,
						   use_ccall = false, use_threading = true, 
						   use_simd = true, interleaving = false,
						   use_gpu = false)

	flt_type = (ieee_bits == 64) ? Float64 : Float32

	# Make sure that all floating point types are the same
	cfl = flt_type(cfl)
	Dt = flt_type(Dt)
	maxtime = flt_type(maxtime)
	
	if riemann == :one_iteration_acoustic iterations = 1 end
	if riemann == :two_iteration_acoustic iterations = 2 end

	if cst_dt && Dt == zero(flt_type)
		error("Dt == 0 with constant step enabled")
	end

	ideb = nghost + 1
	ifin = nghost + nbcell
	
	return ArmonParameters{flt_type}(flt_type, 
									 test, riemann, scheme, 
									 iterations, 
									 nghost, nbcell, cfl, Dt, ideb, ifin, euler_projection, cst_dt,
									 maxtime, maxcycle,
									 silent, write_output, measure_time,
									 use_ccall, use_threading,
									 use_simd, interleaving,
									 use_gpu)
end


function print_parameters(p::ArmonParameters)
	println("Parameters:")
	print(" - multithreading: ", p.use_threading)
	if p.use_threading
		if p.use_ccall
			println(" (OMP threads: ", ENV["OMP_NUM_THREADS"], ")")
		else
			println(" (Julia threads: ", Threads.nthreads(), ")")
		end
	else
		println("")
	end
	println(" - interleaving: ", p.interleaving)
	println(" - use_simd:   ", p.use_simd)
	println(" - use_ccall:  ", p.use_ccall)
	println(" - use_gpu:    ", p.use_gpu)
	println(" - ieee_bits:  ", sizeof(p.data_type) * 8)
	println("")
	println(" - test:       ", p.test)
	println(" - riemann:    ", p.riemann)
	println(" - scheme:     ", p.scheme)
	println(" - iterations: ", p.iterations)
	println(" - nbcell:     ", p.nbcell)
	println(" - cfl:        ", p.cfl)
	println(" - Dt:         ", p.Dt)
	println(" - euler proj: ", p.euler_projection)
	println(" - cst dt:     ", p.cst_dt)
	println(" - maxtime:    ", p.maxtime)
	println(" - maxcycle:   ", p.maxcycle)
end

#
# Threading and SIMD control macros
#

"""
Allows to enable/disable multithreading of the loop depending on the parameters.

	@threaded for i = 1:n
		y[i] = log10(x[i]) + x[i]
	end
"""
macro threaded(expr)
	# if !@isdefined params
	# 	throw(ArgumentError("Expected a 'param' variable in the current scope"))
	# end

	return esc(quote
		if params.use_threading
			@inbounds @batch $(expr)
		else
			$(expr)
		end
	end)
end


"""
	@simd_threaded_loop(expr)

Allows to enable/disable multithreading and/or SIMD of the loop depending on the parameters.
When using SIMD, `@fastmath` and `@inbounds` are used.

In order to use SIMD and multithreading at the same time, the range of the loop is split in even batches.
Each batch has a size of `params.simd_batch` iterations, meaning that the inner `@simd` loop has a fixed number of iterations,
while the outer threaded loop will have `N ÷ params.simd_batch` iterations.

The inner `@simd` loop assumes there is no dependencies between each iteration.

```julia
	@simd_threaded_loop for i = 1:n
		y[i] = log10(x[i]) + x[i]
	end
```
"""
macro simd_threaded_loop(expr)
	if !Meta.isexpr(expr, :for, 2)
        throw(ArgumentError("Expected a valid for loop"))
    end

	#if !@isdefined params
	#	throw(ArgumentError("Expected a 'param' variable in the current scope"))
	#end

	# Only in for the case of a threaded loop with SIMD:
	# Extract the range of the loop and replace it with the new variables
	modified_loop_expr = copy(expr)
    range_expr = modified_loop_expr.args[1]
    loop_range = copy(range_expr.args[2])  # The original loop range
	range_expr.args[2] = :(__ideb:__ifin)  # The new loop range

	# Same but with interleaving
	interleaved_loop_expr = copy(expr)
	range_expr = interleaved_loop_expr.args[1]
	range_expr.args[2] = :((__first_i + __i_thread):__num_threads:__last_i)  # The interleaved loop range

	return esc(quote
		if params.use_threading
			if params.use_simd
				if params.interleaving
					__loop_range = $(loop_range)
					__total_iter = length(__loop_range)
					__num_threads = Threads.nthreads()
					__first_i = first(__loop_range)
					__last_i = last(__loop_range)
					@batch for __i_thread = 1:__num_threads
						@fastmath @inbounds @simd ivdep $(interleaved_loop_expr)
					end
				else
					__loop_range = $(loop_range)
					__total_iter = length(__loop_range)
					__num_threads = Threads.nthreads()
					__batch = convert(Int, cld(__total_iter, __num_threads))::Int  # __total_iter ÷ __num_threads
					__first_i = first(__loop_range)
					__last_i = last(__loop_range)
					@batch for __i_thread = 1:__num_threads
						__ideb = __first_i + (__i_thread - 1) * __batch
						__ifin = ifelse(__i_thread == __num_threads, __last_i, __ideb + __batch)
						@fastmath @inbounds @simd ivdep $(modified_loop_expr)
					end
				end
			else
				@inbounds @batch $(expr)
			end
		else
			if params.use_simd
				@fastmath @inbounds @simd ivdep $(expr)
			else
				$(expr)
			end
		end
	end)
end

#
# Execution Time Measurement
#

time_contrib = Dict{String, Float64}()
macro time_pos(params, label, expr) 
	return esc(quote
		if params.measure_time
			_t_start = time_ns()
			$(expr)
			_t_end = time_ns()
			if haskey(time_contrib, $(label))
				global time_contrib[$(label)] += _t_end - _t_start
			else
				global time_contrib[$(label)] = _t_end - _t_start
			end
		else
			$(expr)
		end
	end)
end

#
# GPU Kernels
#

const device = use_ROCM ? ROCDevice() : CUDADevice()
const block_size = haskey(ENV, "GPU_BLOCK_SIZE") ? parse(Int, ENV["GPU_BLOCK_SIZE"]) : 32
const reduction_block_size = 1024;
const reduction_block_size_log2 = convert(Int, log2(reduction_block_size))

@kernel function gpu_acoustic_kernel!(i_0, ustar, pstar, @Const(rho), @Const(umat), @Const(pmat), @Const(cmat))
	i = @index(Global) + i_0
	rc_l = rho[i-1]*cmat[i-1]
	rc_r = rho[i]*cmat[i]
	ustar[i] = ( rc_l*umat[i-1] + rc_r*umat[i] + (pmat[i-1] - pmat[i]) ) / (rc_l + rc_r)
	pstar[i] = ( rc_r*pmat[i-1] + rc_l*pmat[i] + rc_l*rc_r*(umat[i-1] - umat[i]) ) / (rc_l + rc_r)
end


function gpu_acoustic!(ideb, ifin, ustar, pstar, rho, umat, pmat, cmat)
	kernel = gpu_acoustic_kernel!(device, block_size)
	event = kernel(ideb - 1, ustar, pstar, rho, umat, pmat, cmat, ndrange=length(ideb:ifin+1))
    wait(event)
end


@kernel function gpu_acoustic_GAD_minmod_kernel!(i_0, ustar, pstar, @Const(rho), @Const(umat), @Const(pmat), @Const(cmat), @Const(ustar_1), @Const(pstar_1), dt, @Const(x))
	i = @index(Global) + i_0

	r_u_m = (ustar_1[i+1] - umat[i]) / (ustar_1[i] - umat[i-1] + 1e-6)
	r_p_m = (pstar_1[i+1] - pmat[i]) / (pstar_1[i] - pmat[i-1] + 1e-6)
	r_u_p = (umat[i-1] - ustar_1[i-1]) / (umat[i] - ustar_1[i] + 1e-6)
	r_p_p = (pmat[i-1] - pstar_1[i-1]) / (pmat[i] - pstar_1[i] + 1e-6)

	r_u_m = max(0., min(1., r_u_m))
	r_p_m = max(0., min(1., r_p_m))
	r_u_p = max(0., min(1., r_u_p))
	r_p_p = max(0., min(1., r_p_p))

	dm_l = rho[i-1]*(x[i]-x[i-1])
	dm_r = rho[i]  *(x[i+1]-x[i])
	Dm = (dm_l + dm_r) / 2
	rc_l = rho[i-1]*cmat[i-1]
	rc_r = rho[i]*cmat[i]
	θ = (rc_l + rc_r) / 2 * (dt / Dm)
	
	ustar[i] = ustar_1[i] + 1/2 * (1 - θ) * (r_u_p*(umat[i] - ustar_1[i]) - r_u_m*(ustar_1[i] - umat[i-1]))
	pstar[i] = pstar_1[i] + 1/2 * (1 - θ) * (r_p_p*(pmat[i] - pstar_1[i]) - r_p_m*(pstar_1[i] - pmat[i-1]))
end


function gpu_acoustic_GAD_minmod!(ideb, ifin, ustar, pstar, rho, umat, pmat, cmat, ustar_1, pstar_1, dt, x)
	kernel = gpu_acoustic_GAD_minmod_kernel!(device, block_size)
	event = kernel(ideb - 1, ustar, pstar, rho, umat, pmat, cmat, ustar_1, pstar_1, dt, x, ndrange=length(ideb:ifin+1))
    wait(event)
end


@kernel function gpu_cell_update_lagrange_kernel!(i_0, ifin, dt, x_, X, @Const(ustar), @Const(pstar), rho, umat, emat, Emat)
	i = @index(Global) + i_0

	X[i] = x_[i] + dt*ustar[i]

	if i == ifin
		X[i+1] = x_[i+1] + dt*ustar[i+1]
	end

	@synchronize

	dm = rho[i]*(x_[i+1]-x_[i])
	rho[i] = dm/(x_[i+1] + dt*ustar[i+1]-(x_[i] + dt*ustar[i]))  # We must use this instead of X[i+1]-X[i] since X can be overwritten by other workgroups
	umat[i] = umat[i] + dt/dm*(pstar[i]-pstar[i+1])
	Emat[i] = Emat[i] + dt/dm*(pstar[i]*ustar[i]-pstar[i+1]*ustar[i+1])
	emat[i] = Emat[i] - 0.5*umat[i]^2

	@synchronize

	x_[i] = X[i]

	if i == ifin
		x_[i+1] = X[i+1]
	end
end


@kernel function gpu_cell_update_euler_kernel!(i_0, ifin, dt, x_, X, @Const(ustar), @Const(pstar), rho, umat, emat, Emat)
	i = @index(Global) + i_0

	X[i] = x_[i] + dt*ustar[i]

	if i == ifin
		X[i+1] = x_[i+1] + dt*ustar[i+1]
	end

	@synchronize

	dm = rho[i]*(x_[i+1]-x_[i])
	dx = x_[i+1] + dt*ustar[i+1]-(x_[i] + dt*ustar[i]) # We must use this instead of X[i+1]-X[i] since X can be overwritten by other workgroups
	rho[i] = dm/dx  
	umat[i] = umat[i] + dt/dm*(pstar[i]-pstar[i+1])
	Emat[i] = Emat[i] + dt/dm*(pstar[i]*ustar[i]-pstar[i+1]*ustar[i+1])
	emat[i] = Emat[i] - 0.5*umat[i]^2
end


function gpu_cell_update!(euler_proj, ideb, ifin, dt, x, X, ustar, pstar, rho, umat, emat, Emat, tmp_rho, tmp_urho, tmp_Erho)
	if euler_proj
		kernel = gpu_cell_update_euler_kernel!(device, block_size)
		event = kernel(ideb - 1, ifin, dt, x, X, ustar, pstar, rho, umat, emat, Emat, ndrange=length(ideb:ifin))
		wait(event)
	else
		kernel = gpu_cell_update_lagrange_kernel!(device, block_size)
		event = kernel(ideb - 1, ifin, dt, x, X, ustar, pstar, rho, umat, emat, Emat, ndrange=length(ideb:ifin))
		wait(event)
	end
end


@kernel function gpu_first_order_euler_remap_kernel!(i_0, dt, X, @Const(ustar), rho, umat, Emat, tmp_rho, tmp_urho, tmp_Erho)
	i = @index(Global) + i_0

	dx = X[i+1] - X[i]
	L₁ = max(0, ustar[i]) * dt
	L₃ = -min(0, ustar[i+1]) * dt
	L₂ = dx - L₁ - L₃
	
	tmp_rho[i]  = (L₁ * rho[i-1] 
				 + L₂ * rho[i] 
				 + L₃ * rho[i+1]) / dx
	tmp_urho[i] = (L₁ * rho[i-1] * umat[i-1] 
				 + L₂ * rho[i]   * umat[i] 
				 + L₃ * rho[i+1] * umat[i+1]) / dx
	tmp_Erho[i] = (L₁ * rho[i-1] * Emat[i-1] 
				 + L₂ * rho[i]   * Emat[i] 
				 + L₃ * rho[i+1] * Emat[i+1]) / dx
end


@kernel function gpu_first_order_euler_remap_2_kernel!(i_0, rho, umat, Emat, tmp_rho, tmp_urho, tmp_Erho)
	i = @index(Global) + i_0

	# (ρ, ρu, ρE) -> (ρ, u, E)
	rho[i] = tmp_rho[i]
	umat[i] = tmp_urho[i] / tmp_rho[i]
	Emat[i] = tmp_Erho[i] / tmp_rho[i]
end


function gpu_first_order_euler_remap!(ideb, ifin, dt, X, ustar, rho, umat, Emat, tmp_rho, tmp_urho, tmp_Erho)
	kernel_1 = gpu_first_order_euler_remap_kernel!(device, block_size)
	kernel_2 = gpu_first_order_euler_remap_2_kernel!(device, block_size)

	event = kernel_1(ideb - 1, dt, X, ustar, rho, umat, Emat, tmp_rho, tmp_urho, tmp_Erho, ndrange=length(ideb:ifin))
	wait(event)
	event = kernel_2(ideb - 1, rho, umat, Emat, tmp_rho, tmp_urho, tmp_Erho, ndrange=length(ideb:ifin))
	wait(event)
end


@kernel function gpu_update_perfect_gas_EOS_kernel!(i_0, gamma, @Const(rho), @Const(emat), pmat, cmat, gmat)
	i = @index(Global) + i_0
	
	pmat[i] = (gamma-1.)*rho[i]*emat[i]
	cmat[i] = sqrt(gamma*pmat[i]/rho[i])
	gmat[i] = (1+gamma)/2
end


@kernel function gpu_update_bizarrium_EOS_kernel!(i_0, @Const(rho), @Const(emat), pmat, cmat, gmat)
	i = @index(Global) + i_0

	data_type = eltype(rho)
	
	rho0::data_type = 10000.; 
	K0::data_type 	= 1e+11; 
	Cv0::data_type 	= 1000.; 
	T0::data_type 	= 300.; 
	eps0::data_type = 0.; 
	S0::data_type 	= 0.; 
	G0::data_type 	= 1.5; 
	s::data_type 	= 1.5; 
	q::data_type 	= -42080895/14941154; 
	r::data_type 	= 727668333/149411540;

	x::data_type = rho[i]/rho0 - 1
	g::data_type = G0*(1-rho0/rho[i]); # formula (4b)

	f0::data_type = (1+(s/3-2)*x+q*x^2+r*x^3)/(1-s*x)       # Formula (15b)
	f1::data_type = (s/3-2+2*q*x+3*r*x^2+s*f0)/(1-s*x)      # Formula (16a)
	f2::data_type = (2*q+6*r*x+2*s*f1)/(1-s*x)              # Formula (16b)
	f3::data_type = (6*r+3*s*f2)/(1-s*x)                    # Formula (16c)

	epsk0::data_type     = eps0 - Cv0*T0*(1+g) + 0.5*(K0/rho0)*x^2*f0                             # Formula (15a)
	pk0::data_type       = -Cv0*T0*G0*rho0 + 0.5*K0*x*(1+x)^2*(2*f0+x*f1)                         # Formula (17a)
	pk0prime::data_type  = -0.5*K0*(1+x)^3*rho0 * (2*(1+3x)*f0 + 2*x*(2+3x)*f1 + x^2*(1+x)*f2)    # Formulae (17b) and (17c)
	pk0second::data_type = 0.5*K0*(1+x)^4*rho0^2 * (12*(1+2x)*f0 + 6*(1+6x+6*x^2)*f1 + 6*x*(1+x)*(1+2x)*f2 + x^2*(1+x)^2*f3)

	pmat[i] = pk0 + G0*rho0*(emat[i] - epsk0)                                         # Formula (5b)
	cmat[i] = sqrt(G0*rho0*(pmat[i] - pk0) - pk0prime) / rho[i]                       # Formula (8)

	gmat[i] = 0.5/(rho[i]^3*cmat[i]^2)*(pk0second+(G0*rho0)^2*(pmat[i]-pk0))          # Formula (8) + (11)
end


function gpu_update_EOS!(test, ideb, ifin, rho, emat, pmat, cmat, gmat)
	if test == :Bizarrium
		kernel = gpu_update_bizarrium_EOS_kernel!(device, block_size)
		event = kernel(ideb - 1, rho, emat, pmat, cmat, gmat, ndrange=length(ideb:ifin))
	else
		gamma::eltype(rho) = 0.0

		if test == :Sod || test == :Woodward
			gamma = 1.4
		elseif test == :Leblanc
			gamma = 5/3
		end

		kernel = gpu_update_perfect_gas_EOS_kernel!(device, block_size)
		event = kernel(ideb - 1, gamma, rho, emat, pmat, cmat, gmat, ndrange=length(ideb:ifin))
	end

    wait(event)
end


@kernel function gpu_boundary_conditions_kernel!(test_bizarrium, ideb, ifin, rho, umat, pmat, cmat, gmat)
	rho[ideb-1]  = rho[ideb]
	umat[ideb-1] = -umat[ideb]
	pmat[ideb-1] = pmat[ideb]
	cmat[ideb-1] = cmat[ideb]
	gmat[ideb-1] = gmat[ideb]

	rho[ifin+1]  = rho[ifin]
	
	pmat[ifin+1] = pmat[ifin]
	cmat[ifin+1] = cmat[ifin]
	gmat[ifin+1] = gmat[ifin]

	if test_bizarrium
		umat[ifin+1] = umat[ifin]
	else
		umat[ifin+1] = -umat[ifin]
	end
end


function gpu_boundary_conditions!(test, ideb, ifin, rho, umat, pmat, cmat, gmat)
	test_bizarrium = test == :Bizarrium
	kernel = gpu_boundary_conditions_kernel!(device, 1, 1)
	event = kernel(test_bizarrium, ideb, ifin, rho, umat, pmat, cmat, gmat)
    wait(event)
end


@kernel function gpu_dtCFL_reduction_kernel!(euler, ideb, ifin, x, cmat, umat, result)
	tid = @index(Local)

    values = @localmem eltype(x) reduction_block_size

    min_val_thread::eltype(x) = Inf
	if euler
		for i in ideb+tid-1:reduction_block_size:ifin
			dt_i = (x[i+1] - x[i]) / max(abs(umat[i] + cmat[i]), abs(umat[i] - cmat[i]))
			min_val_thread = min(min_val_thread, dt_i)
		end
	else
		for i in ideb+tid-1:reduction_block_size:ifin
			dt_i = (x[i+1] - x[i]) / cmat[i]
			min_val_thread = min(min_val_thread, dt_i)
		end
	end
    values[tid] = min_val_thread

    @synchronize
    
    step_size = reduction_block_size >> 1
    
	@unroll for _ in 1:reduction_block_size_log2
        if tid <= step_size
            values[tid] = min(values[tid], values[tid + step_size])
        end

        step_size >>= 1

        @synchronize
	end

    if tid == 1
        result[1] = values[1]
    end
end


function gpu_dtCFL_reduction!(euler, ideb, ifin, x, cmat, umat)
	result = zeros(eltype(x), 1)
	d_result = use_ROCM ? ROCArray(result) : CuArray(result)
	kernel = gpu_dtCFL_reduction_kernel!(device, reduction_block_size, reduction_block_size)
	event = kernel(euler, ideb, ifin, x, cmat, umat, d_result)
    wait(event)
	copyto!(result, d_result)
	return result[1]
end


#
# Equations of State
#

function perfectGasEOS!(params, pmat, cmat, gmat, rho, emat, gamma)
	(; ideb, ifin) = params
	@simd_threaded_loop for i in ideb:ifin
		pmat[i] = (gamma-1.)*rho[i]*emat[i]
		cmat[i] = sqrt(gamma*pmat[i]/rho[i])
		gmat[i] = (1+gamma)/2
	end
end


function BizarriumEOS!(params, pmat, cmat, gmat, rho, emat)
	(; ideb, ifin) = params

	###  O. Heuzé, S. Jaouen, H. Jourdren, "Dissipative issue of high-order shock capturing schemes wtih non-convex equations of state", JCP 2009)

	rho0 = 10000; K0 = 1e+11; Cv0 = 1000; T0 = 300; eps0 = 0; S0 = 0; G0 = 1.5; s = 1.5; q = -42080895/14941154; r = 727668333/149411540

	@simd_threaded_loop for i in ideb:ifin

		x = rho[i]/rho0 - 1; g = G0*(1-rho0/rho[i]); # formula (4b)

		f0 = (1+(s/3-2)*x+q*x^2+r*x^3)/(1-s*x)       # Formula (15b)
		f1 = (s/3-2+2*q*x+3*r*x^2+s*f0)/(1-s*x)      # Formula (16a)
		f2 = (2*q+6*r*x+2*s*f1)/(1-s*x)              # Formula (16b)
		f3 = (6*r+3*s*f2)/(1-s*x)                    # Formula (16c)

		epsk0 = eps0 - Cv0*T0*(1+g) + 0.5*(K0/rho0)*x^2*f0                                # Formula (15a)
		pk0 = -Cv0*T0*G0*rho0 + 0.5*K0*x*(1+x)^2*(2*f0+x*f1)                              # Formula (17a)
		pk0prime = -0.5*K0*(1+x)^3*rho0 * (2*(1+3x)*f0 + 2*x*(2+3x)*f1 + x^2*(1+x)*f2)    # Formulae (17b) and (17c)
		pk0second = 0.5*K0*(1+x)^4*rho0^2 * (12*(1+2x)*f0 + 6*(1+6x+6*x^2)*f1 + 6*x*(1+x)*(1+2x)*f2 + x^2*(1+x)^2*f3)

		pmat[i] = pk0 + G0*rho0*(emat[i] - epsk0)                                         # Formula (5b)
		cmat[i] = sqrt(G0*rho0*(pmat[i] - pk0) - pk0prime) / rho[i]                       # Formula (8)

		gmat[i] = 0.5/(rho[i]^3*cmat[i]^2)*(pk0second+(G0*rho0)^2*(pmat[i]-pk0))          # Formula (8) + (11)
	end
end

#
# Acoustic Riemann problem solvers
# 

function acoustic!(params, ustar, pstar, rho, umat, pmat, cmat, dt, x)
	(; ideb, ifin) = params
	if params.use_ccall
		# void acoustic(double* restrict ustar, double* restrict pstar, 
        #       const double* restrict rho, const double* restrict cmat,
        #       const double* restrict umat, const double* restrict pmat, 
        #       int ideb, int ifin)
		ccall((:acoustic, "./libacoustic.so"), Cvoid, (
				Ref{Float64}, Ref{Float64}, 
				Ref{Float64}, Ref{Float64}, 
				Ref{Float64}, Ref{Float64}, 
				Int32, Int32),
			ustar, pstar, rho, cmat, umat, pmat, ideb, ifin)
	elseif params.use_gpu
		gpu_acoustic!(ideb, ifin, ustar, pstar, rho, umat, pmat, cmat)
	else
		@simd_threaded_loop for i in ideb:ifin+1
			rc_l = rho[i-1]*cmat[i-1]
			rc_r = rho[i]*cmat[i]
			ustar[i] = ( rc_l*umat[i-1] + rc_r*umat[i] +           (pmat[i-1] - pmat[i]) ) / (rc_l + rc_r)
			pstar[i] = ( rc_r*pmat[i-1] + rc_l*pmat[i] + rc_l*rc_r*(umat[i-1] - umat[i]) ) / (rc_l + rc_r)
		end
	end
end


function acoustic_GAD!(params, ustar, pstar, rho, umat, pmat, cmat, ustar_1, pstar_1, dt, x)
	(; scheme, ideb, ifin) = params

	if params.use_ccall
		scheme_int::Int32 = (scheme == :GAD_minmod) ? 1 : ((scheme == :GAD_superbee) ? 2 : 0)

		# void acoustic_GAD(double* restrict ustar, double* restrict pstar, 
        #           double* restrict ustar_1, double* restrict pstar_1,
        #           const double* restrict rho, const double* restrict cmat,
        #           const double* restrict umat, const double* restrict pmat, 
        #           const double* restrict x,
        #           double dt, int ideb, int ifin,
		#           int scheme)
		ccall((:acoustic_GAD, "./libacoustic.so"), Cvoid, (
				Ref{Float64}, Ref{Float64}, 
				Ref{Float64}, Ref{Float64}, 
				Ref{Float64}, Ref{Float64}, 
				Ref{Float64}, Ref{Float64}, 
				Ref{Float64},
				Int32, Int32,
				Int32),
			ustar, pstar, ustar_1, pstar_1, rho, cmat, umat, pmat, x, ideb, ifin, scheme_int)

		return
	elseif params.use_gpu
		if params.scheme != :GAD_minmod
			println("Only the limiter minmod is implemented for GPU")
			exit()
		end
		gpu_acoustic!(ideb, ifin, ustar_1, pstar_1, rho, umat, pmat, cmat)
		gpu_acoustic_GAD_minmod!(ideb, ifin, ustar, pstar, rho, umat, pmat, cmat, ustar_1, pstar_1, dt, x)
		return
	end

	# First order
	@simd_threaded_loop for i in ideb:ifin+1
		rc_l = rho[i-1]*cmat[i-1]
		rc_r = rho[i]*cmat[i]
		ustar_1[i] = ( rc_l*umat[i-1] + rc_r*umat[i] +           (pmat[i-1] - pmat[i]) ) / (rc_l + rc_r)
		pstar_1[i] = ( rc_r*pmat[i-1] + rc_l*pmat[i] + rc_l*rc_r*(umat[i-1] - umat[i]) ) / (rc_l + rc_r)
	end
	
	# Second order
	if scheme == :GAD_minmod
		@simd_threaded_loop for i in ideb:ifin+1
			r_u_m = (ustar_1[i+1] - umat[i]) / (ustar_1[i] - umat[i-1] + 1e-6)
			r_p_m = (pstar_1[i+1] - pmat[i]) / (pstar_1[i] - pmat[i-1] + 1e-6)
			r_u_p = (umat[i-1] - ustar_1[i-1]) / (umat[i] - ustar_1[i] + 1e-6)
			r_p_p = (pmat[i-1] - pstar_1[i-1]) / (pmat[i] - pstar_1[i] + 1e-6)

			r_u_m = max(0., min(1., r_u_m))
			r_p_m = max(0., min(1., r_p_m))
			r_u_p = max(0., min(1., r_u_p))
			r_p_p = max(0., min(1., r_p_p))

			dm_l = rho[i-1]*(x[i]-x[i-1])
			dm_r = rho[i]  *(x[i+1]-x[i])
			Dm = (dm_l + dm_r) / 2
			rc_l = rho[i-1]*cmat[i-1]
			rc_r = rho[i]*cmat[i]
			θ = (rc_l + rc_r) / 2 * (dt / Dm)
			
			ustar[i] = ustar_1[i] + 1/2 * (1 - θ) * (r_u_p*(umat[i] - ustar_1[i]) - r_u_m*(ustar_1[i] - umat[i-1]))
			pstar[i] = pstar_1[i] + 1/2 * (1 - θ) * (r_p_p*(pmat[i] - pstar_1[i]) - r_p_m*(pstar_1[i] - pmat[i-1]))
		end
	elseif scheme == :GAD_superbee
		@simd_threaded_loop for i in ideb:ifin+1
			r_u_m = (ustar_1[i+1] - umat[i]) / (ustar_1[i] - umat[i-1] + 1e-6)
			r_p_m = (pstar_1[i+1] - pmat[i]) / (pstar_1[i] - pmat[i-1] + 1e-6)
			r_u_p = (umat[i-1] - ustar_1[i-1]) / (umat[i] - ustar_1[i] + 1e-6)
			r_p_p = (pmat[i-1] - pstar_1[i-1]) / (pmat[i] - pstar_1[i] + 1e-6)

			r_u_m = max(0., min(1., 2. * r_u_m), min(2., r_u_m))
			r_p_m = max(0., min(1., 2. * r_p_m), min(2., r_p_m))
			r_u_p = max(0., min(1., 2. * r_u_p), min(2., r_u_p))
			r_p_p = max(0., min(1., 2. * r_p_p), min(2., r_p_p))
	
			dm_l = rho[i-1]*(x[i]-x[i-1])
			dm_r = rho[i]  *(x[i+1]-x[i])
			Dm = (dm_l + dm_r) / 2
			rc_l = rho[i-1]*cmat[i-1]
			rc_r = rho[i]*cmat[i]
			θ = (rc_l + rc_r) / 2 * (dt / Dm)
			
			ustar[i] = ustar_1[i] + 1/2 * (1 - θ) * (r_u_p*(umat[i] - ustar_1[i]) - r_u_m*(ustar_1[i] - umat[i-1]))
			pstar[i] = pstar_1[i] + 1/2 * (1 - θ) * (r_p_p*(pmat[i] - pstar_1[i]) - r_p_m*(pstar_1[i] - pmat[i-1]))
		end
	elseif scheme == :GAD_no_limiter
		@simd_threaded_loop for i in ideb:ifin+1
			dm_l = rho[i-1]*(x[i]-x[i-1])
			dm_r = rho[i]  *(x[i+1]-x[i])
			Dm = (dm_l + dm_r) / 2
			rc_l = rho[i-1]*cmat[i-1]
			rc_r = rho[i]*cmat[i]
			θ = (rc_l + rc_r) / 2 * (dt / Dm)

			ustar[i] = ustar_1[i] + 1/2 * (1 - θ) * (r_u_p*(umat[i] - ustar_1[i]) - r_u_m*(ustar_1[i] - umat[i-1]))
			pstar[i] = pstar_1[i] + 1/2 * (1 - θ) * (r_p_p*(pmat[i] - pstar_1[i]) - r_p_m*(pstar_1[i] - pmat[i-1]))
		end
	else
		println("The choice of the scheme for the acoustic solver is not recognized: ", scheme)
		exit()
	end

	return
end

#
# Test initialisation
# 

function init_test(params, x, rho, pmat, umat, emat, Emat, cmat, gmat)
	(; test, nghost, nbcell, ideb, ifin) = params

    if test == :Sod
        if params.maxtime == 0
            params.maxtime = 0.20
        end
    
        if params.cfl == 0
            params.cfl = 0.95
        end

		gamma::params.data_type = 1.4
    
        @threaded for i in ideb:ifin+1
            x[i] = (i-1-nghost) / nbcell
    
            if x[i] < 0.5
                rho[i] = 1.
                pmat[i] = 1.
                umat[i] = 0.
            else
                rho[i] = 0.125
                pmat[i] = 0.1
                umat[i] = 0.
            end
    
            emat[i] = Emat[i] = pmat[i]/((gamma-1.)*rho[i])
            cmat[i] = sqrt(gamma*pmat[i]/rho[i])
            gmat[i] = 0.5*(1.0+gamma)
        end
    elseif test == :Bizarrium
        if params.maxtime == 0
            params.maxtime = 80e-6
        end
    
        if params.cfl == 0
            params.cfl = 0.6
        end
    
        @threaded for i in ideb:ifin+1
            x[i] = (i-1-nghost) / nbcell
    
            if x[i] < 0.5
                rho[i] = 1.42857142857e+4
                umat[i] = 0.
                emat[i] = Emat[i] = 4.48657821135e+6
            else
                rho[i] =  10000.
                umat[i] = 250.
                emat[i] = 0.
                Emat[i] = emat[i] + 0.5*umat[i]^2
            end
        end
    
        BizarriumEOS!(params, pmat, cmat, gmat, rho, emat)
    end
	
	return
end

#
# Boundary conditions
#

function boundaryConditions!(params, rho, umat, pmat, cmat, gmat)
	(; test, ideb, ifin) = params

	if params.use_gpu
		gpu_boundary_conditions!(test, ideb, ifin, rho, umat, pmat, cmat, gmat)
		return
	end

	rho[ideb-1] = rho[ideb]; umat[ideb-1] = -umat[ideb]; pmat[ideb-1] = pmat[ideb]; cmat[ideb-1] = cmat[ideb]; gmat[ideb-1]=gmat[ideb]
	rho[ifin+1] = rho[ifin]; umat[ifin+1] = -umat[ifin]; pmat[ifin+1] = pmat[ifin]; cmat[ifin+1] = cmat[ifin]; gmat[ifin+1]=gmat[ifin]

	if test == :Bizarrium
		rho[ifin+1] = rho[ifin]
		umat[ifin+1] = umat[ifin]
		pmat[ifin+1] = pmat[ifin]
		cmat[ifin+1] = cmat[ifin]
		gmat[ifin+1] = gmat[ifin]
	end
end

#
# Time step computation
#

function dtCFL(params, dta, x, cmat, umat)::typeof(dta)
	(; cfl, Dt, ideb, ifin) = params

	dt::typeof(Dt) = Inf

	if params.cst_dt
		dt = Dt
	elseif params.euler_projection
		if params.use_gpu
			if use_ROCM
				dt = gpu_dtCFL_reduction!(params.euler_projection, ideb, ifin, x, cmat, umat)
			else
				dt = reduce(min, @views ((x[ideb+1:ifin+1] .- x[ideb:ifin]) ./ max.(abs.(umat[ideb:ifin] .+ cmat[ideb:ifin]), abs.(umat[ideb:ifin] .- cmat[ideb:ifin]))))
			end
		else
			@batch threadlocal=typemax(typeof(dta)) for i in ideb:ifin
				threadlocal = min(threadlocal, (x[i+1]-x[i])/max(abs(umat[i] + cmat[i]), abs(umat[i] - cmat[i])))
			end
			dt = minimum(threadlocal)
		end
	else
		if params.use_gpu
			if use_ROCM
				dt = gpu_dtCFL_reduction!(params.euler_projection, ideb, ifin, x, cmat, umat)
			else
				dt = reduce(min, @views ((x[ideb+1:ifin+1] .- x[ideb:ifin]) ./ cmat[ideb:ifin]))
			end
		else
			@batch threadlocal=typemax(typeof(dta)) for i in ideb:ifin
				threadlocal = min(threadlocal, (x[i+1]-x[i])/cmat[i])
			end
			dt = minimum(threadlocal)
		end
	end

	if dta == 0  # First cycle
		if Dt != 0
			return Dt
		else
			return cfl*dt
		end
	else
		return min(cfl*dt, 1.05*dta)   # CFL condition and maximum increase per cycle of the time step
	end
end

#
# Numerical fluxes computation
#

function numericalFluxes!(params, ustar, pstar, rho, umat, pmat, cmat, gmat, ustar_1, pstar_1, dt, x)
    if params.riemann == :acoustic    # 2-state acoustic solver (Godunov)
		if params.scheme == :Godunov
			acoustic!(params, ustar, pstar, rho, umat, pmat, cmat, dt, x)
		else
			acoustic_GAD!(params, ustar, pstar, rho, umat, pmat, cmat, ustar_1, pstar_1, dt, x)
		end
    else
        println("The choice of Riemann solver is not recognized: ", params.riemann)
		exit()
    end

    return
end

# 
# Cell update
# 


function first_order_euler_remap!(params, dt, X, rho, umat, Emat, ustar, tmp_rho, tmp_urho, tmp_Erho)
	(; ideb, ifin) = params

	if params.use_gpu
		gpu_first_order_euler_remap!(ideb, ifin, dt, X, ustar, rho, umat, Emat, tmp_rho, tmp_urho, tmp_Erho)
		return
	end

	# Projection of the conservative variables
	@simd_threaded_loop for i in ideb:ifin+1
		dx = X[i+1] - X[i]
		L₁ =  max(0, ustar[i]) * dt
		L₃ = -min(0, ustar[i+1]) * dt
		L₂ = dx - L₁ - L₃
		
		tmp_rho[i]  = (L₁ * rho[i-1] 
					 + L₂ * rho[i] 
					 + L₃ * rho[i+1]) / dx
		tmp_urho[i] = (L₁ * rho[i-1] * umat[i-1] 
					 + L₂ * rho[i]   * umat[i] 
					 + L₃ * rho[i+1] * umat[i+1]) / dx
		tmp_Erho[i] = (L₁ * rho[i-1] * Emat[i-1] 
					 + L₂ * rho[i]   * Emat[i] 
					 + L₃ * rho[i+1] * Emat[i+1]) / dx
	end

	# (ρ, ρu, ρE) -> (ρ, u, E)
	@simd_threaded_loop for i in ideb:ifin+1
		rho[i]  = tmp_rho[i]
		umat[i] = tmp_urho[i] / tmp_rho[i]
		Emat[i] = tmp_Erho[i] / tmp_rho[i]
	end
end


function cellUpdate!(params, dt, x, X, ustar, pstar, rho, umat, emat, Emat, tmp_rho, tmp_urho, tmp_Erho)
	(; ideb, ifin) = params

	if params.use_gpu
		gpu_cell_update!(params.euler_projection, ideb, ifin, dt, x, X, ustar, pstar, rho, umat, emat, Emat, tmp_rho, tmp_urho, tmp_Erho)
		return
	end

 	@simd_threaded_loop for i in ideb:ifin+1
		X[i] = x[i] + dt*ustar[i]
	end

	@simd_threaded_loop for i in ideb:ifin
		dm = rho[i]*(x[i+1]-x[i])
		rho[i] = dm/(X[i+1]-X[i])
		umat[i] = umat[i] + dt/dm*(pstar[i]-pstar[i+1])
		Emat[i] = Emat[i] + dt/dm*(pstar[i]*ustar[i]-pstar[i+1]*ustar[i+1])
		emat[i] = Emat[i] - 0.5*umat[i]^2
	end

	if !params.euler_projection
		@simd_threaded_loop for i in ideb:ifin+1
			x[i] = X[i]
		end
	end
end


function update_EOS!(params, pmat, cmat, gmat, rho, emat)
	(; test) = params

	if params.use_gpu
		gpu_update_EOS!(test, params.ideb, params.ifin, rho, emat, pmat, cmat, gmat)
		return
	end

	if test == :Sod || test == :Leblanc || test == :Woodward
		gamma::params.data_type = 0.0

		if test == :Sod || test == :Woodward
			gamma = 1.4
		elseif test == :Leblanc
			gamma = 5/3
		end

		perfectGasEOS!(params, pmat, cmat, gmat, rho, emat, gamma)
	elseif test == :Bizarrium
		BizarriumEOS!(params, pmat, cmat, gmat, rho, emat)
	end
end

# 
# Main time loop
# 

function time_loop(params, x, X, rho, umat, pmat, cmat, gmat, emat, Emat, ustar, pstar, ustar_1, pstar_1, tmp_rho, tmp_urho, tmp_Erho)
	(; maxtime, maxcycle, nbcell, silent) = params
    cycle = 0
	t::params.data_type   = 0.
	dta::params.data_type = 0.
	dt::params.data_type  = 0.

	t1 = time_ns()
	t_warmup = t1

	while t < maxtime && cycle < maxcycle
	    @time_pos params "boundaryConditions" boundaryConditions!(params, rho, umat, pmat, cmat, gmat)
		
		@time_pos params "dtCFL" dt = dtCFL(params, dta, x, cmat, umat)

		@time_pos params "numericalFluxes!" numericalFluxes!(params, ustar, pstar, rho, umat, pmat, cmat, gmat, ustar_1, pstar_1, dt, x)

		@time_pos params "cellUpdate!" cellUpdate!(params, dt, x, X, ustar, pstar, rho, umat, emat, Emat, tmp_rho, tmp_urho, tmp_Erho)

		if params.euler_projection
			@time_pos params "first_order_euler_remap!" first_order_euler_remap!(params, dt, X, rho, umat, Emat, ustar, tmp_rho, tmp_urho, tmp_Erho)
		end

		@time_pos params "update_EOS!" update_EOS!(params, pmat, cmat, gmat, rho, emat)

		dta = dt
		cycle += 1
		t += dt

		if silent <= 1
			println("Cycle = ", cycle, ", dt = ", dt, ", t = ", t)
	    end

		if cycle == 5
			t_warmup = time_ns()
		end
	end

	t2 = time_ns()
	
	grind_time = (t2 - t_warmup) / ((cycle - 5)*nbcell)

	if silent < 3
		println(" ")
		println("Time:       ", round((t2 - t1) / 1e9, digits=5), " sec")
		println("Warmup:     ", round(t_warmup / 1e9, digits=5), " sec")
		println("Grind time: ", round(grind_time / 1e3, digits=5), " µs/cell/cycle")
		println("Cells/sec:  ", round(1 / grind_time * 1e3, digits=5), " Mega cells/sec")
		println("Cycles: ", cycle)
		println(" ")
	end

	return params.data_type(grind_time)
end

#
# Output 
#

function write_result(x, rho, umat, pmat, emat, cmat, gmat, ustar, pstar, ideb, ifin, silent)
    f = open("output", "w")

    for i in ideb:ifin
        print(f, 0.5*(x[i]+x[i+1]), ", ", rho[i], ", ", umat[i], ", ", pmat[i], ", ", emat[i], ", ", cmat[i], ", ", gmat[i], ", ", ustar[i], ", ", pstar[i], "\n")
    end
    
    close(f)

	if silent < 2
		println("Output file closed")
	end
end

# 
# Main function
# 

function armon(params::ArmonParameters)
	(; data_type, nghost, nbcell, silent) = params

	if params.measure_time
		empty!(time_contrib)
	end

	if silent < 3
		print_parameters(params)
		if params.use_gpu
			println(" - gpu block size: ", block_size)
		end
	end
	
    x = zeros(data_type, nbcell+2*nghost)
    X = zeros(data_type, nbcell+2*nghost)
    rho = zeros(data_type, nbcell+2*nghost)
    umat = zeros(data_type, nbcell+2*nghost)
    emat = zeros(data_type, nbcell+2*nghost)
    Emat = zeros(data_type, nbcell+2*nghost)
    pmat = zeros(data_type, nbcell+2*nghost)
    cmat = zeros(data_type, nbcell+2*nghost)
    gmat = zeros(data_type, nbcell+2*nghost)
    ustar = zeros(data_type, nbcell+2*nghost)
    pstar = zeros(data_type, nbcell+2*nghost)
	ustar_1 = zeros(data_type, nbcell+2*nghost)
    pstar_1 = zeros(data_type, nbcell+2*nghost)
	tmp_rho  = zeros(data_type, nbcell+2*nghost)
	tmp_urho = zeros(data_type, nbcell+2*nghost)
	tmp_Erho = zeros(data_type, nbcell+2*nghost)

    init_time = @elapsed init_test(params, x, rho, pmat, umat, emat, Emat, cmat, gmat)

 	params.silent <= 2 && @printf("Init time: %.3g sec\n", init_time)

    if params.silent <= -1
        for i in params.ideb:params.ifin
            println("Init, ", 0.5*(x[i]+x[i+1]), ", ", rho[i], ", ", umat[i], ", ", pmat[i], ", ", emat[i], ", ", cmat[i], ", ", gmat[i], "\n")
        end
    end

	if params.use_gpu
		alloc_copy_GPU = use_ROCM ? ROCArray : CuArray

		copy_time = @elapsed begin
			d_x = alloc_copy_GPU(x)
			d_X = alloc_copy_GPU(X)
			d_rho = alloc_copy_GPU(rho)
			d_umat = alloc_copy_GPU(umat)
			d_emat = alloc_copy_GPU(emat)
			d_Emat = alloc_copy_GPU(Emat)
			d_pmat = alloc_copy_GPU(pmat)
			d_cmat = alloc_copy_GPU(cmat)
			d_gmat = alloc_copy_GPU(gmat)
			d_ustar = alloc_copy_GPU(ustar)
			d_pstar = alloc_copy_GPU(pstar)
			d_ustar_1 = alloc_copy_GPU(ustar_1)
			d_pstar_1 = alloc_copy_GPU(pstar_1)
			d_tmp_rho = alloc_copy_GPU(tmp_rho)
			d_tmp_urho = alloc_copy_GPU(tmp_urho)
			d_tmp_Erho = alloc_copy_GPU(tmp_Erho)
		end

		params.silent <= 2 && @printf("Time for copy to device: %.3g sec\n", copy_time)

		if params.silent <= 3
			@time cells_per_sec = time_loop(params, d_x, d_X, d_rho, d_umat, d_pmat, d_cmat, d_gmat, d_emat, d_Emat, d_ustar, d_pstar, d_ustar_1, d_pstar_1, d_tmp_rho, d_tmp_urho, d_tmp_Erho)
		else
			cells_per_sec = time_loop(params, d_x, d_X, d_rho, d_umat, d_pmat, d_cmat, d_gmat, d_emat, d_Emat, d_ustar, d_pstar, d_ustar_1, d_pstar_1, d_tmp_rho, d_tmp_urho, d_tmp_Erho)
		end

		copyto!(x, d_x)
		copyto!(X, d_X)
		copyto!(rho, d_rho)
		copyto!(umat, d_umat)
		copyto!(emat, d_emat)
		copyto!(Emat, d_Emat)
		copyto!(pmat, d_pmat)
		copyto!(cmat, d_cmat)
		copyto!(gmat, d_gmat)
		copyto!(ustar, d_ustar)
		copyto!(pstar, d_pstar)
		copyto!(ustar_1, d_ustar_1)
		copyto!(pstar_1, d_pstar_1)
	else
		if params.silent <= 3
			@time cells_per_sec = time_loop(params, x, X, rho, umat, pmat, cmat, gmat, emat, Emat, ustar, pstar, ustar_1, pstar_1, tmp_rho, tmp_urho, tmp_Erho)
		else
			cells_per_sec = time_loop(params, x, X, rho, umat, pmat, cmat, gmat, emat, Emat, ustar, pstar, ustar_1, pstar_1, tmp_rho, tmp_urho, tmp_Erho)
		end
	end

	if params.write_output
		write_result(x, rho, umat, pmat, emat, cmat, gmat, ustar, pstar, params.ideb, params.ifin, silent)
	end

	if params.measure_time && silent < 3
		total_time = mapreduce(x->x[2], +, collect(time_contrib))
		println("\nTotal time of each step:")
		for (label, time_) in sort(collect(time_contrib))
			@printf(" - %-25s %10.5f ms (%5.2f%%)\n", label, time_ / 1e6, time_ / total_time * 100)
		end
	end

	return cells_per_sec
end


end
