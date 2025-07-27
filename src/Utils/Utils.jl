module Utils

using LinearAlgebra
using StaticArrays
using ForwardDiff
using Statistics
using SpecialFunctions

export adaptive_step_size, runge_kutta_4, compute_jacobian
export estimate_lyapunov_exponents, compute_correlation_dimension
export tropical_operations, TropicalNumber, ⊕, ⊗
export continued_fraction_expansion, convergents
export compute_geodesic_distance, parallel_transport
export validate_electroneutrality, compute_energy_functional

include("numerical_methods.jl")

end # module Utils