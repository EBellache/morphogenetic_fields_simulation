module ContactManifolds

using LinearAlgebra
using StaticArrays
using DifferentialEquations
using ForwardDiff

# Import from parent module
using ..Utils

export ContactManifold, SymplecticLeaf, ChargeDistribution, ReebFlow
export compute_contact_form, compute_reeb_field, evolve_on_leaf
export transition_between_leaves, check_electroneutrality
export ChargeEvolutionProblem  # Add this export


"""
    ContactManifold{N}

Represents a (2N+1)-dimensional contact manifold arising from electroneutrality constraints.
Contains the contact form α and the foliation by symplectic leaves.
"""
struct ContactManifold{N}
    dimension::Int
    contact_form::Function  # α(x) returns a covector
    hamiltonian::Function   # H(x) for dynamics
    constraints::Vector{Function}  # Additional constraints
    
    function ContactManifold{N}(hamiltonian::Function; 
                                constraints::Vector{Function}=Function[]) where N
        dimension = 2N + 1
        new{N}(dimension, x -> compute_contact_form(x, N), hamiltonian, constraints)
    end
end

"""
    compute_contact_form(state, N)

Computes the contact 1-form α = Σᵢ ρᵢ dφᵢ - Σⱼ μⱼ dQⱼ + dt
where state = [ρ₁, ..., ρₙ, φ₁, ..., φₙ, t]
"""
function compute_contact_form(state::AbstractVector, N::Int)
    @assert length(state) == 2N + 1
    
    ρ = @view state[1:N]      # Charge densities
    φ = @view state[N+1:2N]   # Phases
    t = state[2N+1]           # Time
    
    # Contact form coefficients
    α = zeros(2N + 1)
    α[1:N] .= φ              # dρᵢ coefficients
    α[N+1:2N] .= ρ           # dφᵢ coefficients  
    α[2N+1] = 1.0           # dt coefficient
    
    return α
end

"""
    ChargeDistribution{N}

Represents charge distribution across N compartments with phases.
Maintains electroneutrality: Σᵢ Qᵢ = 0
"""
struct ChargeDistribution{N}
    charges::SVector{N, Float64}      # Qᵢ in each compartment
    phases::SVector{N, Float64}       # φᵢ for each compartment
    chemical_potentials::SVector{N, Float64}  # μᵢ
    time::Float64
    
    function ChargeDistribution(charges::AbstractVector, phases::AbstractVector, 
                               chemical_potentials::AbstractVector, time::Real=0.0)
        N = length(charges)
        @assert length(phases) == N && length(chemical_potentials) == N
        @assert abs(sum(charges)) < 1e-10  # Electroneutrality check
        
        new{N}(SVector{N}(charges), SVector{N}(phases), 
               SVector{N}(chemical_potentials), Float64(time))
    end
end

# Include the subfiles after type definitions
include("symplectic_leaves.jl")
include("reeb_flow.jl")
include("charge_dynamics.jl")

end # module ContactManifolds