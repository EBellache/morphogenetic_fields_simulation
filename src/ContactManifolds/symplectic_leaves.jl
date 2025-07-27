"""
    SymplecticLeaf{N}

Represents a symplectic leaf in the contact manifold where total charges 
in each compartment are fixed.
"""
struct SymplecticLeaf{N}
    fixed_charges::SVector{N, Float64}
    dimension::Int
    symplectic_form::Function
    
    function SymplecticLeaf(charges::AbstractVector)
        N = length(charges)
        @assert abs(sum(charges)) < 1e-10  # Electroneutrality
        
        fixed_charges = SVector{N}(charges)
        dimension = 2N  # Phase space dimension on leaf
        
        # Define symplectic form on the leaf
        function ω(x, v, w)
            # x = [ρ₁, ..., ρₙ, φ₁, ..., φₙ] (restricted to leaf)
            # Standard symplectic form: ω = Σᵢ dρᵢ ∧ dφᵢ
            result = 0.0
            N_half = length(x) ÷ 2
            for i in 1:N_half
                result += v[i] * w[N_half + i] - v[N_half + i] * w[i]
            end
            return result
        end
        
        new{N}(fixed_charges, dimension, ω)
    end
end

"""
    project_to_leaf(distribution::ChargeDistribution, leaf::SymplecticLeaf)

Projects a charge distribution onto a specific symplectic leaf.
"""
function project_to_leaf(dist::ChargeDistribution{N}, leaf::SymplecticLeaf{N}) where N
    # On a leaf, charges are fixed, only phases can vary
    projected_charges = leaf.fixed_charges
    
    # Adjust phases to maintain consistency
    phases = dist.phases
    
    return ChargeDistribution(projected_charges, phases, dist.chemical_potentials, dist.time)
end

"""
    evolve_on_leaf(leaf::SymplecticLeaf, initial_state, hamiltonian, tspan)

Evolves dynamics on a single symplectic leaf (Hamiltonian flow).
"""
function evolve_on_leaf(leaf::SymplecticLeaf{N}, 
                       initial_dist::ChargeDistribution{N},
                       hamiltonian::Function, 
                       tspan::Tuple{Float64, Float64}) where N
    
    # Extract phase space coordinates (only phases vary on leaf)
    u0 = Vector(initial_dist.phases)
    
    # Hamiltonian equations on the leaf
    function hamiltonian_flow!(du, u, p, t)
        # u represents phases φᵢ
        # Compute derivatives using automatic differentiation
        H(φ) = hamiltonian(leaf.fixed_charges, φ, t)
        
        # dφᵢ/dt = ∂H/∂ρᵢ (but ρᵢ are fixed on leaf)
        # For now, simple dynamics
        du .= ForwardDiff.gradient(H, u)
    end
    
    # Solve ODE
    prob = ODEProblem(hamiltonian_flow!, u0, tspan)
    sol = solve(prob, Tsit5(), saveat=0.1)
    
    # Convert solution back to ChargeDistribution objects
    trajectory = [ChargeDistribution(
        leaf.fixed_charges, 
        sol.u[i], 
        initial_dist.chemical_potentials,
        sol.t[i]
    ) for i in 1:length(sol.t)]
    
    return trajectory
end

"""
    detect_leaf_transition(dist::ChargeDistribution, threshold::Float64)

Detects when the system is ready to transition between leaves based on
chemical potential gradients.
"""
function detect_leaf_transition(dist::ChargeDistribution{N}, threshold::Float64=0.1) where N
    # Compute driving forces for charge transfer
    forces = Float64[]
    
    for i in 1:N
        for j in i+1:N
            Δμ = dist.chemical_potentials[i] - dist.chemical_potentials[j]
            push!(forces, abs(Δμ))
        end
    end
    
    max_force = maximum(forces)
    return max_force > threshold, max_force
end