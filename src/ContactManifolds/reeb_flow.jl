"""
    compute_reeb_field(manifold::ContactManifold, state)

Computes the Reeb vector field R satisfying:
- ι_R dα = 0  (R is in kernel of dα)
- α(R) = 1    (normalization)
"""
function compute_reeb_field(manifold::ContactManifold{N}, state::AbstractVector) where N
    # Get contact form at current state
    α = manifold.contact_form(state)
    
    # Compute dα using finite differences
    h = 1e-8
    dα = zeros(2N+1, 2N+1)
    
    for i in 1:2N+1
        for j in 1:2N+1
            if i != j
                ei = zeros(2N+1); ei[i] = h
                ej = zeros(2N+1); ej[j] = h
                
                α_i = manifold.contact_form(state + ei)
                α_j = manifold.contact_form(state + ej)
                
                dα[i,j] = (α_j[i] - α[i])/h - (α_i[j] - α[j])/h
            end
        end
    end
    
    # Solve for Reeb field: find R such that dα(R,·) = 0 and α(R) = 1
    # This is a constrained linear system
    A = [dα; α']
    b = zeros(2N+2); b[end] = 1.0
    
    # Solve using least squares
    R = A \ b
    R = R[1:2N+1]
    
    # Normalize to ensure α(R) = 1
    R = R / dot(α, R)
    
    return R
end

"""
    ReebFlow

Represents the flow along the Reeb vector field, which preserves
electroneutrality while allowing charge redistribution.
"""
struct ReebFlow{N}
    manifold::ContactManifold{N}
    conductivity_matrix::Matrix{Float64}  # σᵢⱼ between compartments
    temperature::Float64
    
    function ReebFlow(manifold::ContactManifold{N}, σ::Matrix{Float64}, T::Float64=300.0) where N
        @assert size(σ) == (N, N)
        @assert issymmetric(σ)
        new{N}(manifold, σ, T)
    end
end

"""
    compute_charge_flux(flow::ReebFlow, dist::ChargeDistribution)

Computes charge flux between compartments driven by chemical potential differences.
"""
function compute_charge_flux(flow::ReebFlow{N}, dist::ChargeDistribution{N}) where N
    flux = zeros(N, N)
    
    for i in 1:N
        for j in i+1:N
            # Charge flux from i to j driven by chemical potential difference
            Δμ = dist.chemical_potentials[i] - dist.chemical_potentials[j]
            flux[i,j] = flow.conductivity_matrix[i,j] * Δμ / flow.temperature
            flux[j,i] = -flux[i,j]  # Conservation
        end
    end
    
    return flux
end

"""
    reeb_flow_dynamics!(du, u, p, t)

Dynamics along the Reeb flow, preserving electroneutrality.
"""
function reeb_flow_dynamics!(du, u, flow::ReebFlow{N}, t) where N
    # u = [Q₁, ..., Qₙ, φ₁, ..., φₙ]
    charges = @view u[1:N]
    phases = @view u[N+1:2N]
    
    # Compute chemical potentials (simplified model)
    μ = charges  # In reality, this would be more complex
    
    # Create current distribution
    dist = ChargeDistribution(charges, phases, μ, t)
    
    # Compute charge fluxes
    flux = compute_charge_flux(flow, dist)
    
    # Update charges (ensuring conservation)
    dQ = zeros(N)
    for i in 1:N
        dQ[i] = sum(flux[j,i] for j in 1:N)
    end
    
    # Update phases based on charge flow
    dφ = zeros(N)
    for i in 1:N
        # Phase changes coupled to charge flow
        dφ[i] = -μ[i]  # Simplified: phase precesses with chemical potential
    end
    
    du[1:N] .= dQ
    du[N+1:2N] .= dφ
end

"""
    transition_between_leaves(flow::ReebFlow, initial_dist, target_leaf, duration)

Simulates transition between symplectic leaves via Reeb flow (dissipative).
"""
function transition_between_leaves(flow::ReebFlow{N}, 
                                 initial_dist::ChargeDistribution{N},
                                 target_leaf::SymplecticLeaf{N},
                                 duration::Float64) where N
    
    # Initial state vector
    u0 = vcat(Vector(initial_dist.charges), Vector(initial_dist.phases))
    
    # Target charges
    target_charges = target_leaf.fixed_charges
    
    # Modified dynamics that drives towards target
    function transition_dynamics!(du, u, p, t)
        # Standard Reeb flow
        reeb_flow_dynamics!(du, u, flow, t)
        
        # Add driving term towards target
        current_charges = @view u[1:N]
        charge_error = target_charges - current_charges
        
        # Exponential relaxation towards target
        τ = duration / 3  # Time constant
        du[1:N] .+= charge_error / τ
    end
    
    # Solve ODE
    prob = ODEProblem(transition_dynamics!, u0, (0.0, duration))
    sol = solve(prob, Tsit5(), saveat=duration/100)
    
    # Convert to trajectory of distributions
    trajectory = ChargeDistribution{N}[]
    for i in 1:length(sol.t)
        charges = @view sol.u[i][1:N]
        phases = @view sol.u[i][N+1:2N]
        
        # Compute chemical potentials
        μ = charges  # Simplified
        
        push!(trajectory, ChargeDistribution(charges, phases, μ, sol.t[i]))
    end
    
    return trajectory
end