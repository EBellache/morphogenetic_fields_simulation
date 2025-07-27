# Tropical arithmetic operations for predictive coding simulations

"""
    TropicalNumber{T}

Represents a number in the tropical semiring (max-plus algebra).
"""
struct TropicalNumber{T<:Real}
    value::T
end

# Define tropical operations
⊕(a::TropicalNumber, b::TropicalNumber) = TropicalNumber(max(a.value, b.value))
⊗(a::TropicalNumber, b::TropicalNumber) = TropicalNumber(a.value + b.value)

"""
    adaptive_step_size(error_estimate, tolerance, current_step, order)

Computes adaptive step size for numerical integration based on error estimates.
Implements the standard step size control formula for embedded Runge-Kutta methods.
"""
function adaptive_step_size(error_estimate::Float64, tolerance::Float64, 
                          current_step::Float64, order::Int)
    safety_factor = 0.9
    min_factor = 0.2
    max_factor = 5.0
    
    if error_estimate < eps()
        return current_step * max_factor
    end
    
    factor = safety_factor * (tolerance / error_estimate)^(1.0 / (order + 1))
    factor = clamp(factor, min_factor, max_factor)
    
    return current_step * factor
end

"""
    runge_kutta_4(f, y0, t_span, dt)

Fourth-order Runge-Kutta integration for systems of ODEs.
Particularly useful for Hamiltonian systems on symplectic leaves.
"""
function runge_kutta_4(f::Function, y0::AbstractVector, t_span::Tuple{Float64,Float64}, dt::Float64)
    t0, tf = t_span
    t = t0:dt:tf
    n = length(t)
    m = length(y0)
    
    y = zeros(m, n)
    y[:, 1] = y0
    
    for i in 1:(n-1)
        k1 = dt * f(t[i], y[:, i])
        k2 = dt * f(t[i] + dt/2, y[:, i] + k1/2)
        k3 = dt * f(t[i] + dt/2, y[:, i] + k2/2)
        k4 = dt * f(t[i] + dt, y[:, i] + k3)
        
        y[:, i+1] = y[:, i] + (k1 + 2k2 + 2k3 + k4) / 6
    end
    
    return t, y
end

"""
    compute_jacobian(f, x)

Computes the Jacobian matrix of function f at point x using automatic differentiation.
Essential for stability analysis and bifurcation detection.
"""
function compute_jacobian(f::Function, x::AbstractVector)
    return ForwardDiff.jacobian(f, x)
end

"""
    estimate_lyapunov_exponents(trajectory, dt, embedding_dim, tau)

Estimates Lyapunov exponents from a trajectory using the Rosenstein algorithm.
Useful for characterizing chaotic dynamics in morphogenetic fields.
"""
function estimate_lyapunov_exponents(trajectory::Matrix{Float64}, dt::Float64, 
                                   embedding_dim::Int, tau::Int)
    n_points = size(trajectory, 2)
    n_vars = size(trajectory, 1)
    
    # Reconstruct phase space using time-delay embedding
    embedded_points = []
    for i in 1:(n_points - (embedding_dim - 1) * tau)
        point = Float64[]
        for j in 0:(embedding_dim - 1)
            append!(point, trajectory[:, i + j * tau])
        end
        push!(embedded_points, point)
    end
    
    # Find nearest neighbors and track divergence
    n_embedded = length(embedded_points)
    lyapunov_sum = 0.0
    count = 0
    
    for i in 1:(n_embedded - 1)
        # Find nearest neighbor
        min_dist = Inf
        nearest_idx = 0
        
        for j in (i+1):n_embedded
            if abs(j - i) > embedding_dim * tau  # Avoid temporally correlated points
                dist = norm(embedded_points[i] - embedded_points[j])
                if dist < min_dist && dist > eps()
                    min_dist = dist
                    nearest_idx = j
                end
            end
        end
        
        if nearest_idx > 0 && nearest_idx < n_embedded
            # Track divergence
            initial_sep = min_dist
            final_sep = norm(embedded_points[i+1] - embedded_points[nearest_idx+1])
            
            if initial_sep > eps() && final_sep > eps()
                lyapunov_sum += log(final_sep / initial_sep) / dt
                count += 1
            end
        end
    end
    
    return count > 0 ? lyapunov_sum / count : 0.0
end

"""
    compute_correlation_dimension(trajectory, r_min, r_max, n_r)

Computes the correlation dimension of an attractor from trajectory data.
Provides a measure of the fractal dimension of morphogenetic patterns.
"""
function compute_correlation_dimension(trajectory::Matrix{Float64}, 
                                     r_min::Float64, r_max::Float64, n_r::Int)
    n_points = size(trajectory, 2)
    
    # Compute pairwise distances
    distances = Float64[]
    for i in 1:n_points
        for j in (i+1):n_points
            push!(distances, norm(trajectory[:, i] - trajectory[:, j]))
        end
    end
    
    # Compute correlation integral for different radii
    radii = exp.(range(log(r_min), log(r_max), length=n_r))
    correlation_integral = zeros(n_r)
    
    for (idx, r) in enumerate(radii)
        correlation_integral[idx] = count(d -> d < r, distances) / (n_points * (n_points - 1) / 2)
    end
    
    # Estimate dimension from log-log slope
    log_r = log.(radii)
    log_c = log.(correlation_integral .+ eps())
    
    # Use central region for fitting
    start_idx = findfirst(c -> c > 0.01, correlation_integral)
    end_idx = findlast(c -> c < 0.9, correlation_integral)
    
    if start_idx !== nothing && end_idx !== nothing && end_idx > start_idx
        # Linear regression in log-log space
        X = [ones(end_idx - start_idx + 1) log_r[start_idx:end_idx]]
        y = log_c[start_idx:end_idx]
        
        coeffs = X \ y
        return coeffs[2]  # Slope is the correlation dimension
    else
        return NaN
    end
end

"""
    continued_fraction_expansion(x, max_terms)

Computes the continued fraction expansion of a real number.
Used for analyzing resonances and frequency locking in biological oscillators.
"""
function continued_fraction_expansion(x::Real, max_terms::Int=20)
    coefficients = Int[]
    remainder = x
    
    for i in 1:max_terms
        a = floor(Int, remainder)
        push!(coefficients, a)
        
        remainder = remainder - a
        if abs(remainder) < eps()
            break
        end
        
        remainder = 1.0 / remainder
    end
    
    return coefficients
end

"""
    convergents(cf_coefficients)

Computes the convergents (rational approximations) from continued fraction coefficients.
Returns numerators and denominators separately.
"""
function convergents(cf_coefficients::Vector{Int})
    n = length(cf_coefficients)
    numerators = zeros(Int, n)
    denominators = zeros(Int, n)
    
    # Initial values
    numerators[1] = cf_coefficients[1]
    denominators[1] = 1
    
    if n > 1
        numerators[2] = cf_coefficients[2] * cf_coefficients[1] + 1
        denominators[2] = cf_coefficients[2]
    end
    
    # Recurrence relation
    for i in 3:n
        numerators[i] = cf_coefficients[i] * numerators[i-1] + numerators[i-2]
        denominators[i] = cf_coefficients[i] * denominators[i-1] + denominators[i-2]
    end
    
    return numerators, denominators
end

"""
    compute_geodesic_distance(manifold, point1, point2, metric)

Computes geodesic distance between two points on a Riemannian manifold.
Uses numerical integration of the geodesic equation.
"""
function compute_geodesic_distance(metric::Function, point1::AbstractVector, 
                                 point2::AbstractVector, n_steps::Int=100)
    # Straight line initialization (to be refined)
    t = range(0, 1, length=n_steps)
    path = [point1 + s * (point2 - point1) for s in t]
    
    # Iterative refinement using gradient descent
    for iteration in 1:50
        # Compute path length
        length = 0.0
        for i in 1:(n_steps-1)
            tangent = path[i+1] - path[i]
            g = metric(path[i])
            length += sqrt(dot(tangent, g * tangent))
        end
        
        # Update interior points to minimize length
        for i in 2:(n_steps-1)
            # Compute gradient of length functional
            g_prev = metric(path[i-1])
            g_curr = metric(path[i])
            g_next = metric(path[i+1])
            
            grad = (g_curr * (path[i] - path[i-1]) / norm(path[i] - path[i-1]) -
                   g_curr * (path[i+1] - path[i]) / norm(path[i+1] - path[i]))
            
            # Gradient descent step
            path[i] -= 0.01 * grad / iteration
        end
    end
    
    # Compute final geodesic distance
    distance = 0.0
    for i in 1:(n_steps-1)
        tangent = path[i+1] - path[i]
        g = metric(path[i])
        distance += sqrt(dot(tangent, g * tangent))
    end
    
    return distance, path
end

"""
    parallel_transport(vector, path, connection)

Parallel transports a vector along a path using the given connection.
Essential for comparing vectors at different points on the manifold.
"""
function parallel_transport(vector::AbstractVector, path::Vector{<:AbstractVector}, 
                          connection::Function)
    transported = copy(vector)
    
    for i in 1:(length(path)-1)
        # Tangent to path
        tangent = path[i+1] - path[i]
        
        # Connection coefficients at current point
        Γ = connection(path[i])
        
        # Update vector using parallel transport equation
        for j in 1:length(transported)
            update = 0.0
            for k in 1:length(transported)
                for l in 1:length(tangent)
                    update += Γ[j, k, l] * transported[k] * tangent[l]
                end
            end
            transported[j] -= update
        end
    end
    
    return transported
end

"""
    validate_electroneutrality(charges, tolerance)

Validates that a charge distribution satisfies electroneutrality.
Returns whether the constraint is satisfied and the violation magnitude.
"""
function validate_electroneutrality(charges::AbstractVector, tolerance::Float64=1e-10)
    total_charge = sum(charges)
    is_neutral = abs(total_charge) < tolerance
    return is_neutral, abs(total_charge)
end

"""
    compute_energy_functional(director_field, params)

Computes the frustrated energy functional for a complex director field.
Includes chiral elastic energy, surface energy, and flow dissipation.
"""
function compute_energy_functional(director_field::Matrix{ComplexF64}, 
                                 params::NamedTuple)
    # Extract parameters
    κ_chiral = params.chiral_stiffness
    γ_surface = params.surface_tension
    σ_flow = params.flow_conductivity
    
    # Compute gradient using finite differences
    nx, ny = size(director_field)
    dx = params.dx
    dy = params.dy
    
    # Chiral elastic energy
    E_chiral = 0.0
    for i in 2:(nx-1)
        for j in 2:(ny-1)
            # Compute derivatives
            dD_dx = (director_field[i+1, j] - director_field[i-1, j]) / (2*dx)
            dD_dy = (director_field[i, j+1] - director_field[i, j-1]) / (2*dy)
            
            # Anti-holomorphic part (measures deviation from holomorphicity)
            dD_dz_bar = 0.5 * (dD_dx + im * dD_dy)
            
            E_chiral += κ_chiral * abs2(dD_dz_bar) * dx * dy
        end
    end
    
    # Surface energy contribution
    E_surface = 0.0
    for i in 1:nx
        for j in 1:ny
            if i == 1 || i == nx || j == 1 || j == ny
                E_surface += γ_surface * abs2(director_field[i, j]) * dx * dy
            end
        end
    end
    
    # Flow dissipation energy
    E_flow = 0.0
    for i in 2:(nx-1)
        for j in 2:(ny-1)
            # Velocity field from director evolution
            v_x = imag(conj(director_field[i, j]) * (director_field[i+1, j] - director_field[i-1, j]) / (2*dx))
            v_y = imag(conj(director_field[i, j]) * (director_field[i, j+1] - director_field[i, j-1]) / (2*dy))
            
            E_flow += σ_flow * (v_x^2 + v_y^2) * dx * dy
        end
    end
    
    return E_chiral + E_surface + E_flow
end