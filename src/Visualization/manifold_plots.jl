"""
Advanced visualization functions for morphogenetic fields and contact manifolds.
Implements sophisticated 3D visualizations with proper mathematical representations.
"""

# Color schemes for different visualizations
const LEAF_COLORS = ColorScheme([colorant"#E8F4F8", colorant"#B8E0E8", 
                                colorant"#88CCD8", colorant"#58B8C8", 
                                colorant"#28A4B8"])
const FLOW_COLORS = ColorScheme([colorant"#FFE6E6", colorant"#FFCCCC", 
                                colorant"#FFB3B3", colorant"#FF9999", 
                                colorant"#FF8080"])
const ENERGY_COLORS = :viridis

"""
    plot_phase_portrait(manifold, charge_range, phase_range; kwargs...)

Creates a phase portrait showing the flow on a 2D slice of the contact manifold.
Useful for understanding local dynamics and fixed points.
"""
function plot_phase_portrait(manifold::ContactManifold{N}, 
                           charge_range::Tuple{Float64,Float64},
                           phase_range::Tuple{Float64,Float64};
                           resolution::Int=30,
                           slice_params::Dict=Dict(),
                           figure_size=(800, 800)) where N
    
    fig = Figure(resolution=figure_size)
    ax = Axis(fig[1,1], 
              xlabel="Charge (compartment 1)",
              ylabel="Phase (compartment 1)",
              title="Phase Portrait on Contact Manifold Slice")
    
    # Create grid
    charges = range(charge_range..., length=resolution)
    phases = range(phase_range..., length=resolution)
    
    # Compute vector field
    u_field = zeros(resolution, resolution)
    v_field = zeros(resolution, resolution)
    
    for (i, q) in enumerate(charges)
        for (j, φ) in enumerate(phases)
            # Construct full state (with other compartments from slice_params)
            state = zeros(2N+1)
            state[1] = q
            state[N+1] = φ
            
            # Fill in other values from slice_params
            for (key, val) in slice_params
                if key isa Int && key <= 2N+1
                    state[key] = val
                end
            end
            
            # Ensure electroneutrality
            if N > 1
                state[N] = -sum(state[1:N-1])
            end
            
            # Compute Reeb field at this point
            R = compute_reeb_field(manifold, state)
            
            u_field[i,j] = R[1]      # dQ/dt
            v_field[i,j] = R[N+1]    # dφ/dt
        end
    end
    
    # Normalize for visualization
    speed = sqrt.(u_field.^2 + v_field.^2)
    
    # Plot streamlines
    streamplot!(ax, charges, phases, u_field, v_field,
                colormap=:plasma,
                arrow_size=10,
                linewidth=2,
                density=1.5)
    
    # Find and mark fixed points
    fixed_points = find_fixed_points(u_field, v_field, charges, phases)
    if !isempty(fixed_points)
        scatter!(ax, [fp[1] for fp in fixed_points], [fp[2] for fp in fixed_points],
                color=:red, markersize=15, marker=:star5)
    end
    
    # Add nullclines
    contour!(ax, charges, phases, u_field, levels=[0], color=:blue, linewidth=2, label="dQ/dt = 0")
    contour!(ax, charges, phases, v_field, levels=[0], color=:green, linewidth=2, label="dφ/dt = 0")
    
    axislegend(ax, position=:rt)
    
    return fig
end

"""
    plot_symplectic_leaves(manifold, leaves; kwargs...)

Visualizes multiple symplectic leaves and their relationships in 3D space.
Shows how the contact manifold foliates into symplectic submanifolds.
"""
function plot_symplectic_leaves(manifold::ContactManifold{N}, 
                              leaves::Vector{SymplecticLeaf{N}};
                              figure_size=(1000, 800),
                              show_connections=true) where N
    
    fig = Figure(resolution=figure_size)
    ax = Axis3(fig[1,1], 
               xlabel="Σ|Qᵢ|",
               ylabel="Phase variance",
               zlabel="Energy",
               title="Symplectic Foliation Structure",
               azimuth=0.3π)
    
    # Plot each leaf as a surface
    for (idx, leaf) in enumerate(leaves)
        # Parametrize the leaf surface
        u_range = range(0, 2π, length=40)
        v_range = range(0, 1, length=20)
        
        X = zeros(length(u_range), length(v_range))
        Y = zeros(length(u_range), length(v_range))
        Z = zeros(length(u_range), length(v_range))
        
        for (i, u) in enumerate(u_range)
            for (j, v) in enumerate(v_range)
                # Create a state on this leaf
                phases = [u + 0.5*sin(2π*i/N) for i in 1:N]
                dist = ChargeDistribution(leaf.fixed_charges, phases, zeros(N), 0.0)
                
                X[i,j] = sum(abs.(leaf.fixed_charges))
                Y[i,j] = std(phases) * (1 + 0.3*v)
                Z[i,j] = compute_leaf_energy(manifold, dist) * (1 + 0.1*v*cos(3u))
            end
        end
        
        # Color based on total charge magnitude
        color_val = sum(abs.(leaf.fixed_charges))
        surf = surface!(ax, X, Y, Z, 
                       color=fill(color_val, size(X)),
                       colormap=LEAF_COLORS,
                       alpha=0.7,
                       transparency=true)
    end
    
    # Show connections between leaves if requested
    if show_connections && length(leaves) > 1
        for i in 1:length(leaves)
            for j in i+1:length(leaves)
                if can_transition(leaves[i], leaves[j])
                    # Draw connection line
                    p1 = leaf_center(leaves[i])
                    p2 = leaf_center(leaves[j])
                    lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]], [p1[3], p2[3]],
                          color=:red, linewidth=2, alpha=0.5)
                end
            end
        end
    end
    
    # Add colorbar
    Colorbar(fig[1,2], limits=(0, N), label="Total |Q|", colormap=LEAF_COLORS)
    
    return fig
end

"""
    plot_reeb_flow(manifold, initial_points, time_span; kwargs...)

Visualizes the Reeb flow trajectories from multiple initial conditions.
Shows how the flow preserves the contact structure while redistributing charge.
"""
function plot_reeb_flow(manifold::ContactManifold{N}, 
                       initial_points::Vector{<:AbstractVector},
                       time_span::Tuple{Float64,Float64};
                       dt::Float64=0.01,
                       figure_size=(1200, 800)) where N
    
    fig = Figure(resolution=figure_size)
    
    # 3D trajectory plot
    ax3d = Axis3(fig[1:2, 1:2], 
                 xlabel="Q₁",
                 ylabel="Q₂", 
                 zlabel="φ₁",
                 title="Reeb Flow Trajectories",
                 azimuth=0.4π,
                 elevation=0.2π)
    
    # Store all trajectories
    all_trajectories = []
    
    for (idx, point) in enumerate(initial_points)
        # Integrate Reeb flow
        function reeb_dynamics!(du, u, p, t)
            R = compute_reeb_field(manifold, u)
            du .= R
        end
        
        prob = ODEProblem(reeb_dynamics!, point, time_span)
        sol = solve(prob, RK4(), dt=dt)
        
        # Extract coordinates for plotting
        if N >= 2
            q1 = [u[1] for u in sol.u]
            q2 = [u[2] for u in sol.u]
            φ1 = [u[N+1] for u in sol.u]
            
            # Color by time
            colors = sol.t
            
            lines!(ax3d, q1, q2, φ1, 
                   color=colors, 
                   colormap=:viridis,
                   linewidth=3,
                   alpha=0.8)
            
            # Mark initial point
            scatter!(ax3d, [q1[1]], [q2[1]], [φ1[1]], 
                    color=:green, markersize=10)
            
            # Mark final point
            scatter!(ax3d, [q1[end]], [q2[end]], [φ1[end]], 
                    color=:red, markersize=10)
        end
        
        push!(all_trajectories, sol)
    end
    
    # Time series of electroneutrality violation
    ax_neutral = Axis(fig[1, 3],
                      xlabel="Time",
                      ylabel="|ΣQ|",
                      title="Electroneutrality Conservation",
                      yscale=log10)
    
    for sol in all_trajectories
        violations = [abs(sum(u[1:N])) for u in sol.u]
        lines!(ax_neutral, sol.t, violations .+ 1e-16, linewidth=2)
    end
    
    # Phase space projection
    ax_phase = Axis(fig[2, 3],
                    xlabel="Σ|Qᵢ|",
                    ylabel="Σ|φᵢ|",
                    title="Phase Space Projection")
    
    for sol in all_trajectories
        total_charge = [sum(abs.(u[1:N])) for u in sol.u]
        total_phase = [sum(abs.(u[N+1:2N])) for u in sol.u]
        lines!(ax_phase, total_charge, total_phase, linewidth=2)
    end
    
    return fig
end

"""
    create_morphogenetic_field_visualization(field_data, grid_points; kwargs...)

Creates a comprehensive visualization of a morphogenetic field showing:
- Director field D = ρe^{iφ} as arrows with color-coded phase
- Entropic skins as isosurfaces
- Energy landscape as background
"""
function create_morphogenetic_field_visualization(
    field_data::Array{ComplexF64,3},
    grid_points::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}};
    figure_size=(1400, 1000),
    show_entropic_skins=true,
    skin_threshold=0.1)
    
    fig = Figure(resolution=figure_size)
    
    # Main 3D visualization
    ax = Axis3(fig[1:2, 1:2],
               xlabel="X",
               ylabel="Y",
               zlabel="Z",
               title="Morphogenetic Field Structure",
               azimuth=0.3π,
               elevation=0.15π)
    
    x, y, z = grid_points
    
    # Compute field properties
    ρ = abs.(field_data)
    φ = angle.(field_data)
    
    # Create a slice at z_mid for vector field visualization
    z_mid = length(z) ÷ 2
    
    # Subsample for arrow visualization
    stride = 3
    x_sub = x[1:stride:end]
    y_sub = y[1:stride:end]
    
    # Director field arrows on the slice
    for (i, xi) in enumerate(x_sub)
        for (j, yj) in enumerate(y_sub)
            i_full = (i-1)*stride + 1
            j_full = (j-1)*stride + 1
            
            # Director field components
            D = field_data[i_full, j_full, z_mid]
            ρ_val = abs(D)
            φ_val = angle(D)
            
            # Arrow direction from phase gradient
            if i_full < length(x) && j_full < length(y)
                dφ_dx = angle(field_data[i_full+1, j_full, z_mid]) - φ_val
                dφ_dy = angle(field_data[i_full, j_full+1, z_mid]) - φ_val
                
                # Normalize
                norm_grad = sqrt(dφ_dx^2 + dφ_dy^2) + 1e-10
                arrow_dx = 0.5 * stride * dφ_dx / norm_grad * ρ_val
                arrow_dy = 0.5 * stride * dφ_dy / norm_grad * ρ_val
                
                # Color by phase
                color = HSV(φ_val * 180/π, 0.8, 0.8)
                
                # Draw arrow
                arrows!(ax, [xi], [yj], [z[z_mid]], 
                       [arrow_dx], [arrow_dy], [0.0],
                       color=[color],
                       linewidth=2,
                       arrowsize=Vec3f(0.1, 0.1, 0.15))
            end
        end
    end
    
    # Entropic skins as isosurfaces
    if show_entropic_skins
        # Compute energy density
        E = compute_field_energy_density(field_data, (x[2]-x[1], y[2]-y[1], z[2]-z[1]))
        
        # Find entropic skin locations (high gradient regions)
        ∇E = compute_gradient_magnitude(E)
        
        # Plot isosurface
        contour!(ax, x, y, z, ∇E,
                alpha=0.3,
                levels=[skin_threshold],
                color=:red,
                transparency=true)
    end
    
    # Side panel: Energy landscape
    ax_energy = Axis(fig[1, 3],
                     xlabel="X",
                     ylabel="Y",
                     title="Energy Density (z-slice)")
    
    E_slice = compute_field_energy_density(field_data[:, :, z_mid], (x[2]-x[1], y[2]-y[1], 1.0))
    hm = heatmap!(ax_energy, x, y, E_slice, colormap=ENERGY_COLORS)
    Colorbar(fig[1, 4], hm, label="Energy")
    
    # Side panel: Phase distribution
    ax_phase = Axis(fig[2, 3],
                    xlabel="Phase",
                    ylabel="Probability",
                    title="Phase Distribution")
    
    phase_hist = histogram!(ax_phase, vec(φ), bins=50, color=:blue, alpha=0.7)
    
    return fig
end

"""
    plot_tropical_computation(tropical_matrix, initial_state, n_steps; kwargs...)

Visualizes tropical matrix computations relevant to predictive coding.
Shows how max-plus operations create piecewise linear dynamics.
"""
function plot_tropical_computation(A::Matrix{TropicalNumber{Float64}}, 
                                 x0::Vector{TropicalNumber{Float64}},
                                 n_steps::Int;
                                 figure_size=(1200, 600))
    
    fig = Figure(resolution=figure_size)
    
    # Left: Tropical matrix structure
    ax_matrix = Axis(fig[1, 1],
                     xlabel="Column",
                     ylabel="Row",
                     title="Tropical Matrix A",
                     yreversed=true)
    
    # Extract values for heatmap
    n = size(A, 1)
    A_vals = [A[i,j].value for i in 1:n, j in 1:n]
    
    hm = heatmap!(ax_matrix, A_vals, colormap=:thermal)
    Colorbar(fig[1, 2], hm, label="Value")
    
    # Right: Evolution of state
    ax_evolution = Axis(fig[1, 3],
                       xlabel="Step",
                       ylabel="Component value",
                       title="Tropical Evolution: x⁽ᵏ⁾ = A ⊗ x⁽ᵏ⁻¹⁾")
    
    # Compute evolution
    states = Vector{TropicalNumber{Float64}}[]
    push!(states, x0)
    
    for k in 1:n_steps
        x_new = similar(x0)
        for i in 1:n
            # Tropical matrix-vector multiplication
            x_new[i] = TropicalNumber(-Inf)
            for j in 1:n
                x_new[i] = x_new[i] ⊕ (A[i,j] ⊗ states[end][j])
            end
        end
        push!(states, x_new)
    end
    
    # Plot evolution
    for i in 1:n
        values = [s[i].value for s in states]
        lines!(ax_evolution, 0:n_steps, values, 
               linewidth=2, label="x$i")
    end
    
    axislegend(ax_evolution, position=:lt)
    
    return fig
end

"""
    plot_hierarchical_structure(tower_levels, connections; kwargs...)

Visualizes the Hopf-Galois tower structure showing hierarchical organization.
Each level is shown with its characteristic scale and connections.
"""
function plot_hierarchical_structure(tower_levels::Vector{NamedTuple},
                                   connections::Matrix{Bool};
                                   figure_size=(1000, 1200))
    
    fig = Figure(resolution=figure_size)
    
    ax = Axis(fig[1,1],
              xlabel="Organizational Scale",
              ylabel="Level",
              title="Hopf-Galois Tower Structure",
              xscale=log10)
    
    n_levels = length(tower_levels)
    
    # Plot each level
    for (i, level) in enumerate(tower_levels)
        y = i
        
        # Characteristic scale on x-axis (log scale)
        x = level.scale
        
        # Size represents complexity
        markersize = 20 + 10 * level.complexity
        
        # Color represents operation type
        color = level.operation_type == :multiplication ? :blue :
                level.operation_type == :comultiplication ? :red :
                level.operation_type == :antipode ? :green : :gray
        
        scatter!(ax, [x], [y], 
                markersize=markersize,
                color=color,
                marker=:circle)
        
        # Label
        text!(ax, x, y + 0.3, text=level.name,
              align=(:center, :bottom))
    end
    
    # Plot connections
    for i in 1:n_levels
        for j in i+1:n_levels
            if connections[i,j]
                x1, y1 = tower_levels[i].scale, i
                x2, y2 = tower_levels[j].scale, j
                
                # Use log-space interpolation for x
                t = range(0, 1, length=20)
                x_interp = exp.(log(x1) .+ t .* (log(x2) - log(x1)))
                y_interp = y1 .+ t .* (y2 - y1)
                
                lines!(ax, x_interp, y_interp,
                      color=(:black, 0.3),
                      linewidth=2)
            end
        end
    end
    
    # Legend
    elem_1 = MarkerElement(color=:blue, marker=:circle, markersize=15)
    elem_2 = MarkerElement(color=:red, marker=:circle, markersize=15)
    elem_3 = MarkerElement(color=:green, marker=:circle, markersize=15)
    
    Legend(fig[2,1], [elem_1, elem_2, elem_3],
           ["Multiplication", "Comultiplication", "Antipode"],
           orientation=:horizontal,
           tellheight=true)
    
    return fig
end

# Helper functions

function find_fixed_points(u_field, v_field, x_range, y_range)
    fixed_points = Tuple{Float64,Float64}[]
    nx, ny = size(u_field)
    
    for i in 2:(nx-1)
        for j in 2:(ny-1)
            # Check for sign changes in both fields
            if (u_field[i,j] * u_field[i+1,j] < 0 || 
                u_field[i,j] * u_field[i,j+1] < 0) &&
               (v_field[i,j] * v_field[i+1,j] < 0 || 
                v_field[i,j] * v_field[i,j+1] < 0)
                
                # Refine using linear interpolation
                x_fp = x_range[i]
                y_fp = y_range[j]
                push!(fixed_points, (x_fp, y_fp))
            end
        end
    end
    
    return fixed_points
end

function compute_leaf_energy(manifold::ContactManifold, dist::ChargeDistribution)
    # Simplified energy calculation
    state = vcat(Vector(dist.charges), Vector(dist.phases), [dist.time])
    return manifold.hamiltonian(dist.charges, dist.phases, dist.time)
end

function can_transition(leaf1::SymplecticLeaf, leaf2::SymplecticLeaf)
    # Check if transition is allowed by charge conservation
    charge_diff = norm(leaf1.fixed_charges - leaf2.fixed_charges)
    return charge_diff < 0.5  # Threshold for allowed transitions
end

function leaf_center(leaf::SymplecticLeaf)
    # Representative point for visualization
    total_charge = sum(abs.(leaf.fixed_charges))
    avg_phase = π  # Default phase
    energy = total_charge^2 / 2  # Simplified energy
    
    return [total_charge, 0.0, energy]
end

function compute_field_energy_density(field::Array{ComplexF64}, spacing)
    E = similar(field, Float64)
    
    for I in CartesianIndices(field)
        # Local energy density (simplified)
        E[I] = abs2(field[I])
        
        # Add gradient contributions if not at boundary
        for dim in 1:ndims(field)
            if I[dim] > 1 && I[dim] < size(field, dim)
                i_plus = I + CartesianIndex(ntuple(d -> d == dim ? 1 : 0, ndims(field)))
                i_minus = I - CartesianIndex(ntuple(d -> d == dim ? 1 : 0, ndims(field)))
                
                grad = (field[i_plus] - field[i_minus]) / (2 * spacing[dim])
                E[I] += 0.5 * abs2(grad)
            end
        end
    end
    
    return E
end

function compute_gradient_magnitude(E::Array{Float64})
    ∇E = similar(E)
    
    for I in CartesianIndices(E)
        grad_mag = 0.0
        
        for dim in 1:ndims(E)
            if I[dim] > 1 && I[dim] < size(E, dim)
                i_plus = I + CartesianIndex(ntuple(d -> d == dim ? 1 : 0, ndims(E)))
                i_minus = I - CartesianIndex(ntuple(d -> d == dim ? 1 : 0, ndims(E)))
                
                grad_component = (E[i_plus] - E[i_minus]) / 2
                grad_mag += grad_component^2
            end
        end
        
        ∇E[I] = sqrt(grad_mag)
    end
    
    return ∇E
end