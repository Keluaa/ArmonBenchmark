
struct TopoEntity
    name::String
    is_node::Bool  # `false` if switch, `true` if node (leaf)
    level::Int  # Tree depth
    link_speed::Int  # ?
    connections::Vector{String}  # Connections to entities with a `level` less or equal
    backedges::Vector{String}  # Connections to entities with a higher level (non-empty only if `is_node == false`)
end


"""
    expand_range(str)

Expands a Slurm range (e.g. "abc[100-115,230,533-555],def333") into a list of entity names.

See [`node_list_to_range`](@ref) for the opposite operation.
"""
function expand_range(str)
    objects = String[]

    for m in eachmatch(r"(\w+)(?>\[([0-9,-]+)])?,?", str)
        obj_name = m[1]
        if isnothing(m[2])
            # Name without a range
            push!(objects, obj_name)
            continue
        end

        for range in split(m[2], ',')
            if occursin('-', range)
                r_start, r_end = split(range, '-')
            else
                r_start = r_end = range
            end
    
            r_start = parse(Int, r_start)
            r_end = parse(Int, r_end)
    
            for i in r_start:r_end
                push!(objects, "$obj_name$i")
            end
        end
    end

    return objects
end


"""
    node_list_to_range(node_list::Vector{<:AbstractString})

The opposite of [`expand_range`](@ref): from a list of nodes/switches, return a compact Slurm range.
"""
function node_list_to_range(node_list::Vector{<:AbstractString})
    clusters = Dict{String, Vector{Int}}()

    # Split nodes by cluster (the prefix before the numbers)
    for node in node_list
        m = match(r"^(\w*?)(\d*)$", node)
        isnothing(m) && error("Node '$node' does not respect the expected name format")
        cluster = m[1]
        if isempty(m[2])
            if haskey(clusters, cluster)
                error("Node '$node' does not end with digits yet appears more than once")
            end
            clusters[cluster] = Int[]
        else
            number = parse(Int, m[2])
            cluster_list = get!(clusters, cluster) do; Int[] end
            push!(cluster_list, number)
        end
    end

    ranges = String[]
    for (cluster_name, cluster) in clusters
        groups = Vector{UnitRange{Int}}()
        for num in cluster
            # Attempt to merge `num` with each group
            for (i, group) in enumerate(groups)
                if num == first(group) - 1
                    groups[i] = num:last(group)
                    @goto next_num
                elseif last(group) + 1 == num
                    groups[i] = first(group):num
                    @goto next_num
                end
            end

            # `num` did not fit in any groups
            push!(groups, num:num)

            @label next_num
        end

        # Merge stack => for each group in the stack find a neighbouring group and combine, repeat until no change
        change = true
        while change
            change = false
            for (i, group) in enumerate(groups)
                for (j_, o_group) in enumerate(groups[i+1:end])
                    j = j_ + i
                    if last(o_group) + 1 == first(group)
                        groups[i] = first(o_group):last(group)
                        deleteat!(groups, j)
                        change = true
                        @goto restart_merge
                    elseif last(group) == first(o_group) - 1
                        groups[i] = first(group):last(o_group)
                        deleteat!(groups, j)
                        change = true
                        @goto restart_merge
                    end
                end
            end
            @label restart_merge
        end

        # Turn the groups into a valid Slurm range strings
        cluster_ranges = String[]
        for group in groups
            if length(group) == 1
                push!(cluster_ranges, string(only(group)))
            else
                push!(cluster_ranges, string(first(group), '-', last(group)))
            end
        end

        # Compactify as needed
        if isempty(cluster_ranges)
            push!(ranges, cluster_name)
        elseif length(cluster_ranges) == 1 && !occursin('-', only(cluster_ranges))
            push!(ranges, cluster_name * only(cluster_ranges))
        else
            push!(ranges, cluster_name * '[' * join(cluster_ranges, ',') * ']')
        end
    end

    return join(ranges, ',')
end


function parse_topo_entity!(nodes, fields)
    switch_name = nothing
    level = nothing
    link_speed = nothing
    connected_nodes = String[]
    connected_switches = String[]

    for field in fields
        field_name, value = split(field, '=')

        if field_name == "SwitchName"
            switch_name = value
        elseif field_name == "Level"
            level = parse(Int, value)
        elseif field_name == "LinkSpeed"
            link_speed = parse(Int, value)
        elseif field_name == "Nodes"
            node_names = expand_range(value)
            for node_name in node_names
                push!(connected_nodes, node_name)
                if !haskey(nodes, node_name)
                    nodes[node_name] = TopoEntity(node_name, true, 0, 0, TopoEntity[], TopoEntity[])
                end
            end
        elseif field_name == "Switches"
            switch_names = expand_range(value)
            for switch_name in switch_names
                push!(connected_switches, switch_name)
            end
        else
            error("Unknown field in topology: $field_name")
        end
    end

    if level > 0
        # Switches of level > 0 are indirectly connected to nodes. We only keep direct connections.
        connected_nodes = String[]
    end

    append!(connected_nodes, connected_switches)

    if !isempty(connected_nodes)
        nodes[switch_name] = TopoEntity(switch_name, false, level, link_speed, connected_nodes, TopoEntity[])
    end
end


function fill_backedges!(topo::Dict{String, TopoEntity})
    for (name, entity) in topo
        entity.is_node && continue
        for connection in entity.connections
            connected_entity = topo[connection]
            if entity.level > connected_entity.level
                push!(connected_entity.backedges, name)
            end
        end
    end
    return topo
end


"""
    parse_job_topo()

Retrieves the current job's topology using the "SLURM_JOB_NODELIST" environment variable.
"""
function parse_job_topo()
    node_range = get(ENV, "SLURM_JOB_NODELIST", nothing)
    isnothing(node_range) && error("SLURM_JOB_NODELIST not defined, are you in a Slurm Job allocation?")
    return parse_nodes_topo(node_range)
end


"""
    parse_job_topo(job_id; running=false)

Retrieves the job's topology using `sacct`. If there is no entry then it tries with `squeue`
(or if `running == true`).
"""
function parse_job_topo(job_id::Int; running=false)
    raw_steps_node_lists = running ? "" : readchomp(`sacct -j $job_id -P -n -o 'NodeList'`)
    if !isempty(raw_steps_node_lists)
        steps_node_lists = split(raw_steps_node_lists, '\n') .|> strip
        node_list = unique(Iterators.flatten(expand_range.(steps_node_lists)))
    else
        # The job may be currently running, therefore there is no entry in the Slurm database
        raw_node_list = readchomp(`squeue --jobs=$job_id -h -O 'NodeList'`)  # May be inaccurate for some cases?
        isempty(raw_node_list) && error("no node list for job $job_id")
        node_list = expand_range(raw_node_list)
    end
    return parse_nodes_topo(node_list)
end


"""
    parse_partition_topo(partition)

Retrieves the whole `partition` topology.
"""
function parse_partition_topo(partition)
    raw_partition_info = readchomp(`scontrol show partition $partition`)
    partition_info = split(raw_partition_info, r"\s"; keepempty=false)
    nodes = nothing
    for info in partition_info
        !startswith(info, "Nodes=") && continue
        _, nodes = split(info, '=')
    end
    isnothing(nodes) && error("could not find 'Nodes' in the partition info")

    node_list = expand_range(nodes)
    nodes = parse_topo()
    filter_topo!(nodes, node_list)
    return nodes
end


"""
    parse_nodes_topo(node_range::String)
    parse_nodes_topo(node_list::Vector{<:AbstractString})

Retrieves the topology of the given nodes.
"""
parse_nodes_topo(node_range::String) = parse_nodes_topo(expand_range(node_range))

function parse_nodes_topo(node_list::Vector{<:AbstractString})
    nodes = parse_topo()
    filter_topo!(nodes, node_list)
    return nodes
end


"""
    parse_topo()

Retrieves the topology of the entire cluster.
"""
function parse_topo()
    raw_topo = readchomp(`scontrol show topo`)
    raw_topo = split.(split(raw_topo, '\n'), ' ')

    nodes = Dict{String, TopoEntity}()
    for entity in raw_topo
        parse_topo_entity!(nodes, entity)
    end

    fill_backedges!(nodes)
    return nodes
end


"""
    filter_topo!(topology::Dict{String, TopoEntity}, node_list::Vector{<:AbstractString})

Removes `node_list` from the given `topology`.
"""
function filter_topo!(nodes::Dict{String, TopoEntity}, node_list::Vector{<:AbstractString})
    # Keep all switches but only nodes within `node_list`
    filter!(p -> !p.second.is_node || p.first in node_list, nodes)

    # Remove switches with no connections. Repeat until the network doesn't change.
    changed = true
    while changed
        changed = false
        filter!(p -> p.second.is_node || !isempty(p.second.connections), nodes)
        for entity in values(nodes)
            entity.is_node && continue
            prev_length = length(entity.connections)
            # Remove connections to nodes not in the topology
            filter!(node -> haskey(nodes, node), entity.connections)
            length(entity.connections) != prev_length && (changed = true)
        end
    end

    # Cleanup backedges
    for entity in values(nodes)
        entity.is_node && continue
        filter!(node -> haskey(nodes, node), entity.backedges)
    end

    return nodes
end


"""
    available_nodes(partition::String; completing=true, allocated=false)

Returns a `Vector{String}` of the available nodes of the given `partition`.

If `completing == true`, then nodes in the `"completing"` state (which will be available soon) will
be included.

If `allocated == true`, then nodes in the `"allocated"` state (which are currently in use) will
be included.
"""
function available_nodes(partition::String; completing=true, allocated=false)
    # see `man sinfo`
    format = "%T|%N"  # state, node list
    partition_info = readchomp(`sinfo -p $partition -h -o $format`)
    partition_info = split.(split(partition_info, '\n'; keepempty=false), '|')

    all_suffixes = ['*', '~', '#', '!', '%', '@', '^', '-']  # All `sinfo` node state suffixes

    # Acceptable combinaisons of node states
    available_states = ["idle", "power_up"]
    available_suffixes = ['#']  # powered up

    completing && push!(available_states, "completing")  # soon available
    allocated && push!(available_states, "allocated", "allocated+")  # currently in use

    bad_suffixes = setdiff(all_suffixes, available_suffixes)

    available_node_list = []
    for nodes in partition_info
        any(startswith.(Ref(nodes[1]), available_states)) || continue
        any(endswith.(Ref(nodes[1]), bad_suffixes)) && continue
        push!(available_node_list, nodes[2])
    end

    available_nodes = String[]
    if !isempty(available_node_list)
        append!(available_nodes, expand_range.(available_node_list)...)
    end
    return available_nodes
end


"""
    filter_available_nodes!(topo::Dict{String, TopoEntity}, partition::String; completing=true, allocated=false)

Utility combining [`available_nodes`](@ref) and [`filter_topo!`](@ref). Only available nodes in the
`partition` are kept in the `topo`logy.
"""
function filter_available_nodes!(topo::Dict{String, TopoEntity}, partition::String; completing=true, allocated=false)
    nodes = available_nodes(partition; completing, allocated)
    filter_topo!(topo, nodes)
end


"""
    topo_max_depth(topo::Dict{String, TopoEntity})

Depth of the topology: the maximum switch level.
"""
topo_max_depth(topo::Dict{String, TopoEntity}) = maximum(e -> e.level, values(topo))


"""
    get_switch_nodes(topo::Dict{String, TopoEntity}, switch::String)

The list of nodes *directly* connected to the `switch` in the `topo`logy.
"""
get_switch_nodes(topo::Dict{String, TopoEntity}, switch::String) = filter(n -> topo[n].is_node, topo[switch].connections)


function get_root_node(topo::Dict{String, TopoEntity})
    max_level = -1
    max_level_node = nothing
    is_alone = false
    for (name, entity) in topo
        entity.is_node && continue
        if entity.level > max_level
            max_level = entity.level
            max_level_node = name
            is_alone = true
        elseif entity.level == max_level
            is_alone = false
        end
    end

    isnothing(max_level_node) && error("Couldn't find the root node: no switches with level ≥ 0")
    !is_alone && error("Max level $max_level contains multiple switches: no unique root node")
    return max_level_node
end


function visit_tree_breadth_first(f, topo::Dict{String, TopoEntity}, start_switch::TopoEntity)
    # Calls `f` for each `entity` (a `TopoEntity`), until `f(entity, distance)` returns `true`.
    # `distance` being the distance to `start_switch`, in switches.
    # This is done by parsing the tree from `start_switch` and by working upwards, each step
    # recursing into the lower entity first before going higher up the tree.
    stack = Set()

    function recurse_downwards(connections, distance)
        for name in connections
            name in stack && continue
            entity = topo[name]
            f(entity, distance) && return true
            push!(stack, name)
            !entity.is_node && recurse_downwards(entity.connections, distance+1) && return true
        end
        return false
    end

    function recurse_upwards(switch, distance)
        recurse_downwards(switch.connections, distance) && return true  # Direct connections => no increase in distance
        for name in switch.backedges
            name in stack && continue
            entity = topo[name]
            f(entity, distance) && return true
            push!(stack, name)
            recurse_upwards(entity, distance+1) && return true
        end
        return false
    end

    return recurse_upwards(start_switch, 0)
end


function select_minimal_tree_depth(topo::Dict{String, TopoEntity}, node_count::Int)
    # Returns `(list_of_level_0_switches, solution_nodes, solution_depth)` with `solution_nodes`
    # containing at least `node_count`.
    topo_leaf_switches = filter(p -> !p.second.is_node && p.second.level == 0, topo)

    # Bruteforce on all switches of level 0: this maximizes the chance of finding the optimal solution.
    optimal_depth = typemax(Int)
    optimal_solution = String[]
    optimal_solution_nodes = String[]
    for (start_switch_name, _) in topo_leaf_switches
        stack = String[start_switch_name]
        solution_nodes = get_switch_nodes(topo, start_switch_name)
        current_node_count = length(solution_nodes)

        # Visit the tree breadth-first, bottom to top, while collecting all switches which we want,
        # sorted by distance.
        matches = Tuple{Int, Int, String}[]
        visit_tree_breadth_first(topo, topo[last(stack)]) do entity, distance
            (entity.is_node || entity.level > 0 ||entity.name in stack) && return false
            # Do not count connections to nodes already part of the solution
            connected_nodes = count(c -> topo[c].is_node && !(c in solution_nodes), entity.connections)
            match = (distance, -connected_nodes, entity.name)  # `-connected_nodes` for sorting
            # Insert into `matches` by sorting by distance (ascending), then connections (descending), then name.
            i = searchsortedfirst(matches, match)
            insert!(matches, i, match)
            return connected_nodes + current_node_count ≥ node_count # If we find an optimal match, no need to continue
        end

        isempty(matches) && error("Could not find any valid switches from '$start_switch_name'")

        # `matches` is sorted by goodness, so we just have to take the matches until `node_count` is satisfied
        while current_node_count < node_count && !isempty(matches)
            _, _, switch_name = popfirst!(matches)
            push!(stack, switch_name)
            switch_nodes = get_switch_nodes(topo, switch_name)
            append!(solution_nodes, switch_nodes)
            unique!(solution_nodes)  # Make sure we don't include a node multiple times into the solution
            current_node_count = length(solution_nodes)
        end

        current_node_count < node_count && continue

        # Keep the solution with the lowest depth, break ties using the one with the lowest amount of switches
        solution_depth = maximum(s -> topo[s].level, stack)
        if solution_depth < optimal_depth || (solution_depth == optimal_depth && length(stack) < length(optimal_solution))
            optimal_depth = solution_depth
            optimal_solution = stack
            optimal_solution_nodes = solution_nodes
        end
    end

    isempty(optimal_solution) && error("Could not find any set of switches able to contain $node_count nodes")
    return optimal_solution, optimal_solution_nodes, optimal_depth
end


"""
    contiguous_node_list(partition::String, node_count::Int; all=false, kwargs...)

Parses the topology of `partition`, filters it with [`filter_available_nodes!`](@ref) if `all == false`,
then calls `contiguous_node_list(topo, node_count)`.

If `all == false`, then `kwargs` are passed to [`filter_available_nodes!`](@ref).
"""
function contiguous_node_list(partition::String, node_count::Int; all=false, kwargs...)
    partition_topo = parse_partition_topo(partition)

    # TODO: right now we exclude nodes which are unavailable, but what if we want to exclude switches
    # with a node currently in use? This way we could completely eliminate most network effects.

    !all && filter_available_nodes!(partition_topo, partition; kwargs...)
    return contiguous_node_list(partition_topo, node_count)
end


"""
    contiguous_node_list(topo::Dict{String, TopoEntity}, node_count::Int)

Returns a subset of `node_count` nodes in the `topo` forming a contiguous allocation, minimizing
the number of switch levels and switches.

Without Slurm's topology-aware allocation, this allows to build and check the quality of an
allocation's topology without doing it.

Returns tuple of `Vector{String}` of node names and switches.
"""
function contiguous_node_list(topo::Dict{String, TopoEntity}, node_count::Int)
    # The goal here is to return a set of `node_count` nodes which are as close to each other as
    # possible, by minimizing the number of switches between them. This can be used to manually
    # perform topology-aware allocations, when the Slurm controller does not offer it.

    available_node_count = count(p -> p.second.is_node, topo)
    if available_node_count < node_count
        error("Requested $node_count, but only $available_node_count nodes are available")
    end

    # List switches, sorted by how many nodes they are connected to
    switches = []
    for (name, entity) in topo
        entity.is_node && continue
        connected_nodes = count(n -> topo[n].is_node, entity.connections)
        push!(switches, connected_nodes => name)
    end
    sort!(switches)

    # Try to pick the first switch with at least `node_count` nodes (if any)
    i_switch = searchsortedfirst(switches, node_count; by=first)
    if i_switch <= length(switches)
        switch = last(switches[i_switch])
        switch_nodes = get_switch_nodes(topo, switch)
        return switch_nodes[1:node_count]
    end

    # Now things get dirty: since there is no trivial solution, we need to find a set of `node_count`
    # nodes in the topology which minimizes the maximum depth of the topology.
    switches, nodes = select_minimal_tree_depth(topo, node_count)

    if length(nodes) < node_count
        error("Oops, the solution does not have enough nodes, expected $node_count, got $(length(nodes)) nodes")
    end

    return nodes, switches
end


function build_graph(nodes::Dict{String, TopoEntity}; excluded=[])
    # using Graphs

    vertices = UInt[]
    vertices_labels = String[]
    vertices_map = Dict{String, UInt}()
    for (k, v) in nodes
        k in excluded && continue
        node_i = length(vertices) + 1
        push!(vertices, node_i)
        push!(vertices_labels, k)
        vertices_map[k] = node_i
    end

    g = SimpleGraph(length(vertices))
    for (node_name, node) in nodes
        k in excluded && continue
        node_i = vertices_map[node_name]
        for connection in node.connections
            connected_node_i = vertices_map[connection]
            if !add_edge!(g, node_i, connected_node_i)
                error("Could not connect $node_i to $connection")
            end
        end
    end

    return g, vertices_labels
end


"""
    topo_to_graphviz_script(io, topo::Dict{String, TopoEntity}; excluded=[], highlight=[], max_line_width=0)

Writes the `topo`logy to `io` in the form of a GraphViz graph file.
Node names in `excluded` are excluded from the file, which `highlight` are highlighted in green.

`max_line_width` is the maximum number of characters per line, including the 4 spaces indent. Widths
too small for a single graph node name will have only one node per line.
"""
function topo_to_graphviz_script(io, nodes::Dict{String, TopoEntity}; excluded=[], highlight=[], max_line_width=0)
    # The best 'circo' layout is obtained by placing switches first, then nodes,
    # as it will create nice circles around the switches.
    # We sort by placing switches first, sorted by ascending level, then nodes, in alphabetical order.
    sorted_nodes = sort(collect(nodes); by=((k, v),) -> (v.is_node, -v.level, k))
    filter!(!in(excluded) ∘ first, sorted_nodes)

    # Header
    indent = "    "
    println(io, "digraph {")
    println(io, indent, "graph [layout=circo rankdir=BT normalize=true mindist=0.1]")

    # Print the node+switch range as a comment in order to easily replicate the topology
    println(io)
    topo_range = node_list_to_range(first.(sorted_nodes))
    println(io, indent, "# ", topo_range)

    # Nodes
    line_width = 0
    for (node_name, node) in sorted_nodes
        node_text = node_name
        if !node.is_node
            node_text *= " [color=red]"
        elseif node_name in highlight
            node_text *= " [color=green]"
        end

        node_text_len = length(node_text)
        if line_width + 2 + node_text_len > max_line_width || line_width == 0
            print(io, "\n", indent, node_text)
            line_width = length(indent)
        else
            print(io, "; ", node_text)
            line_width += 2
        end
        line_width += node_text_len
    end
    println(io)

    # Edges
    for (node_name, node) in sorted_nodes
        node.is_node && continue
        line_width = length(indent) + 1
        print(io, indent, '{')
        for (i, connection) in enumerate(node.connections)
            connection_len = (i == 1 ? 0 : 1) + length(connection)
            if line_width + 1 + connection_len > max_line_width && i != 1
                print(io, '\n', indent, ' ', connection)
                line_width = length(indent) + 1 + connection_len
            elseif i > 1
                print(io, ' ', connection)
                line_width += 1 + connection_len
            else
                print(io, connection)
                line_width += connection_len
            end
        end

        switch_text = "} -> " * node_name
        if line_width + length(switch_text) > max_line_width
            print(io, '\n', indent)
        end
        println(io, switch_text)
    end

    println(io, "}")
end
