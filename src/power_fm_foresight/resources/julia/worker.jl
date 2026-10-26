# Adapted from lumina-sdk distillation/RAMBO/{backend.py,julia/model.jl,julia/worker.jl}
# revision e75ffcd74854c14e2960b9a9411ddd72377e5239 (Apache-2.0).
# Copyright 2026 UChicago Argonne, LLC. See THIRD_PARTY_NOTICES.md.
# Changes: OPF-only protocol, explicit IDs/units, duals, iterations and primal starts.
using JuMP, Ipopt, PowerModels, JSON, LinearAlgebra
const MOI = JuMP.MOI
const CASES = Dict{String,Any}()
PowerModels.silence()
BLAS.set_num_threads(1)

function solve_request(request)
    path = request["casefile"]
    original = get!(CASES, path) do
        PowerModels.parse_file(path)
    end
    data = deepcopy(original)
    base = data["baseMVA"]
    bus_ids = sort(parse.(Int, collect(keys(data["bus"]))))
    gen_ids = sort(parse.(Int, collect(keys(data["gen"]))))
    branch_ids = sort(parse.(Int, collect(keys(data["branch"]))))
    bus_index = Dict(id => i for (i,id) in enumerate(bus_ids))
    length(request["pd_mw"]) == length(bus_ids) || error("demand dimension mismatch")
    seen = Set{Int}()
    for (_,load) in data["load"]
        id = load["load_bus"]
        id ∉ seen || error("multiple loads per bus require explicit allocation")
        push!(seen,id)
        i = bus_index[id]
        load["pd"] = request["pd_mw"][i] / base
        load["qd"] = request["qd_mvar"][i] / base
    end
    for (i,id) in enumerate(bus_ids)
        if id ∉ seen && (request["pd_mw"][i] != 0 || request["qd_mvar"][i] != 0)
            error("demand supplied at a bus without a load")
        end
    end
    if haskey(request,"initial_state") && request["initial_state"] !== nothing
        start = request["initial_state"]
        for (i,id) in enumerate(bus_ids)
            data["bus"][string(id)]["vm_start"] = start["vm"][i]
            data["bus"][string(id)]["va_start"] = start["va_rad"][i]
        end
        for (i,id) in enumerate(gen_ids)
            data["gen"][string(id)]["pg_start"] = start["pg_mw"][i] / base
            data["gen"][string(id)]["qg_start"] = start["qg_mvar"][i] / base
        end
    end
    optimizer = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level"=>0, "sb"=>"yes", "tol"=>request["tolerance"],
        "acceptable_tol"=>request["tolerance"], "bound_relax_factor"=>0.0,
        "max_iter"=>request["max_iter"], "max_cpu_time"=>Float64(request["seconds"]))
    pm = PowerModels.instantiate_model(data, ACPPowerModel, PowerModels.build_opf;
        setting=Dict("output"=>Dict("duals"=>true,"branch_flows"=>true)))
    result = PowerModels.optimize_model!(pm; optimizer=optimizer)
    status = termination_status(pm.model)
    record = Dict{String,Any}(
        "status"=>string(status),
        "success"=>status in (MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED),
        "primal_status"=>string(primal_status(pm.model)),
        "dual_status"=>string(dual_status(pm.model)),
        "solve_seconds"=>get(result,"solve_time",nothing),
        "iterations"=>MOI.get(backend(pm.model), MOI.BarrierIterations()),
        "bus_ids"=>bus_ids,"gen_ids"=>gen_ids,"branch_ids"=>branch_ids,
        "baseMVA"=>base,"initial_state_used"=>get(request,"initial_state",nothing) !== nothing)
    if has_values(pm.model) && haskey(result,"solution")
        s = result["solution"]
        record["objective"] = objective_value(pm.model)
        record["state"] = Dict(
            "vm"=>[s["bus"][string(id)]["vm"] for id in bus_ids],
            "va_rad"=>[s["bus"][string(id)]["va"] for id in bus_ids],
            "pg_mw"=>[get(get(s["gen"],string(id),Dict()),"pg",NaN)*base for id in gen_ids],
            "qg_mvar"=>[get(get(s["gen"],string(id),Dict()),"qg",NaN)*base for id in gen_ids])
        record["branch"] = Dict(field=>[get(get(s["branch"],string(id),Dict()),field,NaN)*base for id in branch_ids]
                                for field in ("pf","qf","pt","qt"))
        if has_duals(pm.model)
            record["dual_p_raw"] = [s["bus"][string(id)]["lam_kcl_r"] for id in bus_ids]
            record["dual_q_raw"] = [s["bus"][string(id)]["lam_kcl_i"] for id in bus_ids]
        end
    end
    return record
end

for line in eachline(stdin)
    try
        request = JSON.parse(line)
        started = time()
        stage = request["stage"]
        if stage == "versions"
            result = Dict{String,Any}("julia"=>string(VERSION), "JuMP"=>string(pkgversion(JuMP)),
                          "Ipopt"=>string(pkgversion(Ipopt)), "PowerModels"=>string(pkgversion(PowerModels)))
        elseif stage == "opf"
            result = solve_request(request)
        else
            error("unsupported stage: $stage")
        end
        result["wall_seconds"] = time()-started
        println("GENCO_RESULT ",JSON.json(result))
    catch error
        println("GENCO_RESULT ",JSON.json(Dict("success"=>false,"status"=>"ERROR",
            "error"=>sprint(showerror,error,catch_backtrace()))))
    end
    flush(stdout)
end
