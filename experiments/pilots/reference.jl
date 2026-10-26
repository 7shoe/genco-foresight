using PowerModels, JSON
PowerModels.silence()
data = PowerModels.parse_file(ARGS[1])
println("REFERENCE_JSON ", JSON.json(Dict(
    "PowerModels" => string(pkgversion(PowerModels)),
    "bus_ids" => sort(parse.(Int, collect(keys(data["bus"])))),
    "reference_buses" => [b for b in values(data["bus"]) if b["bus_type"] == 3],
    "constraint_va_rad" => 0.0,
    "constraint" => "PowerModels.AbstractPolarModels constraint_theta_ref: va[i] == 0")))
