"""Independent P/Ipopt comparisons on identical forecast inputs."""
from pathlib import Path
import numpy as np
from power_fm_foresight._config import read_json, write_json
from power_fm_foresight._data.coordinates import bus_qg
from power_fm_foresight._solver.checks import check_solution
from power_fm_foresight.scoring import paired_error
from power_fm_foresight._training.physics import physics


def same_input_p(folder, prediction, network, solver):
    leads=np.array([0,6,12,18,23]);folder=Path(folder)
    path=folder/'P_solver_references.json';records=read_json(path) if path.exists() else {}
    state={k:np.full((1,len(leads),n),np.nan) for k,n in (('vm',network.n_bus),('va_rad',network.n_bus),('pg_mw',network.n_gen),('qg_mvar',network.n_gen))}
    demand={k:prediction[k][:,leads] for k in ('pd_mw','qd_mvar')};valid=np.zeros((1,len(leads)),dtype=bool)
    for j,h in enumerate(leads):
        key=str(h);pd=demand['pd_mw'][0,j];qd=demand['qd_mvar'][0,j]
        if key not in records:
            solved=solver.solve(pd,qd);check=check_solution(network,pd,qd,solved)
            records[key]=dict(pd_mw=pd.tolist(),qd_mvar=qd.tolist(),solver=solved,check=check)
            write_json(path,records)
        saved=records[key]
        np.testing.assert_array_equal(saved['pd_mw'],pd);np.testing.assert_array_equal(saved['qd_mvar'],qd)
        valid[0,j]=saved['check']['valid']
        if valid[0,j]:
            for k in state:state[k][0,j]=saved['solver']['state'][k]
    state['qg_mvar']=bus_qg(network,state['qg_mvar'])
    reference=physics(dict(state,**demand),network)
    raw={k:v[:,leads] for k,v in prediction.items()}
    result=dict(same_inputs=True,mode='P',sample_ids=[0],lead_indices=leads.tolist(),
        resolved=int(valid.sum()),attempts=int(valid.size),same_input_approximation=paired_error(raw,reference,valid),
        unresolved_policy='Unavailable; solver nonconvergence is not declared infeasibility')
    write_json(folder/'P_audit.json',result)
    return result
