# GENCO Foresight

<table>
  <tr>
    <td width="35%" valign="top">
      <img src="./assets/graph_predict.png" alt="GENCO Foresight prediction graph" width="100%">
    </td>
    <td valign="top">
      <p>• Extends pre-trained GENCO from snapshot AC-OPF prediction to a probabilistic forecasting system</p>
      <br>
      <p>• Operating-stress calibration for voltage and line events, with held-out comparison against climatology and numerical references</p>
      <br>
      <p>• Synthetic temporal dataset: 10,080 Ipopt-labeled hours driven by EIA ERCOT 2025 demand</p>
      <br>
      <p>• Anticipatory solver warm starts evaluated against cold starts (no solver benefit observed at 2.93× cost)</p>
      <br>
      <p>• Reproducible on Aurora (XPU) HPC and NVIDIA (CUDA) systems, with CPU fallback</p>
    </td>
  </tr>
</table>


## References

[ERCO: EIA's Hourly Electric Grid Monitor](https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/balancing_authority/ERCO) from the Electric Reliability Council of Texas, Inc. (ERCO)

[GridFM graphkit](https://gridfm.github.io/gridfm-graphkit/) - A library to train, finetune and interact with a foundation model for the electric power grid.

[GENCO](https://arxiv.org/abs/2608.09921) - A Unified Neural Solver Embedded in a Development Framework for Steady-State Grid Analysis.