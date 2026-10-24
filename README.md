# GENCO Foresight

<img align="left" width="45%" src="./assets/graph_predict.png" alt="GENCO Foresight prediction graph">

<p>• <b>MODEL</b>: Extends pre-trained GENCO from AC-OPF solver to a probabilistic forecaster</p>
<br>
<p>• <b>DATA</b>: Temporal dimension from EIA ERCOT 2025, operating-stress calibration for voltage and line events</p>
<br>
<p>• <b>TRAINING</b>: Synthetic temporal dataset: 10,080 Ipopt-labeled hours driven by EIA ERCOT 2025 demand</p>
<br>
<p>• <b>OPTIMIZER</b> Anticipatory solver warm starts evaluated against cold starts (no solver benefit observed at 2.93× cost)</p>
<br>
<p>• <b>SYSTEM</b> Reproducible on Aurora (XPU) HPC and NVIDIA (CUDA) systems, with CPU fallback</p>

<br clear="all">

## References
[ERCO: EIA's Hourly Electric Grid Monitor](https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/balancing_authority/ERCO) from the Electric Reliability Council of Texas, Inc. (ERCO)

[GridFM graphkit](https://gridfm.github.io/gridfm-graphkit/) - A library to train, finetune and interact with a foundation model for the electric power grid. (Puech, et al., 2025). <span title="@article{puech2025gridfmdatakitv1pythonlibraryscalable,
  title={gridfm-datakit-v1: A Python Library for Scalable and Realistic Power Flow and Optimal Power Flow Data Generation},
  author={Alban Puech and Matteo Mazzonelli and Celia Cintas and Tamara R. Govindasamy and Mangaliso Mngomezulu and Jonas Weiss and Matteo Baù and Anna Varbella and François Mirallès and Kibaek Kim and Le Xie and Hendrik F. Hamann and Etienne Vos and Thomas Brunschwiler},
  journal={arXiv preprint arXiv:2512.14658},
  year={2025},
  url={https://arxiv.org/abs/2512.14658}
}">Citation</span>

[GENCO](https://arxiv.org/abs/2608.09921) - A Unified Neural Solver Embedded in a Development Framework for Steady-State Grid Analysis; (<u>GE</u>ometric <u>N</u>eural <u>C</u>orrective <u>O</u>ptimizer) (Puech et. al, 2026)

[IPOPT](https://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf) - On the implementation of an interior-point filter line-search
algorithm for large-scale nonlinear programming (Wächter, Biegler, 2005)