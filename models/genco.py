"""Thin upstream GENCO integration; immutable checkpoint normalizer."""
from __future__ import annotations

from pathlib import Path
import torch
import yaml
from torch_geometric.data import Batch

# just check: https://github.com/gridfm/gridfm-graphkit
from gridfm_graphkit.io.config_version import resolve_config
from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.models.gnn_heterogeneous_gns import GNS_heterogeneous
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
from gridfm_graphkit.datasets.task_transforms import OptimalPowerFlowTransforms
from data.grids import check_case


class Genco:
    """
    Model class wrapper for GENCO
    (GEometric Neural Corrective Optimizer)
    """

    def __init__(self, checkpoint_root, settings, device="cpu", grid=None):
        """Load model config"""
        # check config
        folder = Path(checkpoint_root) / settings["directory"]
        config_path = folder / settings["configuration"]
        raw = yaml.safe_load(config_path.read_text())

        # assign model
        self.config = resolve_config(raw, config_path=str(config_path), conversion_mode="auto_inline")
        self.args = NestedNamespace(**self.config)
        self.case = settings["directory"].split("/")[0]
        networks = self.args.data.networks
        if list(networks) != [self.case]:
            raise ValueError(f"Case mismatch: checkpoint YAML has {networks!r}; expected {[self.case]!r}")
        if grid is not None:
            check_case(grid.case, self.case, "checkpoint YAML")
        self.device = torch.device(device)
        self.grid = grid
        self.model = GNS_heterogeneous(self.args)
        state = torch.load(folder / settings["weights"], map_location="cpu", weights_only=True)
        keys = {k.replace("model._orig_mod.", "model.").removeprefix("model."): v for k,v in state.items()}
        self.model.load_state_dict(keys, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)

        # data preprocessing
        stats = torch.load(folder / settings["normalizer"], map_location="cpu", weights_only=True)
        if self.case not in stats:
            raise ValueError(f"Case mismatch: normalizer has no statistics for {self.case!r}")
        self.normalizer = HeteroDataMVANormalizer(self.args)
        self.normalizer.fit_from_dict(stats[self.args.data.networks[0]])
        self.transform = OptimalPowerFlowTransforms(self.args)

    def prepare(self, graph):
        """Use Genco preprocessing"""
        check_case(self.case, getattr(graph, "case_id", None), "input graph")
        graph = graph.clone()
        self.normalizer.transform(graph)
        graph = self.transform(graph)
        for kind in ("bus", "gen"):
            graph[kind].label_available = torch.isfinite(graph[kind].y)
        return graph

    def batch(self, graphs):
        """Preps each graph, combines them into one PyTorch Geometric batc"""
        return Batch.from_data_list([self.prepare(g) for g in graphs]).to(self.device)

    def predict(self, graphs, return_embeddings=False):
        """Run mode eval-style"""
        batch = self.batch(graphs)
        with torch.no_grad():
            output = self.model(batch, return_embeddings=return_embeddings)
        if return_embeddings:
            output, embeddings = output
        self.normalizer.inverse_output(output, batch)
        if return_embeddings:
            return output, embeddings, batch
        return output
