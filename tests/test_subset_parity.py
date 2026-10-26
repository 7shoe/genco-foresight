"""Fixed sampling and separation from the enclosing MPI launcher."""
import numpy as np
import pytest

from experiments.subset_parity import selected_ids, serial_environment


def test_fixed_subset_is_unique_repeatable_and_inside_intersection():
    intersection = np.arange(3, 6003, 3)
    first = selected_ids(intersection)
    assert len(first) == len(np.unique(first)) == 200
    np.testing.assert_array_equal(first, selected_ids(intersection))
    assert np.isin(first, intersection).all()
    assert not np.array_equal(first, intersection[:200])
    assert not np.array_equal(first, selected_ids(intersection, seed=43))
    with pytest.raises(ValueError):selected_ids(intersection[:199])


def test_cpu_reference_does_not_join_parent_mpi_group():
    env = {'PMIX_RANK': '1', 'PMIX_NAMESPACE': 'job', 'PALS_PMI': 'pmix',
        'PMI_RANK': '1', 'WORLD_SIZE': '2', 'LOCAL_RANK': '0', 'MASTER_PORT': '1000',
        'PBS_JOBID': 'job.pbs', 'LD_LIBRARY_PATH': '/vendor', 'OMP_NUM_THREADS': '1'}
    assert serial_environment(env) == {'PBS_JOBID': 'job.pbs', 'LD_LIBRARY_PATH': '/vendor', 'OMP_NUM_THREADS': '1'}
    assert env['PMIX_RANK'] == '1'
