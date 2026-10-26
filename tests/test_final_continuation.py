"""Budget extensions must not silently change the resumed scientific recipe."""
from copy import deepcopy
import unittest

from experiments.v2.final_continuation import validate_config


class ContinuationConfigTest(unittest.TestCase):
    def setUp(self):
        self.parent = {'paths': {'training_run': '/old', 'runs_root': '/old-runs', 'data': '/data'},
                       'stage_b': {'training_budget_seconds': 16200, 'training_samples': 8,
                                   'optimizer': {'lr': 1e-4}, 'deterministic_algorithms': False},
                       'v2': {'run_identity': 'parent', 'protocol_sha256': 'old'}}
        self.child = deepcopy(self.parent)
        self.child['paths'].update(training_run='/new', runs_root='/new-runs')
        self.child['stage_b']['training_budget_seconds'] = 19800
        self.child['v2'].update(run_identity='child', protocol_sha256='new', continuation={'parent': 'parent'})

    def test_declared_extension(self):
        validate_config(self.parent, self.child)

    def test_reject_scientific_changes(self):
        for key, value in [('training_samples', 2), ('deterministic_algorithms', True), ('optimizer', {'lr': 2e-4})]:
            with self.subTest(key=key):
                changed = deepcopy(self.child)
                changed['stage_b'][key] = value
                with self.assertRaises(ValueError):
                    validate_config(self.parent, changed)

    def test_reject_wrong_budget_and_parent_overwrite(self):
        changed = deepcopy(self.child)
        changed['stage_b']['training_budget_seconds'] = 23400
        with self.assertRaises(ValueError):
            validate_config(self.parent, changed)
        for group, key in [('paths', 'training_run'), ('v2', 'run_identity')]:
            changed = deepcopy(self.child)
            changed[group][key] = self.parent[group][key]
            with self.assertRaises(ValueError):
                validate_config(self.parent, changed)

    def test_reject_data_change(self):
        self.child['paths']['data'] = '/other-data'
        with self.assertRaises(ValueError):
            validate_config(self.parent, self.child)


if __name__ == '__main__':
    unittest.main()
