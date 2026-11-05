"""Scientific parameters must survive the dashboard interface unchanged."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock


PAGE = Path('ui/pages/short_medium_benchmarking.py')


class ReferenceParameterTests(unittest.TestCase):
    def test_default_dry_confirmation_matches_reference_and_explicit_overrides_survive(self):
        config_path = Path('ai_weather/config.py')
        assignment = next(node for node in ast.parse(config_path.read_text()).body
                          if isinstance(node, ast.Assign) and any(
                              isinstance(target, ast.Name) and target.id == 'dry_extent'
                              for target in node.targets))
        self.assertEqual(ast.literal_eval(assignment.value), 0)
        for path in (Path('apps/benchmarking/ROMP/momp/params.py'),
                     Path('apps/benchmarking/ROMP/momp/params/config.in')):
            assignment = next(node for node in ast.parse(path.read_text()).body
                              if isinstance(node, ast.Assign) and any(
                                  isinstance(target, ast.Name) and target.id == 'dry_extent'
                                  for target in node.targets))
            for config, expected in ((SimpleNamespace(), 0), (SimpleNamespace(dry_extent=21), 21)):
                namespace = {'config': config}
                exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), 'exec'), namespace)
                self.assertEqual(namespace['dry_extent'], expected)

    def test_runtime_orders_day_bins_chronologically_before_scoring(self):
        path = Path('run_momp.py')
        function = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'validate_runtime_config')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        cfg = SimpleNamespace(init_type='daily', init_days=(),
                              verification_window_list=((1, 15),), tolerance_days_list=(3,),
                              max_forecast_day=15, day_bins=((11, 15), (1, 5), (6, 10)),
                              probabilistic=False)
        namespace['validate_runtime_config'](cfg)
        self.assertEqual(cfg.day_bins, ((1, 5), (6, 10), (11, 15)))

    def test_dashboard_preserves_configured_dry_threshold(self):
        tree = ast.parse(PAGE.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'render_shared_settings')
        # Use widget defaults as returned values so the actual config-to-widget
        # mapping is exercised without scientific imports or external inputs.
        st = MagicMock()
        column = MagicMock()
        column.number_input.side_effect = lambda *args, **kwargs: kwargs['value']
        st.columns.side_effect = lambda values, **kwargs: [column] * (values if isinstance(values, int) else len(values))
        st.selectbox.side_effect = lambda label, options, **kwargs: options[kwargs.get('index', 0)]
        config = SimpleNamespace(wet_init=1., wet_threshold=20., wet_spell=4,
                                 dry_threshold=2.5, dry_spell=8, dry_extent=30,
                                 nc_mask=None)
        namespace = {'st': st, 'config': config, 'section_header': MagicMock(),
                     'EXTERNAL_DATA_DIR': Path('/data'), 'ONSET_REFERENCE_DIR': Path('/reference'),
                     'Dict': dict, 'Any': object}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(PAGE), 'exec'), namespace)
        settings = namespace['render_shared_settings']()
        self.assertEqual(settings['dry_threshold'], 2.5)
        self.assertEqual(settings['wet_spell_length'], 4)
        self.assertEqual(settings['dry_spell_limit'], 8)
        self.assertEqual(settings['dry_spell_search_extension'], 30)

    def test_preflight_preserves_full_enabled_confirmation_horizon(self):
        tree = ast.parse(PAGE.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'page')
        assignments = [node for node in ast.walk(function) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id in
                               {'confirmation_days', 'required_days'} for target in node.targets)]
        assignments.sort(key=lambda node: node.lineno)
        self.assertEqual(len(assignments), 2)
        for extent, dry_spell, expected in ((21, 7, 35), (0, 7, 17), (21, 0, 17), (2, 1, 17)):
            namespace = {'window': {'max_forecast_day': 15},
                         'shared_settings': {'wet_spell_length': 3,
                                             'dry_spell_search_extension': extent,
                                             'dry_spell_limit': dry_spell}}
            with self.subTest(extent=extent, dry_spell=dry_spell):
                exec(compile(ast.Module(body=assignments, type_ignores=[]), str(PAGE), 'exec'), namespace)
                self.assertEqual(namespace['required_days'], expected)
