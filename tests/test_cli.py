"""
Tests for the CLI args → config-overrides partition.

CLI flags split into two buckets:
  - `_CLI_ONLY`: scaffolding (subcommand name, --frames, --num-error-samples,
    etc.) — must NOT leak into `C` as an override of a same-named config key.
  - everything else: applied as a config override after YAML load.

Adding a new stage flag without listing its `dest` in `_CLI_ONLY` would
silently start overriding any same-named config value. These tests make
that mistake fail loudly.
"""

import argparse

from swarmflow.cli import (_build_parser, _config_overrides, _CLI_ONLY,
                           _STAGE_ARGS)


def test_stage_only_flags_are_filtered_out():
    parser = _build_parser()
    args = parser.parse_args([
        'run',
        '--frames', 'last', '--every', '5',
        '--num-error-samples', '500', '--n-blocks', '2',
        '--skip-block-average',
        '--name', 'my-system', '--gpu', '0,1',
    ])
    overrides = _config_overrides(args)

    # CLI-only flags must NOT appear as overrides
    for cli_only in ('frames', 'every', 'num_error_samples',
                     'n_blocks', 'skip_block_average',
                     'config', 'command', 'from_', 'to'):
        assert cli_only not in overrides, \
            f'{cli_only!r} leaked into config overrides'

    # Real config flags must appear
    assert overrides['name'] == 'my-system'
    assert overrides['cuda_gpus'] == '0,1'   # split happens later in load_config


def test_stage_arg_adders_register_only_dests_in_CLI_ONLY():
    """Every dest registered by a `_STAGE_ARGS` adder must be listed in
    `_CLI_ONLY`. Forgetting one means that stage flag would override a
    config key of the same name without anyone noticing."""
    for stage_name, adder in _STAGE_ARGS.items():
        p = argparse.ArgumentParser()
        adder(p)
        for action in p._actions:
            if action.dest in ('help',):
                continue
            assert action.dest in _CLI_ONLY, (
                f'{stage_name} adder registers --{action.dest!s} but it is '
                f'not in _CLI_ONLY; it would silently shadow a config key')


def test_init_subcommand_does_not_require_config_load():
    """`init` is the one stage subcommand that scaffolds a project before
    config.yml exists. Its parsed args must be runnable without --config."""
    parser = _build_parser()
    args = parser.parse_args(['init', 'my-proj'])
    assert args.command == 'init'
    assert args.project_dir == 'my-proj'


def test_status_subcommand_accepts_global_options():
    parser = _build_parser()
    args = parser.parse_args(['status', '--name', 'foo'])
    assert args.command == 'status'
    assert args.name == 'foo'
