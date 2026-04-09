# Plugins and Presets

Two parts of the public surface are easy to miss because they are smaller than
the main `run` flows: plugins and presets.

## Plugins

The harness supports external dataset and model adapters through Python entry
points.

### Entry-Point Groups

- `guard_eval_harness.datasets`
- `guard_eval_harness.models`

At startup, the registry loader imports built-in modules and then discovers
these entry points.

### How To Check Discovery

```bash
geh list plugins
```

This command shows the active registry view for datasets and models after
built-ins and entry-point plugins have been loaded.

Use it to confirm that an installed plugin was actually discovered.

## Presets

Presets are code-defined benchmark suites exposed through:

```bash
geh list presets
```

At the moment, the built-in canonical preset is:

- `21x31`

Conceptually:

- packs are public, user-facing suites for `geh run --pack ...`
- presets are reproducible benchmark definitions used by higher-level workflows
  and reproduction efforts

## When To Use Which

Use a plugin when:

- you need to ship a new dataset or model adapter outside the core repo
- you want installation-time discovery through entry points

Use a preset when:

- you need a named, reproducible benchmark definition beyond the smaller pack
  surface
- you are organizing reproduction or benchmark program workflows

Use a pack when:

- you want the simplest public CLI entry point for a starter evaluation
