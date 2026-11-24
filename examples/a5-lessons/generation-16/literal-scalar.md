When lowering fails with `as_numeric() takes 1 positional argument but 2 were given` at an `as_numeric(...)` call site, treat it as an arity mismatch: the pinned runtime signature exposes one positional parameter (`obj`), so no dtype argument is accepted. Apply this only to that exact two-positional error, not as a blanket conversion rule.

Replacement rule, one form per constant kind:
- Float constants: use the dtype constructor, e.g. `tla.Float32(-1.0)`.
- Integer constants: pass a bare integer to the one-argument helper, e.g. `tla.as_numeric(0)`.

Do not assume the helper accepts a dtype until confirmed against the pinned signature in the runtime in use. Keep other call-site semantics unchanged, and leave diagnosis and further implementation choices to the actor.