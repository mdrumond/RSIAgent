Rename a reused dynamic-loop induction variable in SIMT `tla.vec.func` bodies.

Condition: a `mode="simt"` `tla.vec.func` body contains two sequential (sibling, not nested) dynamic `tla.range` loops and the second reuses the first loop's induction variable name. Preprocessing then rejects the kernel with `SyntaxError: dynamic Tla for induction variables cannot be used after the loop: <name>`, even when the second loop never reads that name as a value.

Action: give each loop its own induction name. Rename only the second loop's induction variable and update that loop body's uses of the index; e.g. `for j in tla.range(0, n, 1): ...` followed by `for k in tla.range(n, tla.Int32(COLS), 1): gm_pad[base + k] = tla.Float32(-5.0)`.

Exceptions: change nothing else — region nesting, bound expressions, dtype casts, flat-vs-coordinate indexing, any `p = base + j` intermediate, and the host compile/launch path stay as-is. Do not treat the message as an escape or scope error about the name being live after the loop; the discriminator is name reuse.