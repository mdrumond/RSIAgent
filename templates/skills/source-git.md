# Local source history

Use `rsi_git` with an `argv` array for ordinary Git commands in the authoritative
controller-local project source. The remote Ascend Bash tool runs on another
filesystem; use source editing tools and `rsi_git` locally, then the normal
execution environment for device validation.

Inspect `status`, `diff`, and relevant history. Commit meaningful implementation
and optimisation milestones with a short message explaining the change, for
example `{"argv":["commit","-m","Reduce redundant tensor transfers"]}` after
staging the intended source paths. Prefer explicit paths over `add .`; inspect
the staged diff before committing. Preserve existing user changes and frozen
verifiers. Never automatically discard dirty files to make history clean.

History is advisory. A commit, clean tree, or particular commit count is not an
acceptance requirement and does not prove correctness. Use the normal independent
verifier and performance evidence. Setup creates no initial commit; the first
commit is your decision. The controller records HEAD and dirty state separately.

The project repository and attempt branch are managed outside source, so there
is no `.git` directory in source fingerprints or remote snapshots. Network
commands, remotes, and credential configuration are unavailable. Keep generated
artifacts, dependency trees, runtime binaries, caches, and credentials out of
history. Default excludes protect common generated paths during ordinary adds;
do not force-add excluded assets. Keep receipts outside source through the
normal artifact facility.
