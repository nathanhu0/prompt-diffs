# Codex Notes

Start by reading:
- `CLAUDE.md`
- `model_organisms/CLAUDE.md`

Working conventions:
- Check `git status` before editing.
- Prefer repo-native scripts in `model_organisms/` over ad hoc scripts when promoting a workflow.
- Treat `claude_scripts/` as exploratory or legacy unless explicitly requested.
- Follow existing config-driven entrypoints and sweep patterns instead of adding parallel one-off paths.
- Do not revert unrelated local changes in a dirty worktree.
