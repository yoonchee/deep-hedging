# CLAUDE.md

## Code Style & Rules
- Enforce strict type hints (`typing.Annotated`, `torch.Tensor`).
- All tensor transformations must include comments documenting shape changes: `# [Batch, Time, Features] -> [Batch, Features]`.
- Run tests after every code generation phase.