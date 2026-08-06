# CLAUDE.md

## Code Style & Rules
- Use PyTorch for all neural network modules (`nn.Module`).
- Enforce strict type hints (`typing.Annotated`, `torch.Tensor`).
- All tensor transformations must include comments documenting shape changes: `# [Batch, Time, Features] -> [Batch, Features]`.
- Avoid placeholder returns or stubbed logic (`pass`). Implement complete functions.
- Run tests after every code generation phase.

## Test Commands
- Run test suite: `pytest tests/ -v`
- Run single module test: `pytest tests/test_generator.py -v`