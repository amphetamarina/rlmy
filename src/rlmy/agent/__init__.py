"""
RLMY Agent — core agent logic, trajectory, and workspace management.

Key exports:
    InterruptableRLM — interrupt-safe RLM subclass with prior trajectory injection
    RLMContext — mutable context passed to contextual tools
"""

# NOTE: Heavy imports (InterruptableRLM, etc.) are deferred to avoid loading
# the entire agent stack on `import rlmy`. Import directly from submodules:
#   from rlmy.agent.rlm import InterruptableRLM, RLMContext
#   from rlmy.agent.trajectory import save_trajectory, load_trajectory
#   from rlmy.agent.sandbox import SandboxManager

