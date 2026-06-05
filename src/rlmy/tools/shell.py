"""
Shell tool with deny-list safety and time-based approval.

Usage:
    from rlmy.tools.shell import make_shell_tool
    standard_tools = [..., make_shell_tool()]
"""

import re
import signal
import subprocess
import time

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

_console = Console()


# =============================================================================
# Denied commands (from MeshClaw defaults.json, verbatim)
# =============================================================================

DENIED_PATTERNS: list[str] = [
    # AWS S3 writes
    r"aws s3 cp .* s3://.*",
    r"aws s3 mv .* s3://.*",
    r"aws s3 sync .* s3://.*",
    # Credential exfiltration
    r".*echo.*\$AWS_SECRET.*",
    r".*echo.*\$AWS_SESSION.*",
    r".*echo.*\$AWS_ACCESS.*",
    r".*printenv.*AWS.*",
    r".*env.*grep.*AWS.*",
    r".*python.*boto3.*get_credentials.*",
    r".*python.*botocore.*credentials.*",
    r".*curl.*169\.254\.169\.254.*",
    r".*wget.*169\.254\.169\.254.*",
    r".*curl.*\$AWS_SECRET.*",
    r".*curl.*\$AWS_ACCESS.*",
    r".*curl.*\$AWS_SESSION.*",
    # Destructive AWS
    r"aws autoscaling delete-.*",
    r"aws cloudformation delete-stack.*",
    r"aws dynamodb delete-table.*",
    r"aws ec2 delete-.*",
    r"aws ec2 terminate-instances.*",
    r"aws ecr delete-.*",
    r"aws ecs delete-.*",
    r"aws eks delete-cluster.*",
    r"aws elasticache delete-.*",
    r"aws elb delete-.*",
    r"aws elbv2 delete-.*",
    r"aws glue delete-.*",
    r"aws iam create-access-key.*",
    r"aws iam delete-.*",
    r"aws kinesis delete-.*",
    r"aws kms schedule-key-deletion.*",
    r"aws lambda delete-function.*",
    r"aws logs delete-.*",
    r"aws opensearch delete-.*",
    r"aws rds delete-.*",
    r"aws redshift delete-.*",
    r"aws route53 delete-.*",
    r"aws s3 rb.*",
    r"aws s3 rm.*",
    r"aws s3api delete-.*",
    r"aws secretsmanager delete-secret.*",
    r"aws sns delete-.*",
    r"aws sqs delete-.*",
    r"aws stepfunctions delete-.*",
    # IaC destroy
    r"cdk destroy.*",
    r"pulumi destroy.*",
    r"terraform destroy.*",
    # Dangerous system commands
    r"chmod 777.*",
    r"curl .* \| bash",
    r"curl .* \| sh",
    r"dd if=.*",
    r"mkfs.*",
    r"nc -e.*",
    r"ncat -e.*",
    r"rm -rf /.*",
    r"rm -rf ~.*",
    r"wget .* \| bash",
    # Destructive SQL
    r"DROP DATABASE.*",
    r"DROP TABLE.*",
    r"TRUNCATE TABLE.*",
    # Git footguns
    r"export AWS_ACCESS.*",
    r"export AWS_SECRET.*",
    r"git push\b.*",
    r"git reset --hard.*",
    # Kubernetes
    r"kubectl delete namespace.*",
    # Sensitive file reads
    r".*(cat|head|tail|less|more|strings|base64|cp).*~/.aws/.*",
    r".*(cat|head|tail|less|more|strings|base64|cp).*~/.ssh/.*",
    r".*(cat|head|tail|less|more|strings|base64|cp).*~/.gnupg/.*",
    r".*(cat|head|tail|less|more|strings|base64|cp).*~/.gpg/.*",
    r".*(cat|head|tail|less|more|strings|base64|cp).*~/.netrc.*",
    r".*(cat|head|tail|less|more|strings|base64|cp).*~/.git-credentials.*",
    r".*(cat|head|tail|less|more|strings|base64|cp).*~/.npmrc.*",
    r".*(cat|head|tail|less|more|strings|base64|cp).*~/.pypirc.*",
    r".*(cat|head|tail|less|more|strings|base64|cp).*~/.docker/config.json.*",
    r".*(cat|head|tail|less|more|strings|base64|cp).*~/.kube/config.*",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in DENIED_PATTERNS]


# =============================================================================
# Checking
# =============================================================================


def find_violation(command: str) -> str | None:
    """Returns the matched pattern if command is denied, else None."""
    for i, regex in enumerate(_COMPILED):
        if regex.search(command):
            return DENIED_PATTERNS[i]
    return None


# =============================================================================
# Time grants — two independent pools
# =============================================================================


class TimeGrants:
    """Two pools:
    - Safe pool:   one timer covering all non-denied commands.
    - Unsafe pool: one timer per deny-list pattern. Independent of each other.
    """

    def __init__(self):
        self._safe_expiry: float | None = None
        self._unsafe_expiry: dict[str, float] = {}  # pattern -> expiry

    def allow_safe_until(self, duration_sec: float):
        self._safe_expiry = time.time() + duration_sec

    def allow_pattern_until(self, pattern: str, duration_sec: float):
        self._unsafe_expiry[pattern] = time.time() + duration_sec

    def is_safe_approved(self) -> bool:
        if self._safe_expiry is None:
            return False
        if time.time() > self._safe_expiry:
            self._safe_expiry = None
            return False
        return True

    def is_pattern_approved(self, pattern: str) -> bool:
        exp = self._unsafe_expiry.get(pattern)
        if exp is None:
            return False
        if time.time() > exp:
            del self._unsafe_expiry[pattern]
            return False
        return True


# =============================================================================
# Terminal prompt
# =============================================================================

_DURATIONS = {"1min": 60.0, "10min": 600.0}


def _ask_permission(command: str, reason: str | None) -> tuple[str, str]:
    """Prompt the user about a command. Returns (decision, user_text).

    decision: "once" | "1min" | "10min" | "deny"
    user_text: whatever the user typed (for deny, this is their feedback).

    Renders a Rich panel — striking red for unsafe commands, blue for safe ones.
    """
    if reason:
        # ── Unsafe command: red, bold, impossible to miss ──
        body = Text()
        body.append("⚠️  UNSAFE COMMAND\n\n", style="bold red")
        body.append("  $ ", style="dim")
        body.append(command, style="bold white on red")
        body.append("\n\n")
        body.append("  Matched rule: ", style="dim red")
        body.append(reason, style="italic red")
        body.append("\n\n")
        body.append("  [1] ", style="bold green")
        body.append("Allow once\n")
        body.append("  [2] ", style="bold yellow")
        body.append("Allow this pattern for 1 minute\n")
        body.append("  [3] ", style="bold yellow")
        body.append("Allow this pattern for 10 minutes\n")
        body.append("  [*] ", style="bold red")
        body.append("Deny ", style="red")
        body.append("(your text becomes feedback to the agent)", style="dim")

        _console.print(
            Panel(
                body,
                title="[bold red on white] 🛑 BLOCKED — UNSAFE COMMAND 🛑 [/bold red on white]",
                border_style="bold red",
                padding=(1, 2),
                width=min(90, _console.width),
            )
        )
    else:
        # ── Safe command: blue, informational ──
        body = Text()
        body.append("🔧  Shell Command\n\n", style="bold blue")
        body.append("  $ ", style="dim")
        body.append(command, style="bold white")
        body.append("\n\n")
        body.append("  [1] ", style="bold green")
        body.append("Allow once\n")
        body.append("  [2] ", style="bold cyan")
        body.append("Allow all safe commands for 1 minute\n")
        body.append("  [3] ", style="bold cyan")
        body.append("Allow all safe commands for 10 minutes\n")
        body.append("  [*] ", style="bold red")
        body.append("Deny ", style="red")
        body.append("(your text becomes feedback to the agent)", style="dim")

        _console.print(
            Panel(
                body,
                title="[bold blue] 🐚 Shell Command Approval [/bold blue]",
                border_style="blue",
                padding=(1, 2),
                width=min(90, _console.width),
            )
        )

    # Temporarily restore default SIGINT so Ctrl+C works during input.
    # During RLM execution, our cooperative SIGINT handler is active (sets a flag
    # instead of raising KeyboardInterrupt). Without this, input() hangs.
    prev_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        raw = input("  ❯ ").strip()
    except (KeyboardInterrupt, EOFError):
        return ("deny", "cancelled by user")
    finally:
        signal.signal(signal.SIGINT, prev_handler)

    if raw == "1":
        return ("once", "")
    elif raw == "2":
        return ("1min", "")
    elif raw == "3":
        return ("10min", "")
    else:
        return ("deny", raw or "denied by user")


# =============================================================================
# Execution
# =============================================================================


def _execute(command, work_dir, timeout) -> str:
    """Run command via subprocess, return plain text output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir or None,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip()
    except subprocess.TimeoutExpired:
        return f"[Error] Command timed out after {timeout}s"
    except Exception as e:
        return f"[Error] {e}"


# =============================================================================
# Factory
# =============================================================================


def make_shell_tool(ask=_ask_permission):
    """Create the shell tool function. Add it to standard_tools.

    Args:
        ask: Callable(command, reason|None) -> (decision, feedback).
             Default is terminal input(). Replace for other UIs.
    """
    grants = TimeGrants()

    def run_shell(command: str, work_dir: str = "", timeout: int = 300) -> str:
        """Run a shell command on the user's machine.

        Every command requires user approval before execution.
        Approval can be granted once, or for a time window (1 or 10 minutes).

        Unsafe commands (destructive ops, credential access, force-push, etc.)
        are flagged separately and must be approved independently — a safe
        command grant does not cover them.

        If a command is denied, the user's feedback is returned — read it
        and adjust your approach.

        Args:
            command: The shell command to execute (e.g. "ls -la", "git status").
            work_dir: Working directory. Empty string = current directory.
            timeout: Max seconds to wait (default 300).
        """
        violation = find_violation(command)

        if violation:
            # Unsafe — needs its own pattern-specific grant
            if not grants.is_pattern_approved(violation):
                decision, feedback = ask(command, violation)

                if decision == "deny":
                    raise RuntimeError(f"DENIED by user: {feedback}")
                if decision in _DURATIONS:
                    grants.allow_pattern_until(violation, _DURATIONS[decision])
        else:
            # Safe — needs safe grant
            if not grants.is_safe_approved():
                decision, feedback = ask(command, None)

                if decision == "deny":
                    raise RuntimeError(f"DENIED by user: {feedback}")
                if decision in _DURATIONS:
                    grants.allow_safe_until(_DURATIONS[decision])

        return _execute(command, work_dir, timeout)

    return run_shell
