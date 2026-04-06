"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from my_env_v4 import MyEnvV4Action, MyEnvV4Env
IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

# Max possible reward: each token contributes 0.1, across all steps
_MAX_REWARD_PER_STEP = MAX_TOKENS * 0.1
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are interacting with a simple echo environment.
    Each turn you must send a message. The environment will echo it back.
    Reward is proportional to message length: reward = len(message) * 0.1
    Your goal is to maximize total reward by sending meaningful, substantive messages.
    Reply with exactly one message string — no quotes, no prefixes, just the message text.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Last echoed message: {last_echoed!r}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Send your next message.
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, last_echoed, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "hello"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "hello"


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset() # OpenENV.reset()
        last_echoed = result.observation.echoed_message
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            message = get_model_message(client, step, last_echoed, last_reward, history)

            result = await env.step(MyEnvV4Action(message=message))
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_echoed = obs.echoed_message
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)

            logger.info(
                f"[REQ-{request_id}] RESULT: infinite_loop=True, "
                f"tests_passed=0, tests_failed=0, reward=-1.00"
            )

            observation = self._apply_transform(observation)
            return observation

        try:
            full_result = self._executor.run(combined_code, timeout=timeout)
            execution_time = time.time() - start_time

            logger.info(
                f"[REQ-{request_id}] Execution completed in {execution_time:.2f}s, exit_code={full_result.exit_code}"
            )

            # Log stderr if present (often contains errors or test output)
            if full_result.stderr:
                stderr_preview = (
                    full_result.stderr[:500] + "..."
                    if len(full_result.stderr) > 500
                    else full_result.stderr
                )
                logger.debug(f"[REQ-{request_id}] Stderr: {stderr_preview}")

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"[REQ-{request_id}] EXECUTION FAILED after {execution_time:.2f}s: {e}"
            )
            raise

        # Parse test results from execution output
        tests_passed, tests_failed = self._parse_test_results(
            full_result.stdout, full_result.stderr
        )

        # Infer compilation status from execution
        # If tests ran, code compiled successfully
        # If exit_code != 0 and no tests ran, code didn't compile
        code_compiles = (
            full_result.exit_code == 0  # Clean execution
            or tests_passed > 0  # Some tests passed (code must have compiled)
            or tests_failed > 0  # Some tests failed (code compiled but tests failed)
        )

        # If no tests detected and non-zero exit, check for compilation errors
        if not code_compiles and tests_passed == 0 and tests_failed == 0:
            # Check stderr for compilation errors
            stderr_lower = full_result.stderr.lower()
            if any(
                err in stderr_lower
                for err in ["error", "syntax", "undefined", "loadError"]
            ):
                code_compiles = False
            else:
                # If no clear compilation error, assume it compiled
                code_compiles = True

        # Calculate reward based on compilation and test results
        reward = self._calculate_reward(code_compiles, tests_passed, tests_failed)

        # Log final results
        logger.info(
            f"[REQ-{request_id}] RESULT: compiles={code_compiles}, "
            f"tests_passed={tests_passed}, tests_failed={tests_failed}, reward={reward:.2f}"
        )

        # Update environment state
        self._state.step_count += 1
        self._state.last_exit_code = full_result.exit_code
        self._state.last_code_compiles = code_compiles
        self._state.total_tests_passed = tests_passed
        self._state.total_tests_failed = tests_failed

        # Build observation
        observation = JuliaObservation(
            stdout=full_result.stdout,
            stderr=full_result.stderr,
            exit_code=full_result.exit_code,
            reward=reward,
            metadata={
                "core_code": action.core_code,
                "test_code": action.test_code or "",
            },
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            code_compiles=code_compiles,
        )

        # Apply safety and quality transforms
        observation = self._apply_transform(observation)

        return observation

    def _parse_test_results(self, stdout: str, stderr: str) -> tuple[int, int]:
        """
        Parse Julia test output to count passed/failed tests.

        Julia's Test module outputs results like:
        "Test Summary:      | Pass  Fail  Total  Time"
        "Add function Tests |    1     1      2  1.5s"

        Also checks error messages:
        "Some tests did not pass: 1 passed, 1 failed, 0 errored, 0 broken."

        Args:
            stdout: Standard output from Julia execution
            stderr: Standard error from Julia execution

        Returns:
            Tuple of (tests_passed, tests_failed)
        """
        # Combine stdout and stderr for analysis
        passed = 0
        failed = 0
        output = stdout + "\n" + stderr

        # Method 1: Look for "Some tests did not pass" error message
        # Pattern: "Some tests did not pass: X passed, Y failed, Z errored, W broken."
        error_pattern = r"Some tests did not pass:\s*(\d+)\s+passed,\s*(\d+)\s+failed,\s*(\d+)\s+errored"
        match = re.search(error_pattern, output)

        if match:
            passed = int(match.group(1))
            failed = int(match.group(2))
            errored = int(match.group(3))
            return passed, failed + errored  # Treat errors as failures

        # Method 2: Look for Test Summary table
        # Multiple possible formats:
        # All pass:     "Test Summary: | Pass  Total  Time"
        #               "My Tests     |    3      3  0.5s"
        # Some fail:    "Test Summary: | Pass  Fail  Total  Time"
        #               "My Tests     |    2     1      3  0.5s"
        # All error:    "Test Summary: | Error  Total  Time"
        #               "My Tests     |     3      3  0.9s"
        # Mixed:        "Test Summary: | Pass  Fail  Error  Total  Time"
        #               "My Tests     |    1     1      1      3  0.5s"
        summary_lines = output.split("\n")
        for i, line in enumerate(summary_lines):
            if "Test Summary:" in line and i + 1 < len(summary_lines):
                header_line = line
                next_line = summary_lines[i + 1]

                # Determine which columns are present
                has_pass = "Pass" in header_line
                has_fail = "Fail" in header_line
                has_error = "Error" in header_line

                # Extract all numbers from the line
                all_numbers = re.findall(r"\d+", next_line)
                if not all_numbers:
                    continue

                # Last number is always Total, second to last is Time (skip it)
                # Extract based on which columns exist
                if has_pass and has_fail and has_error:
                    # Pass  Fail  Error  Total  Time
                    if len(all_numbers) >= 5:
                        passed = int(all_numbers[0])
                        failed = int(all_numbers[1]) + int(
                            all_numbers[2]
                        )  # Fail + Error
                        return passed, failed
                elif has_pass and has_fail:
                    # Pass  Fail  Total  Time
                    if len(all_numbers) >= 4:
                        passed = int(all_numbers[0])
                        failed = int(all_numbers[1])
                        return passed, failed
                elif has_pass and has_error:
                    # Pass  Error  Total  Time
                    if len(all_numbers) >= 4:
                        passed = int(all_numbers[0])
                        failed = int(all_numbers[1])  # Treat errors as failures
                        return passed, failed
                elif has_fail and has_error:
                    # Fail  Error  Total  Time (no passes)
                    if len(all_numbers) >= 4:
                        passed = 0
                        failed = int(all_numbers[0]) + int(all_numbers[1])
                        return passed, failed
                elif has_pass:
                    # Pass  Total  Time (no failures/errors)
                    if len(all_numbers) >= 3:
                        passed = int(all_numbers[0])
                        failed = 0
                        return passed, failed
                elif has_error:
                    # Error  Total  Time (all errors, no passes)
                    if len(all_numbers) >= 3:
                        passed = 0
                        failed = int(all_numbers[0])  # Treat all errors as failures
                        return passed, failed
                elif has_fail:
                    # Fail  Total  Time (all failures, no passes)
                    if len(all_numbers) >= 3:
                        passed = 0
                        failed = int(all_numbers[0])
                        return passed, failed

        return passed, failed

    def _calculate_reward(
        self, code_compiles: bool, tests_passed: int, tests_failed: int
    ) -> float:
        """
        Normalized percentage-based reward for Julia GRPO.
        Returns rewards in [-1, 1.5] range for comparability across problems.
        """
        if not code_compiles:
            return -1.0

        total_tests = tests_passed + tests_failed
        if total_tests == 0:
            return 0.0  # No signal when no tests run

        pass_rate = tests_passed / total_tests

        # Scaled 0-1 with bonus for perfection
        if pass_rate == 1.0:
            return 1.5  # Bonus for passing all tests
        return pass_rate

    def _apply_transform(self, observation: JuliaObservation) -> JuliaObservation:
        """Apply safety and quality transforms to observation."""
        if self.transform:
            observation = self.transform(observation)
        return observation

    @property
    def state(self) -> JuliaState:
        """Return current environment state."""
        return self._state
