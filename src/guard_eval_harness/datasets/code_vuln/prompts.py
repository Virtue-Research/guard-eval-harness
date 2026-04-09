"""Prompt templates and response parsing for code vulnerability
detection benchmarks.

Adapted from the VulnLLM-R evaluation framework
(arXiv:2512.07533).
"""

from __future__ import annotations

import re
from typing import Any


# --- System prompts (model-specific, matching VulnLLM-R) ---

# VulnLLM-R selects system prompts based on model name
# patterns.  See VulnLLM-R/vulscan/test/test.py lines
# 225-249 for the original selection logic.

SYSTEM_PROMPTS: dict[str, str] = {
    "qwen": (
        "You are a helpful and harmless assistant. "
        "You are Qwen developed by Alibaba. "
        "You should think step-by-step."
    ),
    "deepseek": (
        "You are a helpful and harmless code assistant. "
        "You should think step-by-step."
    ),
    "default": ("You are a helpful assistant. You should think step-by-step."),
    "simplescaling": (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    ),
    "sft": (
        "Your role as an assistant involves thoroughly "
        "exploring questions through a systematic long "
        "thinking process before providing the final "
        "precise and accurate solutions. This requires "
        "engaging in a comprehensive cycle of analysis, "
        "summarizing, exploration, reassessment, "
        "reflection, backtracing, and iteration to "
        "develop well-considered thinking process. "
        "Please structure your response into two main "
        "sections: Thought and Solution. "
        "In the Thought section, detail your reasoning "
        "process using the specified format: "
        "<|begin_of_thought|> {thought with steps "
        "separated with '\\n\\n'} <|end_of_thought|> "
        "Each step should include detailed considerations "
        "such as analisying questions, summarizing "
        "relevant findings, brainstorming new ideas, "
        "verifying the accuracy of the current steps, "
        "refining any errors, and revisiting previous "
        "steps. "
        "In the Solution section, based on various "
        "attempts, explorations, and reflections from "
        "the Thought section, systematically present "
        "the final solution that you deem correct. The "
        "solution should remain a logical, accurate, "
        "concise expression style and detail necessary "
        "step needed to reach the conclusion, formatted "
        "as follows: <|begin_of_solution|> {final "
        "formatted, precise, and clear solution} "
        "<|end_of_solution|> "
        "Now, try to solve the following question "
        "through the above guidelines:"
    ),
}

# Default fallback
CODE_VULN_SYSTEM_PROMPT = SYSTEM_PROMPTS["default"]


def resolve_system_prompt(
    model_name: str | None = None,
    system_prompt_key: str | None = None,
) -> str | None:
    """Select the system prompt matching VulnLLM-R's logic.

    If ``system_prompt_key`` is given, it is used directly
    as a key into ``SYSTEM_PROMPTS``.  Otherwise the model
    name is pattern-matched against VulnLLM-R's selection
    rules.
    """
    if system_prompt_key:
        return SYSTEM_PROMPTS.get(system_prompt_key, SYSTEM_PROMPTS["default"])
    if not model_name:
        return SYSTEM_PROMPTS["default"]

    lowered = model_name.lower()
    # Replicate VulnLLM-R's test.py lines 228-248.
    # All checks use lowered for case-insensitive matching
    # so identifiers like "qwen/qwen2.5-7b-instruct" work.
    if "deepseek-r1-distill-qwen" in lowered or "deepcoder" in lowered:
        return SYSTEM_PROMPTS["qwen"]
    if "sky-t1" in lowered:
        return SYSTEM_PROMPTS["sft"]
    if "simplescaling" in lowered:
        return SYSTEM_PROMPTS["simplescaling"]
    if "qwq" in lowered or "qwen" in lowered:
        return SYSTEM_PROMPTS["qwen"]
    if "deepseek-reasoner" in lowered:
        return SYSTEM_PROMPTS["deepseek"]
    if "google" in lowered or "gemma" in lowered:
        return SYSTEM_PROMPTS["default"]
    # GPT, Claude, and other API models use no system
    # prompt in VulnLLM-R (system_prompt=None → the model's
    # own default).
    if "gpt" in lowered or "claude" in lowered:
        return None
    return SYSTEM_PROMPTS["default"]


# --- Chain-of-thought reasoning instructions ---

COT_INSTRUCTIONS = """\
Please think step by step and follow the following procedure.
Step 1: understand the code and identify key instructions \
and program states;
Step 2: come up with the constraints on the identified \
instructions or states to decide if the code is vulnerable;
Step 3: Predict the actual program states and decide if it \
follows the constraints;
Step 4: Tell whether the code is vulnerable based on the \
analysis above
"""

_REASONING_SUFFIX = "You should STRICTLY structure your response as follows:"

# --- CWE short descriptions (for policy section) ---
# Sourced from MITRE CWE database; covers CWEs used in the
# VulnLLM-R test datasets.

CWE_DESCRIPTIONS: dict[str, str] = {
    "CWE-15": "External Control of System or Configuration Setting",
    "CWE-20": "Improper Input Validation",
    "CWE-22": "Improper Limitation of a Pathname to a "
    "Restricted Directory ('Path Traversal')",
    "CWE-23": "Relative Path Traversal",
    "CWE-59": "Improper Link Resolution Before File Access ('Link Following')",
    "CWE-74": "Improper Neutralization of Special Elements "
    "in Output Used by a Downstream Component "
    "('Injection')",
    "CWE-77": "Improper Neutralization of Special Elements "
    "used in a Command ('Command Injection')",
    "CWE-78": "Improper Neutralization of Special Elements "
    "used in an OS Command ('OS Command Injection')",
    "CWE-79": "Improper Neutralization of Input During Web "
    "Page Generation ('Cross-site Scripting')",
    "CWE-89": "Improper Neutralization of Special Elements "
    "used in an SQL Command ('SQL Injection')",
    "CWE-90": "Improper Neutralization of Special Elements "
    "used in an LDAP Query ('LDAP Injection')",
    "CWE-94": "Improper Control of Generation of Code ('Code Injection')",
    "CWE-95": "Improper Neutralization of Directives in "
    "Dynamically Evaluated Code ('Eval Injection')",
    "CWE-119": "Improper Restriction of Operations within "
    "the Bounds of a Memory Buffer",
    "CWE-120": "Buffer Copy without Checking Size of Input "
    "('Classic Buffer Overflow')",
    "CWE-121": "Stack-based Buffer Overflow",
    "CWE-122": "Heap-based Buffer Overflow",
    "CWE-123": "Write-what-where Condition",
    "CWE-124": "Buffer Underwrite ('Buffer Underflow')",
    "CWE-125": "Out-of-bounds Read",
    "CWE-134": "Use of Externally-Controlled Format String",
    "CWE-176": "Improper Handling of Unicode Encoding",
    "CWE-179": "Incorrect Behavior Order: Early Validation",
    "CWE-190": "Integer Overflow or Wraparound",
    "CWE-191": "Integer Underflow (Wrap or Wraparound)",
    "CWE-193": "Off-by-one Error",
    "CWE-200": "Exposure of Sensitive Information to an Unauthorized Actor",
    "CWE-212": "Improper Removal of Sensitive Information "
    "Before Storage or Transfer",
    "CWE-242": "Use of Inherently Dangerous Function",
    "CWE-252": "Unchecked Return Value",
    "CWE-269": "Improper Privilege Management",
    "CWE-276": "Incorrect Default Permissions",
    "CWE-281": "Improper Preservation of Permissions",
    "CWE-284": "Improper Access Control",
    "CWE-287": "Improper Authentication",
    "CWE-288": "Authentication Bypass Using an Alternate Path or Channel",
    "CWE-295": "Improper Certificate Validation",
    "CWE-307": "Improper Restriction of Excessive Authentication Attempts",
    "CWE-319": "Cleartext Transmission of Sensitive Information",
    "CWE-327": "Use of a Broken or Risky Cryptographic Algorithm",
    "CWE-338": "Use of Cryptographically Weak Pseudo-Random "
    "Number Generator (PRNG)",
    "CWE-345": "Insufficient Verification of Data Authenticity",
    "CWE-352": "Cross-Site Request Forgery (CSRF)",
    "CWE-354": "Improper Validation of Integrity Check Value",
    "CWE-362": "Concurrent Execution using Shared Resource "
    "with Improper Synchronization ('Race Condition')",
    "CWE-367": "Time-of-check Time-of-use (TOCTOU) Race Condition",
    "CWE-369": "Divide By Zero",
    "CWE-400": "Uncontrolled Resource Consumption",
    "CWE-401": "Missing Release of Memory after Effective Lifetime",
    "CWE-415": "Double Free",
    "CWE-416": "Use After Free",
    "CWE-426": "Untrusted Search Path",
    "CWE-434": "Unrestricted Upload of File with Dangerous Type",
    "CWE-444": "Inconsistent Interpretation of HTTP "
    "Requests ('HTTP Request/Response Smuggling')",
    "CWE-457": "Use of Uninitialized Variable",
    "CWE-476": "NULL Pointer Dereference",
    "CWE-502": "Deserialization of Untrusted Data",
    "CWE-506": "Embedded Malicious Code",
    "CWE-522": "Insufficiently Protected Credentials",
    "CWE-526": "Cleartext Storage of Sensitive Information "
    "in an Environment Variable",
    "CWE-552": "Files or Directories Accessible to External Parties",
    "CWE-590": "Free of Memory not on the Heap",
    "CWE-601": "URL Redirection to Untrusted Site ('Open Redirect')",
    "CWE-611": "Improper Restriction of XML External Entity Reference",
    "CWE-617": "Reachable Assertion",
    "CWE-665": "Improper Initialization",
    "CWE-667": "Improper Locking",
    "CWE-668": "Exposure of Resource to Wrong Sphere",
    "CWE-672": "Operation on a Resource after Expiration or Release",
    "CWE-681": "Incorrect Conversion between Numeric Types",
    "CWE-703": "Improper Check or Handling of Exceptional Conditions",
    "CWE-704": "Incorrect Type Conversion or Cast",
    "CWE-732": "Incorrect Permission Assignment for Critical Resource",
    "CWE-754": "Improper Check for Unusual or Exceptional Conditions",
    "CWE-758": "Reliance on Undefined, Unspecified, or "
    "Implementation-Defined Behavior",
    "CWE-761": "Free of Pointer not at Start of Buffer",
    "CWE-770": "Allocation of Resources Without Limits or Throttling",
    "CWE-772": "Missing Release of Resource after Effective Lifetime",
    "CWE-775": "Missing Release of File Descriptor or "
    "Handle after Effective Lifetime",
    "CWE-787": "Out-of-bounds Write",
    "CWE-798": "Use of Hard-coded Credentials",
    "CWE-824": "Access of Uninitialized Pointer",
    "CWE-834": "Excessive Iteration",
    "CWE-835": "Loop with Unreachable Exit Condition ('Infinite Loop')",
    "CWE-843": "Access of Resource Using Incompatible Type ('Type Confusion')",
    "CWE-862": "Missing Authorization",
    "CWE-863": "Incorrect Authorization",
    "CWE-908": "Use of Uninitialized Resource",
    "CWE-909": "Missing Initialization of Resource",
    "CWE-915": "Improperly Controlled Modification of "
    "Dynamically-Determined Object Attributes",
    "CWE-918": "Server-Side Request Forgery (SSRF)",
    "CWE-924": "Improper Enforcement of Message Integrity "
    "During Transmission in a Communication Channel",
    "CWE-1333": "Inefficient Regular Expression Complexity",
}

# --- Per-CWE analysis hints ---

CWE_CONSTRAINTS: dict[str, str] = {
    "CWE-22": (
        "Confirm that the code includes checks for absolute "
        "paths, using security flags, and verify that the code "
        "correctly identifies and handles absolute paths by "
        "setting errors and returning failure codes when such "
        "paths are detected."
    ),
    "CWE-78": (
        "Identify potential injection of operating system "
        "commands through user-controlled input that is passed "
        "to system calls without proper sanitization."
    ),
    "CWE-79": (
        "Look for user input that is included in web output "
        "without proper encoding or escaping, allowing "
        "cross-site scripting."
    ),
    "CWE-89": (
        "Look for SQL queries that incorporate user input "
        "without parameterization or proper escaping."
    ),
    "CWE-94": (
        "Identify code that evaluates or executes "
        "user-controlled strings as code."
    ),
    "CWE-119": (
        "Check for buffer operations that may access memory "
        "outside the intended boundaries."
    ),
    "CWE-120": (
        "Look for buffer copy operations that do not check "
        "the size of input, potentially overflowing the "
        "destination buffer."
    ),
    "CWE-121": (
        "Ensure that both lower and upper bounds are checked "
        "before using an index to access an array, and "
        "recognize the addition of a condition of data size "
        "as a fix."
    ),
    "CWE-125": (
        "Ensure that any operation involving buffers or arrays "
        "checks the boundaries before accessing elements. Look "
        "for conditions where the code might access elements "
        "beyond the allocated memory."
    ),
    "CWE-134": (
        "Ensure that format strings are fixed and not "
        "influenced by external input."
    ),
    "CWE-190": (
        "Ensure that input is validated before being used in "
        "arithmetic operations. This includes checking that "
        "the input is within a safe range to prevent overflow."
    ),
    "CWE-191": (
        "Look for arithmetic operations, especially decrement "
        "operations that are performed on variables that can "
        "potentially hold minimum integer values."
    ),
    "CWE-200": (
        "Check whether sensitive information such as system "
        "data, credentials, or internal state is exposed to "
        "unauthorized actors."
    ),
    "CWE-327": (
        "Identify the use of strong, well-regarded "
        "cryptographic algorithms, and understand that the use "
        "of such algorithms mitigates vulnerabilities by "
        "providing adequate cryptographic strength."
    ),
    "CWE-367": (
        "Identify that the benign code does not perform a "
        "separate check before using the resource, and "
        "understand that by directly attempting to use the "
        "resource, the code avoids the window of opportunity "
        "for a race condition."
    ),
    "CWE-369": (
        "Verify that the code includes checks to validate "
        "inputs before they are used in division operations. "
        "This includes ensuring the divisor is not zero or "
        "close to zero."
    ),
    "CWE-400": (
        "Ensure that input values are validated and constrained "
        "within safe limits before being used to control "
        "resource consumption."
    ),
    "CWE-416": (
        "Ensure that memory allocation and deallocation are "
        "handled correctly, with no operations on pointers "
        "after deallocation, and verify that any deallocated "
        "pointers are not used in subsequent operations."
    ),
    "CWE-457": (
        "Ensure that all elements of an array or structure are "
        "initialized before any use. This can be achieved by "
        "initializing the entire array in a loop before any "
        "other operations."
    ),
    "CWE-476": (
        "Ensure that pointers are validated before they are "
        "dereferenced. This includes checking if a pointer is "
        "NULL and handling such cases appropriately."
    ),
    "CWE-502": (
        "Look for deserialization of data from untrusted "
        "sources without proper validation or integrity checks."
    ),
    "CWE-526": (
        "Note the absence of conditional logic that prevents "
        "the exposure of environment variables. If the code "
        "always executes the output of an environment variable "
        "without any checks, it is likely vulnerable."
    ),
    "CWE-758": (
        "Identify code patterns where objects are used without "
        "proper initialization. Specifically, look for "
        "instances where a pointer is dereferenced to access "
        "or copy data from an uninitialized object."
    ),
    "CWE-761": (
        "Prefer index-based traversal over pointer arithmetic "
        "when iterating through a buffer. This ensures that "
        "the original pointer remains unchanged and can be "
        "safely freed."
    ),
    "CWE-787": (
        "Identify code sections where data is written to "
        "buffers. Pay attention to calculations involving "
        "buffer sizes and offsets. Look for operations that "
        "modify buffer pointers or indices, especially in "
        "loops or conditional statements."
    ),
    "CWE-843": (
        "Ensure that the type of data being accessed is "
        "consistent with the type of the variable it is "
        "pointing to."
    ),
    "CWE-918": (
        "Check whether user-controlled URLs or URI components "
        "are used in server-side requests without validation, "
        "enabling server-side request forgery."
    ),
}


# --- Prompt builder ---

_POLICY_PREFIX = (
    "You should only focus on checking and reasoning if the "
    "code contains one of the following CWEs, or other CWE "
    "if you think it is more relevant:"
)

_USER_PROMPT_TEMPLATE = """\
You are an advanced vulnerability detection model. \
Your task is to check if a specific vulnerability exists \
in a given piece of code. \
You need to output whether the code is vulnerable and the \
type of vulnerability present with CWE id (CWE-xx).

## You are given the following code snippet:
```
{code}
```

{cwe_info}

{reasoning}

## Final Answer
#judge: <yes/no>
#type: <vulnerability type>

## Additional Constraint:
- If `#judge: yes`, then `#type:` **must contain exactly \
one CWE**.
- If `#judge: yes`, the model must output **only the most \
probable CWE** related to the given code snippet.
{additional_constraint}

## Example
- If the code is vulnerable to a CWE-79, you should \
finally output:
## Final Answer
#judge: yes
#type: CWE-79

- If the code does not contain vulnerabilities related to \
the given CWE, you should finally output:
## Final Answer
#judge: no
#type: N/A
"""

_LONG_CONTEXT_PROMPT_TEMPLATE = """\
You are an advanced vulnerability detection model. \
Your task is to check if a specific vulnerability exists \
in a given piece of code. \
The code may contain a long context, which is the stack \
trace of the function. \
They are separated by "// context" and "// target function".\
 \
You need to output whether the target function is \
vulnerable and the type of vulnerability present with \
CWE id (CWE-xx).

## You are given the following code snippet:
```
{code}
```

{cwe_info}

{reasoning}

## Final Answer
#judge: <yes/no>
#type: <vulnerability type>

## Additional Constraint:
- If `#judge: yes`, then `#type:` **must contain exactly \
one CWE**.
- If `#judge: yes`, the model must output **only the most \
probable CWE** related to the given code snippet.
{additional_constraint}

## Example
- If the target function is vulnerable to a CWE-79, you \
should finally output:
## Final Answer
#judge: yes
#type: CWE-79

- If the target function does not contain vulnerabilities \
related to the given CWE, you should finally output:
## Final Answer
#judge: no
#type: N/A
"""


def _build_cwe_policy(cwe_list: list[str]) -> str:
    """Build CWE policy section with descriptions.

    Matches VulnLLM-R's ``create_reasoning_test_sample``
    which appends ``get_cwe_info()`` descriptions to each
    CWE in the policy string.
    """
    lines = [_POLICY_PREFIX]
    for cwe in cwe_list:
        desc = CWE_DESCRIPTIONS.get(cwe)
        if desc:
            lines.append(f"- {cwe}: {desc}")
        else:
            lines.append(f"- {cwe}")
    return "\n".join(lines)


def build_code_vuln_prompt(
    *,
    code: str,
    cwe_list: list[str] | None = None,
    use_cot: bool = True,
    use_policy: bool = True,
    use_cwe_constraint: bool = False,
    has_stack_trace: bool = False,
) -> str:
    """Build the user-facing prompt for code vulnerability
    detection.

    Parameters
    ----------
    code:
        The source code snippet to analyse.
    cwe_list:
        CWE identifiers relevant to this sample. Used for
        the policy/focus section of the prompt.
    use_cot:
        Whether to include chain-of-thought reasoning
        instructions.
    use_policy:
        Whether to include the CWE policy/focus section.
    use_cwe_constraint:
        Whether to include per-CWE analysis hints.
    has_stack_trace:
        Whether the code contains long-context stack trace.
        Uses a specialised prompt template when True.
    """
    # CWE info section (with descriptions, matching
    # VulnLLM-R's policy format)
    cwe_info = ""
    if use_policy and cwe_list:
        cwe_info = _build_cwe_policy(cwe_list)

    # Reasoning instructions + structured output suffix
    reasoning_parts: list[str] = []
    if use_cot:
        reasoning_parts.append(COT_INSTRUCTIONS)
    reasoning_parts.append(_REASONING_SUFFIX)
    reasoning = "\n".join(reasoning_parts)

    # Additional per-CWE constraints
    additional_constraint = ""
    if use_cwe_constraint and cwe_list:
        constraints = []
        for cwe in cwe_list:
            if cwe in CWE_CONSTRAINTS:
                constraints.append(f"- {cwe}: {CWE_CONSTRAINTS[cwe]}")
        if constraints:
            additional_constraint = (
                "\n## Additional Analysis Hints:\n" + "\n".join(constraints)
            )

    template = (
        _LONG_CONTEXT_PROMPT_TEMPLATE
        if has_stack_trace
        else _USER_PROMPT_TEMPLATE
    )
    return template.format(
        code=code,
        cwe_info=cwe_info,
        reasoning=reasoning,
        additional_constraint=additional_constraint,
    )


# --- Response parsing ---

_JUDGE_PATTERN = re.compile(r"#judge:\s*(yes|no)", re.IGNORECASE)
_TYPE_LINE_PATTERN = re.compile(r"#type:\s*(.+)", re.IGNORECASE)
_CWE_TOKEN_PATTERN = re.compile(r"CWE-\d+", re.IGNORECASE)


def parse_judge_response(
    text: str,
) -> dict[str, Any]:
    """Extract structured fields from the **last**
    ``#judge:``/``#type:`` fields in a response.

    Uses the last match so that reasoning/examples earlier
    in the output do not override the final answer.

    Returns a dict with keys:

    - ``vulnerable`` (bool | None): True if ``#judge: yes``
    - ``cwe`` (str | None): Predicted CWE or None
    - ``unsafe_score`` (float | None): 1.0 for yes, 0.0 for
      no, None if not found
    """
    judge_matches = list(_JUDGE_PATTERN.finditer(text))
    type_matches = list(_TYPE_LINE_PATTERN.finditer(text))

    vulnerable: bool | None = None
    unsafe_score: float | None = None
    cwe: str | None = None

    if judge_matches:
        answer = judge_matches[-1].group(1).lower()
        vulnerable = answer == "yes"
        unsafe_score = 1.0 if vulnerable else 0.0

    if type_matches:
        type_value = type_matches[-1].group(1).strip()
        if type_value.upper() != "N/A":
            cwes = _CWE_TOKEN_PATTERN.findall(type_value)
            if len(cwes) == 1:
                cwe = cwes[0].upper()

    return {
        "vulnerable": vulnerable,
        "cwe": cwe,
        "unsafe_score": unsafe_score,
    }
