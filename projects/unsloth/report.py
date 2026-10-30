"""Always report failures, including installation failures before Python starts."""
import json
import os
from pathlib import Path

path = Path("output/result.json")
result = json.loads(path.read_text()) if path.exists() else {
    "status": "failed", "stage": "environment_setup", "completed": []
}
result.update(job_status=os.environ["JOB_STATUS"],
              workflow_sha=os.environ["GITHUB_SHA"],
              unsloth_ref=os.environ["UNSLOTH_REF"], zoo_ref=os.environ["ZOO_REF"])
path.write_text(json.dumps(result, indent=2))
with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
    summary.write("## Unsloth Ascend trial\n\n```json\n" +
                  json.dumps(result, indent=2) + "\n```\n")
