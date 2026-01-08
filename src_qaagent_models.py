diff --git a/src/qa_agent/models.py b/src/qa_agent/models.py
new file mode 100644
index 0000000000000000000000000000000000000000..f89767bb4a295f464e2b414d599ee7daacb2a5ca
--- /dev/null
+++ b/src/qa_agent/models.py
@@ -0,0 +1,404 @@
+from __future__ import annotations
+
+from dataclasses import dataclass, field
+from datetime import datetime
+from typing import List, Optional
+
+
+@dataclass
+class CaptureEvent:
+    """A raw DOM event captured from instrumentation."""
+
+    type: str
+    selector: str
+    value: Optional[str] = None
+    text: Optional[str] = None
+
+    def to_step(self) -> "Step":
+        note = self.text or self.value
+        action = self.type
+        if action == "change":
+            action = "fill"
+        elif action == "navigation":
+            action = "visit"
+        return Step(action=action, target=self.selector, note=note)
+
+
+@dataclass
+class Step:
+    """A single user interaction within a flow."""
+
+    action: str
+    target: str
+    note: str | None = None
+
+    def to_dict(self) -> dict:
+        return {"action": self.action, "target": self.target, "note": self.note}
+
+    @classmethod
+    def from_dict(cls, data: dict) -> "Step":
+        return cls(action=data["action"], target=data["target"], note=data.get("note"))
+
+
+@dataclass
+class Flow:
+    """A user journey the agent should protect."""
+
+    name: str
+    path: str
+    goal: str
+    priority: str = "medium"
+    owner: Optional[str] = None
+    tags: List[str] = field(default_factory=list)
+    steps: List[Step] = field(default_factory=list)
+
+    def to_dict(self) -> dict:
+        return {
+            "name": self.name,
+            "path": self.path,
+            "goal": self.goal,
+            "priority": self.priority,
+            "owner": self.owner,
+            "tags": self.tags,
+            "steps": [step.to_dict() for step in self.steps],
+        }
+
+    @classmethod
+    def from_dict(cls, data: dict) -> "Flow":
+        return cls(
+            name=data["name"],
+            path=data["path"],
+            goal=data["goal"],
+            priority=data.get("priority", "medium"),
+            owner=data.get("owner"),
+            tags=data.get("tags", []),
+            steps=[Step.from_dict(item) for item in data.get("steps", [])],
+        )
+
+
+@dataclass
+class TestCase:
+    """A generated test case derived from a flow."""
+
+    flow_name: str
+    title: str
+    estimated_runtime_ms: int
+    selectors: List[str] = field(default_factory=list)
+
+    def to_dict(self) -> dict:
+        return {
+            "flow_name": self.flow_name,
+            "title": self.title,
+            "estimated_runtime_ms": self.estimated_runtime_ms,
+            "selectors": self.selectors,
+        }
+
+
+@dataclass
+class RunPlan:
+    """Metadata for a scheduled run."""
+
+    generated_at: datetime
+    environment: str
+    tests: List[TestCase]
+
+    def to_dict(self) -> dict:
+        return {
+            "generated_at": self.generated_at.isoformat(),
+            "environment": self.environment,
+            "tests": [test.to_dict() for test in self.tests],
+        }
+
+
+@dataclass
+class Environment:
+    """A deployment target the agent can run against."""
+
+    name: str
+    base_url: str
+    auth_mode: str = "anonymous"
+    framework: Optional[str] = None
+    notes: Optional[str] = None
+    tags: List[str] = field(default_factory=list)
+
+    def to_dict(self) -> dict:
+        return {
+            "name": self.name,
+            "base_url": self.base_url,
+            "auth_mode": self.auth_mode,
+            "framework": self.framework,
+            "notes": self.notes,
+            "tags": self.tags,
+        }
+
+    @classmethod
+    def from_dict(cls, data: dict) -> "Environment":
+        return cls(
+            name=data["name"],
+            base_url=data["base_url"],
+            auth_mode=data.get("auth_mode", "anonymous"),
+            framework=data.get("framework"),
+            notes=data.get("notes"),
+            tags=data.get("tags", []),
+        )
+
+
+@dataclass
+class SecretConfig:
+    """A secret reference or token used for CI and notifications."""
+
+    name: str
+    kind: str  # env | file | vault
+    value: str
+    scope: Optional[str] = None
+    notes: Optional[str] = None
+
+    def to_dict(self) -> dict:
+        return {
+            "name": self.name,
+            "kind": self.kind,
+            "value": self.value,
+            "scope": self.scope,
+            "notes": self.notes,
+        }
+
+    @classmethod
+    def from_dict(cls, data: dict) -> "SecretConfig":
+        return cls(
+            name=data["name"],
+            kind=data.get("kind", "env"),
+            value=data.get("value", ""),
+            scope=data.get("scope"),
+            notes=data.get("notes"),
+        )
+
+
+@dataclass
+class EnvironmentOverride:
+    """Per-environment configuration overrides (feature flags, headers, etc.)."""
+
+    environment: str
+    key: str
+    value: str
+    notes: Optional[str] = None
+
+    def to_dict(self) -> dict:
+        return {"environment": self.environment, "key": self.key, "value": self.value, "notes": self.notes}
+
+    @classmethod
+    def from_dict(cls, data: dict) -> "EnvironmentOverride":
+        return cls(
+            environment=data["environment"],
+            key=data["key"],
+            value=data.get("value", ""),
+            notes=data.get("notes"),
+        )
+
+
+@dataclass
+class NotificationChannel:
+    """Where to send failure and health summaries."""
+
+    kind: str  # slack | email
+    target: str
+    severity: str = "normal"
+    notes: Optional[str] = None
+
+    def to_dict(self) -> dict:
+        return {
+            "kind": self.kind,
+            "target": self.target,
+            "severity": self.severity,
+            "notes": self.notes,
+        }
+
+    @classmethod
+    def from_dict(cls, data: dict) -> "NotificationChannel":
+        return cls(
+            kind=data.get("kind", "slack"),
+            target=data.get("target", ""),
+            severity=data.get("severity", "normal"),
+            notes=data.get("notes"),
+        )
+
+
+@dataclass
+class RunbookEntry:
+    """Operational guardrails and playbooks."""
+
+    title: str
+    instructions: str
+    guardrail: Optional[str] = None
+
+    def to_dict(self) -> dict:
+        return {
+            "title": self.title,
+            "instructions": self.instructions,
+            "guardrail": self.guardrail,
+        }
+
+    @classmethod
+    def from_dict(cls, data: dict) -> "RunbookEntry":
+        return cls(
+            title=data["title"],
+            instructions=data.get("instructions", ""),
+            guardrail=data.get("guardrail"),
+        )
+
+
+@dataclass
+class SuccessMetric:
+    """Success criteria we track during week one and beyond."""
+
+    name: str
+    target: str
+    measurement: str = "pending"
+    owner: Optional[str] = None
+    notes: Optional[str] = None
+
+    def to_dict(self) -> dict:
+        return {
+            "name": self.name,
+            "target": self.target,
+            "measurement": self.measurement,
+            "owner": self.owner,
+            "notes": self.notes,
+        }
+
+    @classmethod
+    def from_dict(cls, data: dict) -> "SuccessMetric":
+        return cls(
+            name=data["name"],
+            target=data["target"],
+            measurement=data.get("measurement", "pending"),
+            owner=data.get("owner"),
+            notes=data.get("notes"),
+        )
+
+
+@dataclass
+class Fixture:
+    """Deterministic fixtures to keep captured flows replayable."""
+
+    name: str
+    kind: str
+    value: str
+    notes: Optional[str] = None
+
+    def to_dict(self) -> dict:
+        return {"name": self.name, "kind": self.kind, "value": self.value, "notes": self.notes}
+
+    @classmethod
+    def from_dict(cls, data: dict) -> "Fixture":
+        return cls(name=data["name"], kind=data["kind"], value=data["value"], notes=data.get("notes"))
+
+
+@dataclass
+class RunArtifact:
+    """A record of a generated artifact from a run (log, screenshot bundle, etc.)."""
+
+    run_id: str
+    created_at: datetime
+    environment: str
+    status: str
+    log_path: str
+    notes: Optional[str] = None
+    attachments: List[str] = field(default_factory=list)
+
+    def to_dict(self) -> dict:
+        return {
+            "run_id": self.run_id,
+            "created_at": self.created_at.isoformat(),
+            "environment": self.environment,
+            "status": self.status,
+            "log_path": self.log_path,
+            "notes": self.notes,
+            "attachments": self.attachments,
+        }
+
+    @classmethod
+    def from_dict(cls, data: dict) -> "RunArtifact":
+        return cls(
+            run_id=data["run_id"],
+            created_at=datetime.fromisoformat(data["created_at"]),
+            environment=data.get("environment", "local"),
+            status=data.get("status", "unknown"),
+            log_path=data.get("log_path", ""),
+            notes=data.get("notes"),
+            attachments=data.get("attachments", []),
+        )
+
+
+@dataclass
+class FlowRunResult:
+    """Outcome for a single flow within a headless test run."""
+
+    flow_name: str
+    run_id: str
+    created_at: datetime
+    environment: str
+    status: str  # passed | failed | flaky
+    attempts: int = 1
+    failure_notes: Optional[str] = None
+    unstable_selectors: List[str] = field(default_factory=list)
+    screenshot: Optional[str] = None
+
+    def to_dict(self) -> dict:
+        return {
+            "flow_name": self.flow_name,
+            "run_id": self.run_id,
+            "created_at": self.created_at.isoformat(),
+            "environment": self.environment,
+            "status": self.status,
+            "attempts": self.attempts,
+            "failure_notes": self.failure_notes,
+            "unstable_selectors": self.unstable_selectors,
+            "screenshot": self.screenshot,
+        }
+
+    @classmethod
+    def from_dict(cls, data: dict) -> "FlowRunResult":
+        return cls(
+            flow_name=data["flow_name"],
+            run_id=data["run_id"],
+            created_at=datetime.fromisoformat(data["created_at"]),
+            environment=data.get("environment", "local"),
+            status=data.get("status", "unknown"),
+            attempts=data.get("attempts", 1),
+            failure_notes=data.get("failure_notes"),
+            unstable_selectors=data.get("unstable_selectors", []),
+            screenshot=data.get("screenshot"),
+        )
+
+
+@dataclass
+class IssueDraft:
+    """A lightweight issue draft generated from a failed or flaky run."""
+
+    flow_name: str
+    run_id: str
+    environment: str
+    title: str
+    body: str
+    created_at: datetime
+
+    def to_dict(self) -> dict:
+        return {
+            "flow_name": self.flow_name,
+            "run_id": self.run_id,
+            "environment": self.environment,
+            "title": self.title,
+            "body": self.body,
+            "created_at": self.created_at.isoformat(),
+        }
+
+    @classmethod
+    def from_dict(cls, data: dict) -> "IssueDraft":
+        return cls(
+            flow_name=data["flow_name"],
+            run_id=data["run_id"],
+            environment=data.get("environment", "local"),
+            title=data["title"],
+            body=data.get("body", ""),
+            created_at=datetime.fromisoformat(data["created_at"]),
+        )
