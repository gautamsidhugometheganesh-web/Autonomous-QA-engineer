diff --git a/src/qa_agent/storage.py b/src/qa_agent/storage.py
new file mode 100644
index 0000000000000000000000000000000000000000..9ff4f19802756af677ea837b4074bb9abbe07ded
--- /dev/null
+++ b/src/qa_agent/storage.py
@@ -0,0 +1,263 @@
+from __future__ import annotations
+
+import json
+from dataclasses import asdict
+from pathlib import Path
+from typing import List, Optional
+
+from .models import (
+    Environment,
+    EnvironmentOverride,
+    Fixture,
+    Flow,
+    FlowRunResult,
+    IssueDraft,
+    NotificationChannel,
+    RunArtifact,
+    RunbookEntry,
+    SecretConfig,
+    SuccessMetric,
+)
+
+
+class Workspace:
+    """Filesystem-backed store for flows and metadata."""
+
+    def __init__(self, root: Path) -> None:
+        self.root = root
+        self.metadata_file = self.root / "metadata.json"
+        self.flows_file = self.root / "flows.json"
+        self.artifacts_file = self.root / "artifacts.json"
+        self.run_results_file = self.root / "run_results.json"
+        self.issues_file = self.root / "issues.json"
+        self.flake_report_file = self.root / "flake_report.json"
+        self.regeneration_queue_file = self.root / "regeneration_queue.json"
+        self.validation_report_file = self.root / "validation_report.json"
+        self.runbooks_file = self.root / "runbooks.json"
+
+    @property
+    def exists(self) -> bool:
+        return self.root.exists()
+
+    def init(self, project_name: str, base_url: str) -> None:
+        self.root.mkdir(parents=True, exist_ok=True)
+        if not self.metadata_file.exists():
+            self.metadata_file.write_text(
+                json.dumps(
+                    {
+                        "project": project_name,
+                        "base_url": base_url,
+                        "stack": {"framework": None, "auth": None},
+                        "demo_app": None,
+                        "environments": [],
+                        "success_metrics": [],
+                        "fixtures": [],
+                        "secrets": [],
+                        "notification_channels": [],
+                        "overrides": [],
+                    },
+                    indent=2,
+                )
+            )
+        if not self.flows_file.exists():
+            self.flows_file.write_text(json.dumps([], indent=2))
+        if not self.artifacts_file.exists():
+            self.artifacts_file.write_text(json.dumps([], indent=2))
+        if not self.run_results_file.exists():
+            self.run_results_file.write_text(json.dumps([], indent=2))
+        if not self.issues_file.exists():
+            self.issues_file.write_text(json.dumps([], indent=2))
+        if not self.runbooks_file.exists():
+            self.runbooks_file.write_text(json.dumps([], indent=2))
+
+    def load_metadata(self) -> dict:
+        if not self.metadata_file.exists():
+            raise FileNotFoundError("Workspace is not initialized. Run `qa-agent init` first.")
+        return json.loads(self.metadata_file.read_text())
+
+    def save_metadata(self, metadata: dict) -> None:
+        self.metadata_file.write_text(json.dumps(metadata, indent=2))
+
+    def load_flows(self) -> list[Flow]:
+        if not self.flows_file.exists():
+            return []
+        data = json.loads(self.flows_file.read_text())
+        return [Flow.from_dict(item) for item in data]
+
+    def save_flow(self, flow: Flow) -> None:
+        flows = self.load_flows()
+        upserted = False
+        for idx, existing in enumerate(flows):
+            if existing.name == flow.name:
+                flows[idx] = flow
+                upserted = True
+                break
+        if not upserted:
+            flows.append(flow)
+        self.flows_file.write_text(json.dumps([asdict(item) for item in flows], indent=2))
+
+    def find_flow(self, name: str) -> Optional[Flow]:
+        for flow in self.load_flows():
+            if flow.name == name:
+                return flow
+        return None
+
+    def save_environment(self, environment: Environment) -> None:
+        metadata = self.load_metadata()
+        envs: List[dict] = metadata.get("environments", [])
+        upserted = False
+        for idx, existing in enumerate(envs):
+            if existing.get("name") == environment.name:
+                envs[idx] = environment.to_dict()
+                upserted = True
+                break
+        if not upserted:
+            envs.append(environment.to_dict())
+        metadata["environments"] = envs
+        self.save_metadata(metadata)
+
+    def load_environments(self) -> list[Environment]:
+        metadata = self.load_metadata()
+        envs = metadata.get("environments", [])
+        return [Environment.from_dict(item) for item in envs]
+
+    def save_success_metrics(self, metrics: List[SuccessMetric]) -> None:
+        metadata = self.load_metadata()
+        metadata["success_metrics"] = [metric.to_dict() for metric in metrics]
+        self.save_metadata(metadata)
+
+    def load_success_metrics(self) -> list[SuccessMetric]:
+        metadata = self.load_metadata()
+        metrics = metadata.get("success_metrics", [])
+        return [SuccessMetric.from_dict(item) for item in metrics]
+
+    def save_fixture(self, fixture: Fixture) -> None:
+        metadata = self.load_metadata()
+        fixtures: List[dict] = metadata.get("fixtures", [])
+        upserted = False
+        for idx, existing in enumerate(fixtures):
+            if existing.get("name") == fixture.name:
+                fixtures[idx] = fixture.to_dict()
+                upserted = True
+                break
+        if not upserted:
+            fixtures.append(fixture.to_dict())
+        metadata["fixtures"] = fixtures
+        self.save_metadata(metadata)
+
+    def load_fixtures(self) -> list[Fixture]:
+        metadata = self.load_metadata()
+        fixtures = metadata.get("fixtures", [])
+        return [Fixture.from_dict(item) for item in fixtures]
+
+    def save_secret(self, secret: SecretConfig) -> None:
+        metadata = self.load_metadata()
+        secrets: List[dict] = metadata.get("secrets", [])
+        upserted = False
+        for idx, existing in enumerate(secrets):
+            if existing.get("name") == secret.name:
+                secrets[idx] = secret.to_dict()
+                upserted = True
+                break
+        if not upserted:
+            secrets.append(secret.to_dict())
+        metadata["secrets"] = secrets
+        self.save_metadata(metadata)
+
+    def load_secrets(self) -> list[SecretConfig]:
+        metadata = self.load_metadata()
+        secrets = metadata.get("secrets", [])
+        return [SecretConfig.from_dict(item) for item in secrets]
+
+    def save_notification_channel(self, channel: NotificationChannel) -> None:
+        metadata = self.load_metadata()
+        channels: List[dict] = metadata.get("notification_channels", [])
+        channels.append(channel.to_dict())
+        metadata["notification_channels"] = channels
+        self.save_metadata(metadata)
+
+    def load_notification_channels(self) -> list[NotificationChannel]:
+        metadata = self.load_metadata()
+        channels = metadata.get("notification_channels", [])
+        return [NotificationChannel.from_dict(item) for item in channels]
+
+    def save_override(self, override: EnvironmentOverride) -> None:
+        metadata = self.load_metadata()
+        overrides: List[dict] = metadata.get("overrides", [])
+        overrides.append(override.to_dict())
+        metadata["overrides"] = overrides
+        self.save_metadata(metadata)
+
+    def load_overrides(self) -> list[EnvironmentOverride]:
+        metadata = self.load_metadata()
+        overrides = metadata.get("overrides", [])
+        return [EnvironmentOverride.from_dict(item) for item in overrides]
+
+    def record_artifact(self, artifact: RunArtifact) -> None:
+        artifacts = self.load_artifacts()
+        artifacts.append(artifact)
+        self.artifacts_file.write_text(json.dumps([item.to_dict() for item in artifacts], indent=2))
+
+    def load_artifacts(self) -> list[RunArtifact]:
+        if not self.artifacts_file.exists():
+            return []
+        data = json.loads(self.artifacts_file.read_text())
+        return [RunArtifact.from_dict(item) for item in data]
+
+    def record_run_result(self, result: FlowRunResult) -> None:
+        results = self.load_run_results()
+        results.append(result)
+        self.run_results_file.write_text(json.dumps([item.to_dict() for item in results], indent=2))
+
+    def load_run_results(self) -> list[FlowRunResult]:
+        if not self.run_results_file.exists():
+            return []
+        data = json.loads(self.run_results_file.read_text())
+        return [FlowRunResult.from_dict(item) for item in data]
+
+    def record_issue_draft(self, draft: IssueDraft) -> None:
+        drafts = self.load_issue_drafts()
+        drafts.append(draft)
+        self.issues_file.write_text(json.dumps([item.to_dict() for item in drafts], indent=2))
+
+    def load_issue_drafts(self) -> list[IssueDraft]:
+        if not self.issues_file.exists():
+            return []
+        data = json.loads(self.issues_file.read_text())
+        return [IssueDraft.from_dict(item) for item in data]
+
+    def save_flake_report(self, report: dict) -> None:
+        self.flake_report_file.write_text(json.dumps(report, indent=2))
+
+    def load_flake_report(self) -> dict | None:
+        if not self.flake_report_file.exists():
+            return None
+        return json.loads(self.flake_report_file.read_text())
+
+    def save_regeneration_queue(self, queue: list[dict]) -> None:
+        self.regeneration_queue_file.write_text(json.dumps(queue, indent=2))
+
+    def load_regeneration_queue(self) -> list[dict]:
+        if not self.regeneration_queue_file.exists():
+            return []
+        data = json.loads(self.regeneration_queue_file.read_text())
+        return data if isinstance(data, list) else []
+
+    def save_validation_report(self, report: dict) -> None:
+        self.validation_report_file.write_text(json.dumps(report, indent=2))
+
+    def load_validation_report(self) -> dict | None:
+        if not self.validation_report_file.exists():
+            return None
+        return json.loads(self.validation_report_file.read_text())
+
+    def save_runbook_entry(self, entry: RunbookEntry) -> None:
+        runbooks = self.load_runbook_entries()
+        runbooks.append(entry)
+        self.runbooks_file.write_text(json.dumps([item.to_dict() for item in runbooks], indent=2))
+
+    def load_runbook_entries(self) -> list[RunbookEntry]:
+        if not self.runbooks_file.exists():
+            return []
+        data = json.loads(self.runbooks_file.read_text())
+        return [RunbookEntry.from_dict(item) for item in data]
