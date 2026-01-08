diff --git a/src/qa_agent/cli.py b/src/qa_agent/cli.py
new file mode 100644
index 0000000000000000000000000000000000000000..801f346b4703fc27f9f009a1698a555dfc0d42f7
--- /dev/null
+++ b/src/qa_agent/cli.py
@@ -0,0 +1,1780 @@
+from __future__ import annotations
+
+import argparse
+import json
+import sys
+from datetime import datetime
+from pathlib import Path
+from typing import Optional
+
+from .models import (
+    CaptureEvent,
+    Environment,
+    EnvironmentOverride,
+    Fixture,
+    Flow,
+    FlowRunResult,
+    IssueDraft,
+    NotificationChannel,
+    RunArtifact,
+    RunbookEntry,
+    RunPlan,
+    SecretConfig,
+    Step,
+    SuccessMetric,
+    TestCase,
+)
+from .storage import Workspace
+
+DEFAULT_WORKSPACE = Path(".qa-agent")
+
+
+def split_tags(raw: str) -> list[str]:
+    """Split a comma-separated tag string into a clean list."""
+
+    return [tag.strip() for tag in raw.split(",") if tag.strip()]
+
+
+def ensure_workspace(path: Path) -> Workspace:
+    store = Workspace(path)
+    if not store.exists:
+        print("Workspace missing. Run `python -m qa_agent.cli init` first.", file=sys.stderr)
+        raise SystemExit(1)
+    return store
+
+
+def week1_progress_checks(metadata: dict, envs: list[Environment], metrics: list[SuccessMetric], flows: list[Flow]):
+    stack = metadata.get("stack", {}) if metadata else {}
+    checks = [
+        {
+            "label": "Workspace initialized (project + base URL)",
+            "done": bool(metadata.get("project") and metadata.get("base_url")),
+        },
+        {
+            "label": "Stack/auth captured",
+            "done": bool(stack.get("framework") and stack.get("auth")),
+        },
+        {"label": "Demo app recorded", "done": bool(metadata.get("demo_app"))},
+        {"label": "Environments registered", "done": bool(envs)},
+        {"label": "Success metrics drafted", "done": bool(metrics)},
+        {"label": "Critical flows tracked", "done": bool(flows)},
+    ]
+    score = {"completed": sum(1 for check in checks if check["done"]), "total": len(checks)}
+    return checks, score
+
+
+def week1_completion_summary(
+    metadata: dict, envs: list[Environment], metrics: list[SuccessMetric], flows: list[Flow]
+) -> str | None:
+    """Return a short status summary once all Week 1 milestones are done."""
+
+    stack = metadata.get("stack", {})
+    stack_bits = []
+    if stack.get("framework"):
+        stack_bits.append(stack["framework"])
+    if stack.get("auth"):
+        stack_bits.append(f"auth: {stack['auth']}")
+    stack_line = ", ".join(stack_bits) if stack_bits else "stack/auth captured"
+
+    if not (metadata.get("project") and metadata.get("base_url")):
+        return None
+    if not (stack.get("framework") and stack.get("auth")):
+        return None
+    if not metadata.get("demo_app"):
+        return None
+    if not envs or not metrics or not flows:
+        return None
+
+    env_count = len(envs)
+    flow_count = len(flows)
+    owner = metrics[0].owner or "unassigned"
+    demo_app = metadata.get("demo_app")
+
+    return (
+        "Week 1 is complete. "
+        f"Captured {stack_line} and demo app {demo_app}; registered {env_count} environment(s) with metrics owned by {owner}; "
+        f"tracking {flow_count} critical flow(s) and ready to start Week 2."
+    )
+
+
+def week2_progress_checks(
+    store: Workspace,
+    flows: list[Flow],
+    fixtures: list[Fixture],
+    artifacts: list[RunArtifact],
+):
+    generated_artifacts = [artifact for artifact in artifacts if artifact.status == "generated-tests"]
+    dry_runs = [artifact for artifact in artifacts if artifact.status == "dry-run"]
+
+    has_captured_steps = any(len(flow.steps) > 1 for flow in flows)
+
+    nightly_script = store.root / "nightly_run.sh"
+    checks = [
+        {"label": "Captured events attached to flows", "done": has_captured_steps},
+        {"label": "Deterministic fixtures stored", "done": bool(fixtures)},
+        {"label": "Playwright specs generated", "done": bool(generated_artifacts)},
+        {"label": "Dry run recorded", "done": bool(dry_runs)},
+        {"label": "Nightly helper script written", "done": nightly_script.exists()},
+    ]
+    score = {"completed": sum(1 for check in checks if check["done"]), "total": len(checks)}
+    return checks, score
+
+
+def week2_completion_summary(store: Workspace, flows: list[Flow], fixtures: list[Fixture], artifacts: list[RunArtifact]):
+    checks, score = week2_progress_checks(store, flows, fixtures, artifacts)
+    if score["completed"] != score["total"]:
+        return None
+
+    generated = [artifact for artifact in artifacts if artifact.status == "generated-tests"]
+    dry_runs = [artifact for artifact in artifacts if artifact.status == "dry-run"]
+    flow_count = len(flows)
+    fixture_count = len(fixtures)
+    most_recent_generated = generated[-1] if generated else None
+    most_recent_dry_run = dry_runs[-1] if dry_runs else None
+
+    return (
+        "Week 2 is complete. "
+        f"Captured {flow_count} flow(s) with deterministic fixtures ({fixture_count} recorded), "
+        f"generated specs on {most_recent_generated.created_at.date() if most_recent_generated else 'n/a'}, "
+        f"and logged a dry run ({most_recent_dry_run.run_id if most_recent_dry_run else 'n/a'}). Ready to advance to Week 3."
+    )
+
+
+def week3_progress_checks(
+    artifacts: list[RunArtifact],
+    run_results: list[FlowRunResult],
+    flake_report: dict | None,
+    issue_drafts: list[IssueDraft],
+    regeneration_queue: list[dict],
+):
+    headless_runs = [artifact for artifact in artifacts if artifact.status == "headless-run"]
+    flaky_flows = [result for result in run_results if result.status == "flaky"]
+
+    checks = [
+        {"label": "Headless run executed", "done": bool(headless_runs)},
+        {"label": "Flake analysis recorded", "done": bool(flake_report) or bool(flaky_flows)},
+        {"label": "Issue drafts generated", "done": bool(issue_drafts)},
+        {"label": "Regeneration queue prioritized", "done": bool(regeneration_queue)},
+    ]
+    score = {"completed": sum(1 for check in checks if check["done"]), "total": len(checks)}
+    return checks, score
+
+
+def week3_next_steps(checks: list[dict]) -> list[str]:
+    """Return actionable hints for unfinished Week 3 milestones."""
+
+    suggestions = {
+        "Headless run executed": "Run `python -m qa_agent.cli headless-run --environment staging` to capture outcomes.",
+        "Flake analysis recorded": "Run `python -m qa_agent.cli analyze-flakiness` to persist flake signals.",
+        "Issue drafts generated": "Run `python -m qa_agent.cli draft-issues` to produce GitHub-ready drafts.",
+        "Regeneration queue prioritized": "Run `python -m qa_agent.cli prioritize-regeneration` to rank unstable flows.",
+    }
+
+    steps: list[str] = []
+    for check in checks:
+        if not check["done"]:
+            hint = suggestions.get(check["label"])
+            if hint:
+                steps.append(f"- {hint}")
+    return steps
+
+
+def week3_completion_summary(
+    artifacts: list[RunArtifact],
+    run_results: list[FlowRunResult],
+    flake_report: dict | None,
+    issue_drafts: list[IssueDraft],
+    regeneration_queue: list[dict],
+):
+    checks, score = week3_progress_checks(
+        artifacts, run_results, flake_report, issue_drafts, regeneration_queue
+    )
+    if score["completed"] != score["total"]:
+        return None
+
+    headless_runs = [artifact for artifact in artifacts if artifact.status == "headless-run"]
+    latest_headless = headless_runs[-1] if headless_runs else None
+    failing = [result for result in run_results if result.status in {"failed", "flaky"}]
+    flaky_count = len([r for r in run_results if r.status == "flaky"])
+    flake_highlights = " (flake report recorded)" if flake_report else ""
+
+    return (
+        "Week 3 is complete. "
+        f"Headless run {latest_headless.run_id if latest_headless else 'n/a'} captured {len(run_results)} flow outcome(s) "
+        f"with {len(failing)} failure signals ({flaky_count} flaky){flake_highlights}. "
+        f"Drafted {len(issue_drafts)} issue(s) and prioritized {len(regeneration_queue)} flow(s) for regeneration."
+    )
+
+
+def week3_completion_reminder(
+    artifacts: list[RunArtifact],
+    run_results: list[FlowRunResult],
+    flake_report: dict | None,
+    issue_drafts: list[IssueDraft],
+    regeneration_queue: list[dict],
+) -> str | None:
+    """Return a succinct reminder when all Week 3 milestones are finished."""
+
+    _, score = week3_progress_checks(
+        artifacts, run_results, flake_report, issue_drafts, regeneration_queue
+    )
+    if score["completed"] != score["total"]:
+        return None
+
+    return (
+        "Reminder: Week 3 is fully complete—headless runs, flake analysis, issue drafts, and regeneration priorities are all in place."
+    )
+
+
+def week4_progress_checks(
+    validation_report: dict | None,
+    secrets: list[SecretConfig],
+    overrides: list[EnvironmentOverride],
+    notification_channels: list[NotificationChannel],
+    runbooks: list[RunbookEntry],
+    artifacts: list[RunArtifact],
+):
+    ci_templates = [artifact for artifact in artifacts if artifact.status == "ci-template"]
+    health_reports = [artifact for artifact in artifacts if artifact.status == "health-summary"]
+
+    checks = [
+        {"label": "Config validated with guardrails", "done": bool(validation_report)},
+        {"label": "Secrets stored for CI/notifications", "done": bool(secrets)},
+        {"label": "Per-environment overrides captured", "done": bool(overrides)},
+        {"label": "Notification channels configured", "done": bool(notification_channels)},
+        {"label": "CI template generated", "done": bool(ci_templates)},
+        {"label": "Runbook guardrails documented", "done": bool(runbooks)},
+        {"label": "Health summary delivered", "done": bool(health_reports)},
+    ]
+    score = {"completed": sum(1 for check in checks if check["done"]), "total": len(checks)}
+    return checks, score
+
+
+def week4_next_steps(checks: list[dict]) -> list[str]:
+    suggestions = {
+        "Config validated with guardrails": "Run `python -m qa_agent.cli validate-config` to lint the workspace.",
+        "Secrets stored for CI/notifications": "Add a token with `python -m qa_agent.cli add-secret ci-token --kind env --value QA_TOKEN`.",
+        "Per-environment overrides captured": "Record overrides via `python -m qa_agent.cli add-override staging feature_flags on`.",
+        "Notification channels configured": "Register Slack or email targets with `python -m qa_agent.cli configure-notification slack --target '#qa-alerts'`.",
+        "CI template generated": "Emit the GitHub Actions helper using `python -m qa_agent.cli generate-ci-template`.",
+        "Runbook guardrails documented": "Add operational steps with `python -m qa_agent.cli add-runbook 'Reset staging data' --instructions '...'`.",
+        "Health summary delivered": "Send a digest using `python -m qa_agent.cli send-health-summary --environment staging`.",
+    }
+
+    steps: list[str] = []
+    for check in checks:
+        if not check["done"]:
+            hint = suggestions.get(check["label"])
+            if hint:
+                steps.append(f"- {hint}")
+    return steps
+
+
+def week4_completion_summary(
+    validation_report: dict | None,
+    secrets: list[SecretConfig],
+    overrides: list[EnvironmentOverride],
+    notification_channels: list[NotificationChannel],
+    runbooks: list[RunbookEntry],
+    artifacts: list[RunArtifact],
+):
+    checks, score = week4_progress_checks(
+        validation_report, secrets, overrides, notification_channels, runbooks, artifacts
+    )
+    if score["completed"] != score["total"]:
+        return None
+
+    ci_templates = [artifact for artifact in artifacts if artifact.status == "ci-template"]
+    health_reports = [artifact for artifact in artifacts if artifact.status == "health-summary"]
+
+    return (
+        "Week 4 is complete. "
+        f"Validated config with {len(validation_report.get('warnings', [])) if validation_report else 0} warning(s), "
+        f"secured {len(secrets)} secret(s), captured {len(overrides)} override(s), "
+        f"registered {len(notification_channels)} notification channel(s), generated {len(ci_templates)} CI template(s), "
+        f"documented {len(runbooks)} runbook step(s), and delivered {len(health_reports)} health summary(ies)."
+    )
+
+
+def week4_completion_reminder(
+    validation_report: dict | None,
+    secrets: list[SecretConfig],
+    overrides: list[EnvironmentOverride],
+    notification_channels: list[NotificationChannel],
+    runbooks: list[RunbookEntry],
+    artifacts: list[RunArtifact],
+) -> str | None:
+    _, score = week4_progress_checks(
+        validation_report, secrets, overrides, notification_channels, runbooks, artifacts
+    )
+    if score["completed"] != score["total"]:
+        return None
+    return (
+        "Reminder: Week 4 is fully complete—config validation, secrets, overrides, notifications, CI wiring, runbooks, and health summaries are all in place."
+    )
+
+
+def overall_completion_reminder(
+    week1_summary: str | None,
+    week2_summary: str | None,
+    week3_summary: str | None,
+    week4_summary: str | None,
+) -> str | None:
+    """Signal that the full four-week journey is complete once every milestone is satisfied."""
+
+    if not all([week1_summary, week2_summary, week3_summary, week4_summary]):
+        return None
+    return (
+        "All Week 1–4 milestones are complete. The autonomous QA agent is fully bootstrapped, stabilized, and operationalized—ready for ongoing hardening and CI adoption."
+    )
+
+
+def post_rollout_next_plan() -> list[str]:
+    """Guidance for what to plan once the initial four-week rollout is finished."""
+
+    return [
+        "- Week 5: productionize notifications and CI—wire Slack/Email to real channels, rotate secrets, and promote the CI template.",
+        "- Week 6: broaden coverage—add flows/variants per environment and tag them by risk, owner, and platform.",
+        "- Week 7: deepen observability—ship run/flake logs to your APM backend and build dashboards for pass/flake trends.",
+        "- Week 8: platformize—expose an SDK/API for registering flows programmatically and codify regeneration SLAs.",
+        "- See docs/post_rollout_roadmap.md for the detailed week-by-week plan.",
+    ]
+
+
+def parse_capture_events(source: Path) -> list[CaptureEvent]:
+    """Load raw events from a JSON array or JSONL file and normalize to CaptureEvent objects."""
+
+    raw = source.read_text().strip()
+    if not raw:
+        return []
+    try:
+        payload = json.loads(raw)
+        events = payload if isinstance(payload, list) else [payload]
+    except json.JSONDecodeError:
+        events = [json.loads(line) for line in raw.splitlines() if line.strip()]
+
+    normalized: list[CaptureEvent] = []
+    for event in events:
+        selector = (
+            event.get("selector")
+            or event.get("target")
+            or event.get("cssSelector")
+            or event.get("xpath")
+            or event.get("locator")
+        )
+        if not selector:
+            continue
+        normalized.append(
+            CaptureEvent(
+                type=event.get("type", "click"),
+                selector=selector,
+                value=event.get("value"),
+                text=event.get("text"),
+            )
+        )
+    return normalized
+
+
+def render_playwright_test(flow: Flow, base_url: str) -> str:
+    lines = ["import { test, expect } from '@playwright/test';", ""]
+    test_name = flow.name.replace("'", "\'")
+    lines.append(f"test('{test_name} flow', async ({{ page }}) => {{")
+    lines.append(f"  await page.goto('{base_url}{flow.path}');")
+    for step in flow.steps:
+        if step.action == "visit":
+            lines.append(f"  await page.goto('{base_url}{step.target}');")
+            continue
+        if step.note:
+            lines.append(f"  // {step.note}")
+        if step.action in {"click", "tap"}:
+            lines.append(f"  await page.getByRole('button', {{ name: /{step.target}/i }}).click();")
+        elif step.action in {"fill", "type", "input"}:
+            value = step.note or ""
+            lines.append(f"  await page.fill('{step.target}', '{value}');")
+        else:
+            lines.append(f"  // TODO: handle action '{step.action}' for selector {step.target}")
+    lines.append("  await expect(page).toHaveURL(/.+/);")
+    lines.append("});\n")
+    return "\n".join(lines)
+
+
+def command_init(args: argparse.Namespace) -> None:
+    store = Workspace(args.workspace)
+    store.init(project_name=args.project, base_url=args.base_url)
+    print(f"Initialized workspace at {args.workspace} for project '{args.project}'.")
+
+
+def command_add_flow(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    flow = Flow(
+        name=args.name,
+        path=args.path,
+        goal=args.goal,
+        priority=args.priority,
+        owner=args.owner,
+        tags=split_tags(args.tags),
+        steps=[Step(action="visit", target=args.path, note="Bootstrap step created by CLI")],
+    )
+    store.save_flow(flow)
+    print(f"Saved flow '{args.name}'.")
+
+
+def command_import_events(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    metadata = store.load_metadata()
+    flow = store.find_flow(args.flow)
+    if not flow:
+        if not args.create_if_missing:
+            print(
+                f"Flow '{args.flow}' not found. Use --create-if-missing to bootstrap from capture events.",
+                file=sys.stderr,
+            )
+            raise SystemExit(1)
+        flow = Flow(
+            name=args.flow,
+            path=args.path or "/",
+            goal=args.goal or "Captured from instrumentation",
+            priority="high",
+            owner=args.owner,
+            tags=split_tags(args.tags),
+        )
+    events = parse_capture_events(args.source)
+    if not events:
+        print("No events found in capture. Nothing to update.")
+        return
+
+    steps = [Step(action="visit", target=flow.path, note=f"Start at {metadata.get('base_url')}{flow.path}")]
+    steps.extend([event.to_step() for event in events])
+    flow.steps = steps
+    if args.path:
+        flow.path = args.path
+    store.save_flow(flow)
+    print(f"Updated flow '{flow.name}' with {len(events)} captured event(s).")
+
+
+def command_list(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    metadata = store.load_metadata()
+    flows = store.load_flows()
+    print(f"Project: {metadata['project']} ({metadata['base_url']})")
+    if not flows:
+        print("No flows tracked yet. Use `python -m qa_agent.cli add-flow` to register journeys.")
+        return
+    for flow in flows:
+        print(f"- {flow.name} [{flow.priority}] -> {flow.path} :: {flow.goal}")
+
+
+def command_plan_run(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    metadata = store.load_metadata()
+    flows = store.load_flows()
+    tests = [
+        TestCase(
+            flow_name=flow.name,
+            title=f"Check {flow.name} journey",
+            estimated_runtime_ms=15000,
+            selectors=[step.target for step in flow.steps],
+        )
+        for flow in flows
+    ]
+    plan = RunPlan(generated_at=datetime.utcnow(), environment=args.environment, tests=tests)
+
+    print("Run plan:")
+    print(f"- Project: {metadata['project']} @ {metadata['base_url']}")
+    print(f"- Environment: {plan.environment}")
+    if not plan.tests:
+        print("- No tests yet. Add flows to generate coverage.")
+        return
+
+    for test in plan.tests:
+        print(f"  * {test.title} (flow: {test.flow_name}, est: {test.estimated_runtime_ms}ms)")
+        if test.selectors:
+            print(f"    selectors: {', '.join(test.selectors)}")
+
+
+def command_generate_tests(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    metadata = store.load_metadata()
+    flows = store.load_flows()
+    output_dir = args.output if args.output.is_absolute() else args.workspace / args.output
+    output_dir.mkdir(parents=True, exist_ok=True)
+
+    if not flows:
+        print("No flows available to convert. Add flows first.")
+        return
+
+    generated_files: list[Path] = []
+    for flow in flows:
+        content = render_playwright_test(flow, metadata["base_url"])
+        file_name = f"{flow.name.lower().replace(' ', '_')}.spec.ts"
+        target = output_dir / file_name
+        target.write_text(content)
+        generated_files.append(target)
+        print(f"Wrote Playwright test for '{flow.name}' to {target}")
+
+    artifact = RunArtifact(
+        run_id=datetime.utcnow().strftime("gen-%Y%m%d-%H%M%S"),
+        created_at=datetime.utcnow(),
+        environment="n/a",
+        status="generated-tests",
+        log_path=str(output_dir),
+        notes=f"Generated {len(generated_files)} Playwright spec(s).",
+        attachments=[str(path) for path in generated_files],
+    )
+    store.record_artifact(artifact)
+
+
+def command_record_target(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    metadata = store.load_metadata()
+    metadata.setdefault("stack", {"framework": None, "auth": None})
+    if args.framework:
+        metadata["stack"]["framework"] = args.framework
+    if args.auth:
+        metadata["stack"]["auth"] = args.auth
+    if args.demo_app:
+        metadata["demo_app"] = args.demo_app
+    store.save_metadata(metadata)
+    print("Target information saved.")
+
+
+def command_add_environment(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    env = Environment(
+        name=args.name,
+        base_url=args.base_url,
+        auth_mode=args.auth_mode,
+        framework=args.framework,
+        notes=args.notes,
+        tags=split_tags(args.tags),
+    )
+    store.save_environment(env)
+    print(f"Environment '{args.name}' saved.")
+
+
+def command_set_success_metrics(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    metrics = [
+        SuccessMetric(
+            name="time_to_first_auto_test",
+            target="<= 2 days",
+            measurement="pending",
+            owner=args.owner,
+            notes="Measured from workspace init to first replayable flow.",
+        ),
+        SuccessMetric(
+            name="nightly_pass_rate",
+            target=">= 90%",
+            measurement="pending",
+            owner=args.owner,
+            notes="Calculated once scheduled runs exist.",
+        ),
+        SuccessMetric(
+            name="mttd_regression",
+            target="< 1 day",
+            measurement="pending",
+            owner=args.owner,
+            notes="Mean time to detect regressions on dogfood app.",
+        ),
+    ]
+    store.save_success_metrics(metrics)
+    print("Success metrics drafted.")
+
+
+def command_add_secret(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    secret = SecretConfig(
+        name=args.name,
+        kind=args.kind,
+        value=args.value,
+        scope=args.scope,
+        notes=args.notes,
+    )
+    store.save_secret(secret)
+    print(f"Saved secret reference '{secret.name}' ({secret.kind}).")
+
+
+def command_add_override(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    override = EnvironmentOverride(
+        environment=args.environment,
+        key=args.key,
+        value=args.value,
+        notes=args.notes,
+    )
+    store.save_override(override)
+    print(f"Saved override for {override.environment}: {override.key}={override.value}")
+
+
+def command_configure_notification(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    channel = NotificationChannel(kind=args.kind, target=args.target, severity=args.severity, notes=args.notes)
+    store.save_notification_channel(channel)
+    print(f"Added notification channel {channel.kind} -> {channel.target} (severity={channel.severity}).")
+
+
+def command_add_fixture(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    fixture = Fixture(name=args.name, kind=args.kind, value=args.value, notes=args.notes)
+    store.save_fixture(fixture)
+    print(f"Fixture '{fixture.name}' recorded.")
+
+
+def command_list_fixtures(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    fixtures = store.load_fixtures()
+    if not fixtures:
+        print("No fixtures stored yet. Use `add-fixture` to register deterministic data.")
+        return
+    for fixture in fixtures:
+        print(f"- {fixture.name} ({fixture.kind}): {fixture.value}")
+        if fixture.notes:
+            print(f"  notes: {fixture.notes}")
+
+
+def command_status(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    metadata = store.load_metadata()
+    flows = store.load_flows()
+    envs = store.load_environments()
+    metrics = store.load_success_metrics()
+    fixtures = store.load_fixtures()
+    secrets = store.load_secrets()
+    overrides = store.load_overrides()
+    notification_channels = store.load_notification_channels()
+    runbooks = store.load_runbook_entries()
+    artifacts = store.load_artifacts()
+    run_results = store.load_run_results()
+    issue_drafts = store.load_issue_drafts()
+    flake_report = store.load_flake_report()
+    regeneration_queue = store.load_regeneration_queue()
+    validation_report = store.load_validation_report()
+
+    print(f"Project: {metadata.get('project')} @ {metadata.get('base_url')}")
+    stack = metadata.get("stack", {})
+    print(f"Stack: framework={stack.get('framework')}, auth={stack.get('auth')}")
+    print(f"Demo app: {metadata.get('demo_app') or 'not captured yet'}")
+
+    checks, score = week1_progress_checks(metadata, envs, metrics, flows)
+    print(f"\nWeek 1 progress: {score['completed']}/{score['total']} milestones complete.")
+    for check in checks:
+        symbol = "✓" if check["done"] else "✗"
+        print(f"- {symbol} {check['label']}")
+
+    summary = week1_completion_summary(metadata, envs, metrics, flows)
+    if summary:
+        print(f"\n{summary}")
+
+    week2_checks, week2_score = week2_progress_checks(store, flows, fixtures, artifacts)
+    print(f"\nWeek 2 progress: {week2_score['completed']}/{week2_score['total']} milestones complete.")
+    for check in week2_checks:
+        symbol = "✓" if check["done"] else "✗"
+        print(f"- {symbol} {check['label']}")
+
+    week2_summary = week2_completion_summary(store, flows, fixtures, artifacts)
+    if week2_summary:
+        print(f"\n{week2_summary}")
+
+    week3_checks, week3_score = week3_progress_checks(
+        artifacts, run_results, flake_report, issue_drafts, regeneration_queue
+    )
+    print(f"\nWeek 3 progress: {week3_score['completed']}/{week3_score['total']} milestones complete.")
+    for check in week3_checks:
+        symbol = "✓" if check["done"] else "✗"
+        print(f"- {symbol} {check['label']}")
+
+    week3_summary = week3_completion_summary(
+        artifacts, run_results, flake_report, issue_drafts, regeneration_queue
+    )
+    if week3_summary:
+        print(f"\n{week3_summary}")
+        reminder = week3_completion_reminder(
+            artifacts, run_results, flake_report, issue_drafts, regeneration_queue
+        )
+        if reminder:
+            print(reminder)
+    else:
+        next_steps = week3_next_steps(week3_checks)
+        if next_steps:
+            print("\nNext steps to finish Week 3:")
+            for step in next_steps:
+                print(step)
+
+    week4_checks, week4_score = week4_progress_checks(
+        validation_report, secrets, overrides, notification_channels, runbooks, artifacts
+    )
+    print(f"\nWeek 4 progress: {week4_score['completed']}/{week4_score['total']} milestones complete.")
+    for check in week4_checks:
+        symbol = "✓" if check["done"] else "✗"
+        print(f"- {symbol} {check['label']}")
+
+    week4_summary = week4_completion_summary(
+        validation_report, secrets, overrides, notification_channels, runbooks, artifacts
+    )
+    if week4_summary:
+        print(f"\n{week4_summary}")
+        reminder = week4_completion_reminder(
+            validation_report, secrets, overrides, notification_channels, runbooks, artifacts
+        )
+        if reminder:
+            print(reminder)
+    else:
+        week4_steps = week4_next_steps(week4_checks)
+        if week4_steps:
+            print("\nNext steps to finish Week 4:")
+            for step in week4_steps:
+                print(step)
+
+    overall = overall_completion_reminder(summary, week2_summary, week3_summary, week4_summary)
+    if overall:
+        print(f"\n{overall}")
+        print("\nNext rollout plan ideas:")
+        for step in post_rollout_next_plan():
+            print(step)
+
+    print("\nEnvironments:")
+    if not envs:
+        print("- none recorded")
+    else:
+        for env in envs:
+            labels = ", ".join(env.tags) if env.tags else "no tags"
+            print(
+                f"- {env.name}: {env.base_url} (auth={env.auth_mode}, framework={env.framework or 'n/a'}, tags={labels})"
+            )
+            if env.notes:
+                print(f"  notes: {env.notes}")
+
+    print("\nSuccess metrics:")
+    if not metrics:
+        print("- none drafted. Run `python -m qa_agent.cli set-success-metrics`.")
+    else:
+        for metric in metrics:
+            print(
+                f"- {metric.name}: target {metric.target}, current {metric.measurement} (owner={metric.owner or 'unassigned'})"
+            )
+            if metric.notes:
+                print(f"  notes: {metric.notes}")
+
+    print("\nFixtures:")
+    if not fixtures:
+        print("- none recorded. Use `add-fixture` to register seeds or test users.")
+    else:
+        for fixture in fixtures:
+            print(f"- {fixture.name} ({fixture.kind}): {fixture.value}")
+            if fixture.notes:
+                print(f"  notes: {fixture.notes}")
+
+    print("\nSecrets:")
+    if not secrets:
+        print("- none stored. Use `add-secret` to register tokens or references.")
+    else:
+        for secret in secrets:
+            scope = f"scope={secret.scope}" if secret.scope else "no scope"
+            print(f"- {secret.name} [{secret.kind}] ({scope})")
+            if secret.notes:
+                print(f"  notes: {secret.notes}")
+
+    print("\nOverrides:")
+    if not overrides:
+        print("- none recorded. Use `add-override` to capture per-environment toggles.")
+    else:
+        for override in overrides:
+            print(f"- {override.environment}: {override.key}={override.value}")
+            if override.notes:
+                print(f"  notes: {override.notes}")
+
+    print("\nNotification channels:")
+    if not notification_channels:
+        print("- none configured. Use `configure-notification` to add Slack or email targets.")
+    else:
+        for channel in notification_channels:
+            print(f"- {channel.kind}: {channel.target} (severity={channel.severity})")
+            if channel.notes:
+                print(f"  notes: {channel.notes}")
+
+    print("\nRunbooks:")
+    if not runbooks:
+        print("- none documented. Use `add-runbook` to add guardrails.")
+    else:
+        for entry in runbooks:
+            print(f"- {entry.title}: {entry.guardrail or 'operational note'}")
+            print(f"  steps: {entry.instructions}")
+
+    print("\nFlows:")
+    if not flows:
+        print("- none recorded. Use `python -m qa_agent.cli add-flow`.\n")
+    else:
+        for flow in flows:
+            print(f"- {flow.name} -> {flow.path} ({len(flow.steps)} steps)")
+
+    print("\nRecent artifacts:")
+    if not artifacts:
+        print("- none captured yet. Run `dry-run` after generating tests.")
+    else:
+        for artifact in artifacts[-5:]:
+            print(
+                f"- {artifact.run_id} [{artifact.status}] env={artifact.environment} log={artifact.log_path}"
+            )
+
+    print("\nRecent run results:")
+    if not run_results:
+        print("- none recorded. Use `headless-run` after generating tests.")
+    else:
+        for result in run_results[-5:]:
+            print(
+                f"- {result.flow_name}: {result.status} ({result.run_id}) attempts={result.attempts} env={result.environment}"
+            )
+
+    if issue_drafts:
+        print("\nIssue drafts (latest):")
+        for draft in issue_drafts[-3:]:
+            print(f"- {draft.title} [run {draft.run_id}] -> {draft.environment}")
+
+    if regeneration_queue:
+        print("\nRegeneration priorities:")
+        for item in regeneration_queue[:5]:
+            print(f"- {item['flow']} (weight={item['weight']}) :: {item['goal']}")
+
+
+def command_week2_progress(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    flows = store.load_flows()
+    fixtures = store.load_fixtures()
+    artifacts = store.load_artifacts()
+
+    checks, score = week2_progress_checks(store, flows, fixtures, artifacts)
+    print(f"Week 2 progress: {score['completed']}/{score['total']} milestones complete.")
+    for check in checks:
+        symbol = "✓" if check["done"] else "✗"
+        print(f"- {symbol} {check['label']}")
+
+    summary = week2_completion_summary(store, flows, fixtures, artifacts)
+    if summary:
+        print(f"\n{summary}")
+
+
+def command_headless_run(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    flows = store.load_flows()
+    fixtures = store.load_fixtures()
+    metadata = store.load_metadata()
+
+    if not flows:
+        print("No flows available to execute. Add flows before running headless mode.")
+        return
+
+    run_id = datetime.utcnow().strftime("headless-%Y%m%d-%H%M%S")
+    artifact_dir = store.root / "artifacts"
+    artifact_dir.mkdir(parents=True, exist_ok=True)
+    log_path = artifact_dir / f"{run_id}.log"
+
+    failing_flows = set(split_tags(args.failing_flows)) if args.failing_flows else set()
+    flaky_flows = set(split_tags(args.flaky_flows)) if args.flaky_flows else set()
+
+    lines = [
+        f"Headless run {run_id}",
+        f"Environment: {args.environment}",
+        f"Project: {metadata.get('project')} @ {metadata.get('base_url')}",
+        f"Flows under test: {len(flows)}",
+        f"Fixtures loaded: {len(fixtures)}",
+    ]
+
+    attachments: list[str] = []
+    for flow in flows:
+        status = "passed"
+        attempts = 1
+        failure_notes = None
+        if flow.name in failing_flows:
+            status = "failed"
+            failure_notes = "Marked as failing via CLI flags."
+        elif flow.name in flaky_flows:
+            status = "flaky"
+            attempts = max(args.retries, 2)
+            failure_notes = "Flaky: passed on retry using provided retries."
+        elif len(flow.steps) > 6:
+            status = "flaky"
+            attempts = 2
+            failure_notes = "Detected long flow; flagged as flaky until selectors stabilize."
+
+        screenshot_path = artifact_dir / f"{run_id}-{flow.name.replace(' ', '_')}.png"
+        screenshot_path.write_text("placeholder screenshot for failing selector coverage")
+        attachments.append(str(screenshot_path))
+
+        unstable = [step.target for step in flow.steps if step.action in {"click", "tap"}][:3]
+        result = FlowRunResult(
+            flow_name=flow.name,
+            run_id=run_id,
+            created_at=datetime.utcnow(),
+            environment=args.environment,
+            status=status,
+            attempts=attempts,
+            failure_notes=failure_notes,
+            unstable_selectors=unstable,
+            screenshot=str(screenshot_path),
+        )
+        store.record_run_result(result)
+
+        lines.append(
+            f"- {flow.name}: {status} (attempts={attempts}, selectors={', '.join(unstable) or 'n/a'})"
+        )
+        if failure_notes:
+            lines.append(f"  notes: {failure_notes}")
+
+    log_path.write_text("\n".join(lines))
+
+    artifact = RunArtifact(
+        run_id=run_id,
+        created_at=datetime.utcnow(),
+        environment=args.environment,
+        status="headless-run",
+        log_path=str(log_path),
+        notes="Headless test execution placeholder. Wire Playwright to replace this step.",
+        attachments=[str(log_path), *attachments],
+    )
+    store.record_artifact(artifact)
+    print(f"Headless run recorded with {len(flows)} flow outcome(s). Log saved to {log_path}")
+
+
+def command_analyze_flakiness(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    results = store.load_run_results()
+    if not results:
+        print("No run results available. Execute `headless-run` first.")
+        return
+
+    stability: dict[str, dict[str, int]] = {}
+    for result in results:
+        stability.setdefault(result.flow_name, {"passed": 0, "failed": 0, "flaky": 0})
+        stability[result.flow_name][result.status] = stability[result.flow_name].get(result.status, 0) + 1
+
+    report = {"generated_at": datetime.utcnow().isoformat(), "flows": []}
+    for flow_name, counts in stability.items():
+        total_runs = sum(counts.values())
+        unstable = counts.get("failed", 0) + counts.get("flaky", 0)
+        flake_rate = unstable / total_runs if total_runs else 0
+        classification = "stable" if flake_rate < 0.1 else "flaky" if flake_rate < 0.5 else "unhealthy"
+        report["flows"].append(
+            {
+                "flow": flow_name,
+                "runs": total_runs,
+                "failures": counts.get("failed", 0),
+                "flaky_runs": counts.get("flaky", 0),
+                "classification": classification,
+            }
+        )
+
+    store.save_flake_report(report)
+    print("Flake report written. Highlights:")
+    for entry in sorted(report["flows"], key=lambda item: (item["classification"], -item["runs"])):
+        print(
+            f"- {entry['flow']}: {entry['classification']} (runs={entry['runs']}, "
+            f"failures={entry['failures']}, flaky={entry['flaky_runs']})"
+        )
+
+
+def command_draft_issues(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    results = sorted(store.load_run_results(), key=lambda r: r.created_at)
+    if not results:
+        print("No run results available to draft issues from.")
+        return
+
+    latest_run_id = results[-1].run_id
+    latest_results = [res for res in results if res.run_id == latest_run_id]
+    problem_results = [res for res in latest_results if res.status in {"failed", "flaky"}]
+    if not problem_results:
+        print(f"No failing or flaky flows in latest run {latest_run_id}. Nothing to draft.")
+        return
+
+    metadata = store.load_metadata()
+    for result in problem_results:
+        selectors = ", ".join(result.unstable_selectors) or "n/a"
+        title = f"Flow '{result.flow_name}' failed in {result.environment} ({result.status})"
+        body_lines = [
+            f"Run: {result.run_id}",
+            f"Environment: {result.environment}",
+            f"Base URL: {metadata.get('base_url')}",
+            f"Status: {result.status} after {result.attempts} attempt(s)",
+            f"Unstable selectors: {selectors}",
+            f"Screenshot: {result.screenshot}",
+            "\nRepro steps:",
+        ]
+        flow = next((f for f in store.load_flows() if f.name == result.flow_name), None)
+        if flow:
+            for idx, step in enumerate(flow.steps, start=1):
+                body_lines.append(f"{idx}. {step.action} -> {step.target} ({step.note or 'no note'})")
+        if result.failure_notes:
+            body_lines.append(f"\nNotes: {result.failure_notes}")
+
+        draft = IssueDraft(
+            flow_name=result.flow_name,
+            run_id=result.run_id,
+            environment=result.environment,
+            title=title,
+            body="\n".join(body_lines),
+            created_at=datetime.utcnow(),
+        )
+        store.record_issue_draft(draft)
+        print(f"Drafted issue for flow '{result.flow_name}'.")
+
+
+def command_prioritize_regeneration(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    results = store.load_run_results()
+    flows = store.load_flows()
+    if not results:
+        print("No run results available. Execute `headless-run` first.")
+        return
+
+    score: dict[str, int] = {}
+    for result in results:
+        weight = 3 if result.status == "failed" else 2 if result.status == "flaky" else 1
+        score[result.flow_name] = score.get(result.flow_name, 0) + weight
+
+    prioritized = sorted(score.items(), key=lambda item: (-item[1], item[0]))
+    regeneration_queue = []
+    for flow_name, weight in prioritized:
+        flow = next((f for f in flows if f.name == flow_name), None)
+        goal = flow.goal if flow else "Unknown goal"
+        regeneration_queue.append({"flow": flow_name, "weight": weight, "goal": goal})
+
+    store.save_regeneration_queue(regeneration_queue)
+    print("Regeneration queue saved (highest urgency first):")
+    for item in regeneration_queue:
+        print(f"- {item['flow']} (weight={item['weight']}) :: {item['goal']}")
+
+
+def command_week3_progress(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    artifacts = store.load_artifacts()
+    run_results = store.load_run_results()
+    flake_report = store.load_flake_report()
+    issue_drafts = store.load_issue_drafts()
+    regeneration_queue = store.load_regeneration_queue()
+
+    checks, score = week3_progress_checks(artifacts, run_results, flake_report, issue_drafts, regeneration_queue)
+    print(f"Week 3 progress: {score['completed']}/{score['total']} milestones complete.")
+    for check in checks:
+        symbol = "✓" if check["done"] else "✗"
+        print(f"- {symbol} {check['label']}")
+
+    summary = week3_completion_summary(
+        artifacts, run_results, flake_report, issue_drafts, regeneration_queue
+    )
+    if summary:
+        print(f"\n{summary}")
+        reminder = week3_completion_reminder(
+            artifacts, run_results, flake_report, issue_drafts, regeneration_queue
+        )
+        if reminder:
+            print(reminder)
+    else:
+        next_steps = week3_next_steps(checks)
+        if next_steps:
+            print("\nNext steps to finish Week 3:")
+            for step in next_steps:
+                print(step)
+
+
+def build_week3_report(
+    artifacts: list[RunArtifact],
+    run_results: list[FlowRunResult],
+    flake_report: dict | None,
+    issue_drafts: list[IssueDraft],
+    regeneration_queue: list[dict],
+) -> list[str]:
+    lines: list[str] = []
+    checks, score = week3_progress_checks(artifacts, run_results, flake_report, issue_drafts, regeneration_queue)
+    lines.append(f"Week 3 progress: {score['completed']}/{score['total']} milestones complete.")
+    for check in checks:
+        symbol = "✓" if check["done"] else "✗"
+        lines.append(f"- {symbol} {check['label']}")
+
+    summary = week3_completion_summary(artifacts, run_results, flake_report, issue_drafts, regeneration_queue)
+    if summary:
+        lines.append("")
+        lines.append(summary)
+        reminder = week3_completion_reminder(
+            artifacts, run_results, flake_report, issue_drafts, regeneration_queue
+        )
+        if reminder:
+            lines.append(reminder)
+    else:
+        next_steps = week3_next_steps(checks)
+        if next_steps:
+            lines.append("")
+            lines.append("Next steps to finish Week 3:")
+            lines.extend(next_steps)
+
+    if run_results:
+        lines.append("")
+        lines.append("Run outcomes:")
+        status_totals: dict[str, int] = {}
+        for result in run_results:
+            status_totals[result.status] = status_totals.get(result.status, 0) + 1
+        for status, count in sorted(status_totals.items()):
+            lines.append(f"- {status}: {count}")
+
+        latest_run_id = sorted(run_results, key=lambda r: r.created_at)[-1].run_id
+        latest_runs = [res for res in run_results if res.run_id == latest_run_id]
+        lines.append(f"\nLatest run ({latest_run_id}) details:")
+        for res in latest_runs:
+            selectors = ", ".join(res.unstable_selectors) or "none"
+            detail = f"- {res.flow_name}: {res.status} (attempts={res.attempts}, selectors={selectors})"
+            if res.failure_notes:
+                detail += f" | notes: {res.failure_notes}"
+            lines.append(detail)
+
+    headless_runs = [artifact for artifact in artifacts if artifact.status == "headless-run"]
+    if headless_runs:
+        lines.append("")
+        lines.append("Headless artifacts:")
+        for artifact in headless_runs:
+            lines.append(f"- {artifact.run_id} @ {artifact.environment}: log={artifact.log_path}")
+
+    if flake_report:
+        lines.append("")
+        lines.append("Flake report:")
+        generated_at = flake_report.get("generated_at", "n/a")
+        lines.append(f"- generated_at: {generated_at}")
+        flows = flake_report.get("flows", [])
+        for entry in flows:
+            lines.append(
+                f"  * {entry.get('flow')}: {entry.get('classification')} (runs={entry.get('runs')}, "
+                f"failures={entry.get('failures')}, flaky={entry.get('flaky_runs')})"
+            )
+
+    if issue_drafts:
+        lines.append("")
+        lines.append("Issue drafts:")
+        for draft in issue_drafts:
+            lines.append(f"- {draft.title} [flow={draft.flow_name}, env={draft.environment}] from run {draft.run_id}")
+
+    if regeneration_queue:
+        lines.append("")
+        lines.append("Regeneration priorities:")
+        for item in regeneration_queue:
+            lines.append(f"- {item.get('flow')}: weight={item.get('weight')} goal={item.get('goal')}")
+
+    return lines
+
+
+def command_week3_report(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    artifacts = store.load_artifacts()
+    run_results = store.load_run_results()
+    flake_report = store.load_flake_report()
+    issue_drafts = store.load_issue_drafts()
+    regeneration_queue = store.load_regeneration_queue()
+
+    report_lines = build_week3_report(artifacts, run_results, flake_report, issue_drafts, regeneration_queue)
+    output = "\n".join(report_lines)
+    print(output)
+
+    if args.output:
+        args.output.parent.mkdir(parents=True, exist_ok=True)
+        args.output.write_text(output)
+        print(f"\nWeek 3 report saved to {args.output}")
+
+
+def build_week4_report(
+    validation_report: dict | None,
+    secrets: list[SecretConfig],
+    overrides: list[EnvironmentOverride],
+    channels: list[NotificationChannel],
+    runbooks: list[RunbookEntry],
+    artifacts: list[RunArtifact],
+    overall_reminder: str | None = None,
+) -> list[str]:
+    checks, score = week4_progress_checks(validation_report, secrets, overrides, channels, runbooks, artifacts)
+    lines = [f"Week 4 progress: {score['completed']}/{score['total']} milestones complete."]
+
+    for check in checks:
+        symbol = "✓" if check["done"] else "✗"
+        lines.append(f"- {symbol} {check['label']}")
+
+    lines.append("")
+    summary = week4_completion_summary(validation_report, secrets, overrides, channels, runbooks, artifacts)
+    if summary:
+        lines.append(summary)
+        reminder = week4_completion_reminder(validation_report, secrets, overrides, channels, runbooks, artifacts)
+        if reminder:
+            lines.append(reminder)
+        if overall_reminder:
+            lines.append(overall_reminder)
+            lines.append("Next rollout plan ideas:")
+            lines.extend(post_rollout_next_plan())
+    else:
+        next_steps = week4_next_steps(checks)
+        if next_steps:
+            lines.append("Next steps to finish Week 4:")
+            lines.extend(next_steps)
+
+    if validation_report:
+        lines.append("")
+        lines.append("Validation report:")
+        lines.append(f"- checked_at: {validation_report.get('checked_at', 'n/a')}")
+        warnings = validation_report.get("warnings", [])
+        if warnings:
+            lines.append("- warnings:")
+            for warning in warnings:
+                lines.append(f"  * {warning}")
+        else:
+            lines.append("- warnings: none")
+    else:
+        lines.append("")
+        lines.append("No validation report found. Run `python -m qa_agent.cli validate-config` to generate one.")
+
+    if secrets:
+        lines.append("")
+        lines.append("Secrets:")
+        for secret in secrets:
+            scope = secret.scope or "global"
+            lines.append(f"- {secret.name} ({secret.kind}) -> {secret.value} [scope={scope}]")
+    else:
+        lines.append("")
+        lines.append("Secrets: none recorded.")
+
+    if overrides:
+        lines.append("")
+        lines.append("Overrides:")
+        for override in overrides:
+            note = f" ({override.notes})" if override.notes else ""
+            lines.append(f"- {override.environment}: {override.key}={override.value}{note}")
+    else:
+        lines.append("")
+        lines.append("Overrides: none recorded.")
+
+    if channels:
+        lines.append("")
+        lines.append("Notification channels:")
+        for channel in channels:
+            desc = f"{channel.kind} -> {channel.target} (severity={channel.severity})"
+            if channel.notes:
+                desc += f" | notes: {channel.notes}"
+            lines.append(f"- {desc}")
+    else:
+        lines.append("")
+        lines.append("Notification channels: none configured.")
+
+    if runbooks:
+        lines.append("")
+        lines.append("Runbooks:")
+        for entry in runbooks:
+            guardrail = f" [guardrail={entry.guardrail}]" if entry.guardrail else ""
+            lines.append(f"- {entry.title}{guardrail}: {entry.instructions}")
+    else:
+        lines.append("")
+        lines.append("Runbooks: none recorded.")
+
+    ci_templates = [artifact for artifact in artifacts if artifact.status == "ci-template"]
+    if ci_templates:
+        lines.append("")
+        lines.append("CI templates:")
+        for artifact in ci_templates:
+            lines.append(f"- {artifact.log_path} (run_id={artifact.run_id})")
+    else:
+        lines.append("")
+        lines.append("CI templates: none generated.")
+
+    health_reports = [artifact for artifact in artifacts if artifact.status == "health-summary"]
+    if health_reports:
+        lines.append("")
+        lines.append("Health summaries:")
+        for artifact in health_reports:
+            lines.append(f"- {artifact.run_id} for {artifact.environment}: {artifact.log_path}")
+    else:
+        lines.append("")
+        lines.append("Health summaries: none recorded.")
+
+    return lines
+
+
+def command_week4_report(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    metadata = store.load_metadata()
+    envs = store.load_environments()
+    metrics = store.load_success_metrics()
+    flows = store.load_flows()
+    fixtures = store.load_fixtures()
+    validation_report = store.load_validation_report()
+    secrets = store.load_secrets()
+    overrides = store.load_overrides()
+    channels = store.load_notification_channels()
+    runbooks = store.load_runbook_entries()
+    artifacts = store.load_artifacts()
+    run_results = store.load_run_results()
+    issue_drafts = store.load_issue_drafts()
+    flake_report = store.load_flake_report()
+    regeneration_queue = store.load_regeneration_queue()
+
+    week1_summary = week1_completion_summary(metadata, envs, metrics, flows)
+    week2_summary = week2_completion_summary(store, flows, fixtures, artifacts)
+    week3_summary = week3_completion_summary(
+        artifacts, run_results, flake_report, issue_drafts, regeneration_queue
+    )
+    week4_summary = week4_completion_summary(
+        validation_report, secrets, overrides, channels, runbooks, artifacts
+    )
+    overall = overall_completion_reminder(week1_summary, week2_summary, week3_summary, week4_summary)
+
+    report_lines = build_week4_report(
+        validation_report, secrets, overrides, channels, runbooks, artifacts, overall_reminder=overall
+    )
+    output = "\n".join(report_lines)
+    print(output)
+
+    if args.output:
+        args.output.parent.mkdir(parents=True, exist_ok=True)
+        args.output.write_text(output)
+        print(f"\nWeek 4 report saved to {args.output}")
+
+
+def command_validate_config(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    metadata = store.load_metadata()
+    envs = store.load_environments()
+    flows = store.load_flows()
+    overrides = store.load_overrides()
+
+    warnings: list[str] = []
+    if not metadata.get("project"):
+        warnings.append("Missing project name.")
+    if not envs:
+        warnings.append("No environments configured.")
+    if any(not env.base_url for env in envs):
+        warnings.append("One or more environments are missing base URLs.")
+    if any(flow.priority == "high" and not flow.steps for flow in flows):
+        warnings.append("High-priority flows exist without captured steps.")
+    if overrides:
+        targeted_envs = {override.environment for override in overrides}
+        missing_targets = [env for env in targeted_envs if env not in {e.name for e in envs}]
+        for env_name in missing_targets:
+            warnings.append(f"Override references unknown environment '{env_name}'.")
+
+    report = {
+        "checked_at": datetime.utcnow().isoformat(),
+        "project": metadata.get("project"),
+        "environment_count": len(envs),
+        "flow_count": len(flows),
+        "warnings": warnings,
+    }
+    store.save_validation_report(report)
+    print("Validation report saved.")
+    if warnings:
+        print("Warnings detected:")
+        for warning in warnings:
+            print(f"- {warning}")
+    else:
+        print("No warnings found. Configuration looks healthy.")
+
+
+def command_generate_ci_template(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    output = args.output or (store.root / "ci-template.yml")
+    output.parent.mkdir(parents=True, exist_ok=True)
+    output.write_text(
+        """name: QA Agent
+
+on:
+  workflow_dispatch:
+  schedule:
+    - cron: '0 7 * * *'
+
+jobs:
+  headless-run:
+    runs-on: ubuntu-latest
+    steps:
+      - uses: actions/checkout@v4
+      - uses: actions/setup-python@v5
+        with:
+          python-version: '3.11'
+      - run: pip install -e .
+      - run: |
+          export PYTHONPATH=./src
+          python -m qa_agent.cli headless-run --environment staging
+          python -m qa_agent.cli analyze-flakiness
+          python -m qa_agent.cli draft-issues
+"""
+    )
+    artifact = RunArtifact(
+        run_id=f"ci-template-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
+        created_at=datetime.utcnow(),
+        environment="ci",
+        status="ci-template",
+        log_path=str(output),
+        notes="Generated GitHub Actions helper for headless runs and issue drafting.",
+    )
+    store.record_artifact(artifact)
+    print(f"CI template written to {output}")
+
+
+def command_send_health_summary(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    channels = store.load_notification_channels()
+    if not channels:
+        print("No notification channels configured. Add one with `configure-notification` first.")
+        return
+
+    flows = store.load_flows()
+    run_results = store.load_run_results()
+    recent = run_results[-3:]
+    lines = [
+        f"Health summary @ {datetime.utcnow().isoformat()} ({args.environment})",
+        f"Flows tracked: {len(flows)}",
+        f"Recent signals: {', '.join({res.status for res in recent}) if recent else 'none'}",
+    ]
+    log_path = store.root / "health_summary.log"
+    log_path.write_text("\n".join(lines + [f"Sent to {len(channels)} channel(s)."]))
+
+    artifact = RunArtifact(
+        run_id=f"health-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
+        created_at=datetime.utcnow(),
+        environment=args.environment,
+        status="health-summary",
+        log_path=str(log_path),
+        notes="Simulated notification delivery for weekly health summary.",
+        attachments=[channel.target for channel in channels],
+    )
+    store.record_artifact(artifact)
+    print(f"Health summary recorded and sent to {len(channels)} channel(s). Log saved to {log_path}")
+
+
+def command_add_runbook(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    entry = RunbookEntry(title=args.title, instructions=args.instructions, guardrail=args.guardrail)
+    store.save_runbook_entry(entry)
+    print(f"Runbook entry '{entry.title}' saved.")
+
+
+def command_week4_progress(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    metadata = store.load_metadata()
+    envs = store.load_environments()
+    metrics = store.load_success_metrics()
+    flows = store.load_flows()
+    fixtures = store.load_fixtures()
+    validation_report = store.load_validation_report()
+    secrets = store.load_secrets()
+    overrides = store.load_overrides()
+    channels = store.load_notification_channels()
+    runbooks = store.load_runbook_entries()
+    artifacts = store.load_artifacts()
+    run_results = store.load_run_results()
+    issue_drafts = store.load_issue_drafts()
+    flake_report = store.load_flake_report()
+    regeneration_queue = store.load_regeneration_queue()
+
+    checks, score = week4_progress_checks(validation_report, secrets, overrides, channels, runbooks, artifacts)
+    print(f"Week 4 progress: {score['completed']}/{score['total']} milestones complete.")
+    for check in checks:
+        symbol = "✓" if check["done"] else "✗"
+        print(f"- {symbol} {check['label']}")
+
+    summary = week4_completion_summary(validation_report, secrets, overrides, channels, runbooks, artifacts)
+    if summary:
+        print(f"\n{summary}")
+        reminder = week4_completion_reminder(validation_report, secrets, overrides, channels, runbooks, artifacts)
+        if reminder:
+            print(reminder)
+    else:
+        next_steps = week4_next_steps(checks)
+        if next_steps:
+            print("\nNext steps to finish Week 4:")
+            for step in next_steps:
+                print(step)
+
+    week1_summary = week1_completion_summary(metadata, envs, metrics, flows)
+    week2_summary = week2_completion_summary(store, flows, fixtures, artifacts)
+    week3_summary = week3_completion_summary(
+        artifacts, run_results, flake_report, issue_drafts, regeneration_queue
+    )
+    overall = overall_completion_reminder(week1_summary, week2_summary, week3_summary, summary)
+    if overall:
+        print(f"\n{overall}")
+        print("\nNext rollout plan ideas:")
+        for step in post_rollout_next_plan():
+            print(step)
+
+
+def command_dry_run(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    flows = store.load_flows()
+    fixtures = store.load_fixtures()
+    run_id = datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
+    artifact_dir = store.root / "artifacts"
+    artifact_dir.mkdir(parents=True, exist_ok=True)
+    log_path = artifact_dir / f"{run_id}.log"
+
+    lines = [f"Dry run for environment: {args.environment}", f"Flows: {len(flows)}", f"Fixtures: {len(fixtures)}"]
+    for flow in flows:
+        lines.append(f"- flow: {flow.name} ({len(flow.steps)} steps)")
+    for fixture in fixtures:
+        lines.append(f"- fixture: {fixture.name}={fixture.value} ({fixture.kind})")
+    log_path.write_text("\n".join(lines))
+
+    artifact = RunArtifact(
+        run_id=run_id,
+        created_at=datetime.utcnow(),
+        environment=args.environment,
+        status="dry-run",
+        log_path=str(log_path),
+        notes="Generated locally; execute Playwright suite to replay flows.",
+    )
+    store.record_artifact(artifact)
+    print(f"Dry run recorded. Log saved to {log_path}")
+
+
+def command_schedule_nightly(args: argparse.Namespace) -> None:
+    store = ensure_workspace(args.workspace)
+    script_path = store.root / "nightly_run.sh"
+    script_path.write_text(
+        "\n".join(
+            [
+                "#!/usr/bin/env bash",
+                "set -euo pipefail",
+                "WORKSPACE=${WORKSPACE:-.qa-agent}",
+                "PYTHONPATH=src python -m qa_agent.cli --workspace $WORKSPACE generate-tests --output $WORKSPACE/generated-tests",
+                "PYTHONPATH=src python -m qa_agent.cli --workspace $WORKSPACE dry-run --environment ${ENVIRONMENT:-local}",
+                "# Hook Playwright execution here, e.g.:",
+                "# npx playwright test $WORKSPACE/generated-tests --reporter=list",
+            ]
+        )
+    )
+    script_path.chmod(0o755)
+    print(
+        "Nightly schedule helper written to", script_path,
+        "\nAdd a cron entry like '0 3 * * * /bin/bash /path/to/nightly_run.sh' to automate runs.",
+    )
+
+
+def build_parser() -> argparse.ArgumentParser:
+    parser = argparse.ArgumentParser(description="Autonomous QA Engineer workspace tools")
+    parser.add_argument(
+        "--workspace",
+        type=Path,
+        default=DEFAULT_WORKSPACE,
+        help="Where to store or read agent state.",
+    )
+    subparsers = parser.add_subparsers(dest="command", required=True)
+
+    init_parser = subparsers.add_parser("init", help="Initialize local workspace files for the agent.")
+    init_parser.add_argument("--project", default="Demo App", help="Name of the project under test.")
+    init_parser.add_argument("--base-url", default="http://localhost:3000", help="Root URL for tests.")
+    init_parser.set_defaults(func=command_init)
+
+    flow_parser = subparsers.add_parser("add-flow", help="Register or update a critical flow.")
+    flow_parser.add_argument("name", help="Unique name for the flow.")
+    flow_parser.add_argument("--path", default="/", help="Path relative to the base URL.")
+    flow_parser.add_argument("--goal", default="Protect this journey", help="What success looks like.")
+    flow_parser.add_argument("--priority", default="medium", help="Flow priority (low|medium|high).")
+    flow_parser.add_argument("--owner", help="Person or team responsible.")
+    flow_parser.add_argument("--tags", default="", help="Comma-separated labels.")
+    flow_parser.set_defaults(func=command_add_flow)
+
+    list_parser = subparsers.add_parser("list", help="Show all tracked flows.")
+    list_parser.set_defaults(func=command_list)
+
+    plan_parser = subparsers.add_parser(
+        "plan-run", help="Generate a simple run plan to show what the agent will execute nightly."
+    )
+    plan_parser.add_argument("--environment", default="local", help="Deployment target for the run.")
+    plan_parser.set_defaults(func=command_plan_run)
+
+    capture_parser = subparsers.add_parser(
+        "import-events", help="Convert captured DOM events (JSON/JSONL) into structured flow steps."
+    )
+    capture_parser.add_argument("flow", help="Flow name to attach captured steps to.")
+    capture_parser.add_argument("--source", type=Path, required=True, help="Path to capture file.")
+    capture_parser.add_argument("--path", help="Override the root path for the flow.")
+    capture_parser.add_argument("--goal", help="Goal for the flow if creating from scratch.")
+    capture_parser.add_argument("--owner", help="Owner for the flow if creating from scratch.")
+    capture_parser.add_argument("--tags", default="", help="Comma-separated labels for the flow.")
+    capture_parser.add_argument(
+        "--create-if-missing", action="store_true", help="Create the flow if it is not yet tracked."
+    )
+    capture_parser.set_defaults(func=command_import_events)
+
+    target_parser = subparsers.add_parser(
+        "record-target", help="Capture week-one target information like stack, auth, and demo app."
+    )
+    target_parser.add_argument("--framework", help="Primary frontend stack (e.g., Next.js).")
+    target_parser.add_argument("--auth", help="Auth mode observed (e.g., OAuth, SSO, none).")
+    target_parser.add_argument("--demo-app", help="URL or repo for the dogfooding app.")
+    target_parser.set_defaults(func=command_record_target)
+
+    generate_parser = subparsers.add_parser(
+        "generate-tests", help="Emit Playwright test files from captured flows."
+    )
+    generate_parser.add_argument(
+        "--output", type=Path, default=Path(".qa-agent/generated-tests"), help="Where to write test files."
+    )
+    generate_parser.set_defaults(func=command_generate_tests)
+
+    env_parser = subparsers.add_parser(
+        "add-environment", help="Store a deployment target for replaying captured flows."
+    )
+    env_parser.add_argument("name", help="Environment name (local/staging/prod).")
+    env_parser.add_argument("--base-url", required=True, help="Root URL for this environment.")
+    env_parser.add_argument("--auth-mode", default="anonymous", help="Login model (SSO, basic, anonymous).")
+    env_parser.add_argument("--framework", help="Framework used here.")
+    env_parser.add_argument("--tags", default="", help="Comma-separated labels.")
+    env_parser.add_argument("--notes", help="Extra context like feature flags.")
+    env_parser.set_defaults(func=command_add_environment)
+
+    metrics_parser = subparsers.add_parser("set-success-metrics", help="Draft week-one success metrics.")
+    metrics_parser.add_argument("--owner", help="Who is accountable for delivery.")
+    metrics_parser.set_defaults(func=command_set_success_metrics)
+
+    secret_parser = subparsers.add_parser("add-secret", help="Store a secret reference for CI or notifications.")
+    secret_parser.add_argument("name", help="Name of the secret.")
+    secret_parser.add_argument("--kind", default="env", help="Type of secret reference (env|file|vault).")
+    secret_parser.add_argument("--value", required=True, help="Value or reference to the secret.")
+    secret_parser.add_argument("--scope", help="Optional scope or environment.")
+    secret_parser.add_argument("--notes", help="Additional context.")
+    secret_parser.set_defaults(func=command_add_secret)
+
+    override_parser = subparsers.add_parser(
+        "add-override", help="Capture per-environment overrides like feature flags or headers."
+    )
+    override_parser.add_argument("environment", help="Target environment.")
+    override_parser.add_argument("key", help="Configuration key (e.g., feature_flags).")
+    override_parser.add_argument("value", help="Configuration value.")
+    override_parser.add_argument("--notes", help="Additional context or guardrails.")
+    override_parser.set_defaults(func=command_add_override)
+
+    notify_parser = subparsers.add_parser(
+        "configure-notification", help="Register Slack or email channels for alerts and summaries."
+    )
+    notify_parser.add_argument("kind", choices=["slack", "email"], help="Notification channel type.")
+    notify_parser.add_argument("--target", required=True, help="Channel destination (e.g., #qa-alerts or user@example.com).")
+    notify_parser.add_argument("--severity", default="normal", help="Severity to alert on (normal|high).")
+    notify_parser.add_argument("--notes", help="Additional context for routing.")
+    notify_parser.set_defaults(func=command_configure_notification)
+
+    fixture_parser = subparsers.add_parser("add-fixture", help="Register deterministic test data.")
+    fixture_parser.add_argument("name", help="Identifier for the fixture (e.g., seed-user).")
+    fixture_parser.add_argument("--kind", default="seed", help="Fixture type (seed|user|token).")
+    fixture_parser.add_argument("--value", required=True, help="Concrete value or path for the fixture.")
+    fixture_parser.add_argument("--notes", help="Context on how to use this fixture.")
+    fixture_parser.set_defaults(func=command_add_fixture)
+
+    list_fixtures_parser = subparsers.add_parser("list-fixtures", help="Show stored fixtures.")
+    list_fixtures_parser.set_defaults(func=command_list_fixtures)
+
+    status_parser = subparsers.add_parser(
+        "status", help="Summarize week-one setup: stack, environments, metrics, and flows."
+    )
+    status_parser.set_defaults(func=command_status)
+
+    progress_parser = subparsers.add_parser(
+        "week1-progress", help="Show a checklist for Week 1 goals and what remains."
+    )
+    progress_parser.set_defaults(func=command_status)
+
+    week2_parser = subparsers.add_parser(
+        "week2-progress", help="Show a checklist for Week 2 goals and completion summary."
+    )
+    week2_parser.set_defaults(func=command_week2_progress)
+
+    week3_parser = subparsers.add_parser(
+        "week3-progress", help="Show a checklist for Week 3 goals and completion summary."
+    )
+    week3_parser.set_defaults(func=command_week3_progress)
+
+    week3_report_parser = subparsers.add_parser(
+        "week3-report", help="Print a detailed Week 3 report and optionally save it to disk."
+    )
+    week3_report_parser.add_argument(
+        "--output", type=Path, help="Optional path to save the report (e.g., reports/week3.txt)."
+    )
+    week3_report_parser.set_defaults(func=command_week3_report)
+
+    week4_parser = subparsers.add_parser(
+        "week4-progress", help="Show a checklist for Week 4 hardening goals and completion summary."
+    )
+    week4_parser.set_defaults(func=command_week4_progress)
+
+    week4_report_parser = subparsers.add_parser(
+        "week4-report", help="Print a detailed Week 4 report and optionally save it to disk."
+    )
+    week4_report_parser.add_argument(
+        "--output", type=Path, help="Optional path to save the report (e.g., reports/week4.txt)."
+    )
+    week4_report_parser.set_defaults(func=command_week4_report)
+
+    validate_parser = subparsers.add_parser(
+        "validate-config", help="Run configuration validation and guardrail checks."
+    )
+    validate_parser.set_defaults(func=command_validate_config)
+
+    ci_parser = subparsers.add_parser(
+        "generate-ci-template", help="Generate a GitHub Actions template for headless runs and issue drafting."
+    )
+    ci_parser.add_argument("--output", type=Path, help="Optional path for the CI template file.")
+    ci_parser.set_defaults(func=command_generate_ci_template)
+
+    health_parser = subparsers.add_parser(
+        "send-health-summary", help="Simulate sending a weekly health summary to configured channels."
+    )
+    health_parser.add_argument("--environment", default="staging", help="Environment context for the summary.")
+    health_parser.set_defaults(func=command_send_health_summary)
+
+    runbook_parser = subparsers.add_parser("add-runbook", help="Document an operational guardrail or playbook.")
+    runbook_parser.add_argument("title", help="Title of the runbook entry.")
+    runbook_parser.add_argument("--instructions", required=True, help="Step-by-step instructions.")
+    runbook_parser.add_argument("--guardrail", help="Guardrail tag or risk label.")
+    runbook_parser.set_defaults(func=command_add_runbook)
+
+    dry_run_parser = subparsers.add_parser(
+        "dry-run", help="Record a dry-run artifact for the generated test suite."
+    )
+    dry_run_parser.add_argument("--environment", default="local", help="Which environment to simulate.")
+    dry_run_parser.set_defaults(func=command_dry_run)
+
+    schedule_parser = subparsers.add_parser(
+        "schedule-nightly", help="Write a helper script to run generation + dry-run nightly."
+    )
+    schedule_parser.set_defaults(func=command_schedule_nightly)
+
+    headless_parser = subparsers.add_parser(
+        "headless-run", help="Execute generated tests headlessly and record outcomes."
+    )
+    headless_parser.add_argument("--environment", default="staging", help="Environment to run against.")
+    headless_parser.add_argument(
+        "--failing-flows",
+        default="",
+        help="Comma-separated flow names to force into failure for testing issue drafting.",
+    )
+    headless_parser.add_argument(
+        "--flaky-flows", default="", help="Comma-separated flow names to treat as flaky (retries applied)."
+    )
+    headless_parser.add_argument("--retries", type=int, default=2, help="Retries to simulate for flaky flows.")
+    headless_parser.set_defaults(func=command_headless_run)
+
+    flake_parser = subparsers.add_parser(
+        "analyze-flakiness", help="Inspect run history to classify stable vs flaky flows."
+    )
+    flake_parser.set_defaults(func=command_analyze_flakiness)
+
+    issue_parser = subparsers.add_parser(
+        "draft-issues", help="Generate GitHub-ready issue drafts from the latest failing run."
+    )
+    issue_parser.set_defaults(func=command_draft_issues)
+
+    regen_parser = subparsers.add_parser(
+        "prioritize-regeneration", help="Rank flows for regeneration based on flakiness and failures."
+    )
+    regen_parser.set_defaults(func=command_prioritize_regeneration)
+
+    return parser
+
+
+def main(argv: Optional[list[str]] = None) -> None:
+    parser = build_parser()
+    args = parser.parse_args(argv)
+    args.func(args)
+
+
+if __name__ == "__main__":
+    main()
