import json
from pathlib import Path

import pytest

from ribench_agentic import batch
from ribench_agentic.build.court import build_court_bundle
from test_batch import plan_for


@pytest.fixture
def setup_plan(tmp_path, cases, injections):
    bundle = build_court_bundle(cases['court'], injections['court'], 'A', output_root=tmp_path / 'bundles')
    plan = plan_for(bundle)
    plan['conditions'] = ['vanilla', 'obligation']
    return plan, tmp_path / 'runs'


def test_completed_before_ledger_recovers_without_pi(setup_plan, fake_pi, monkeypatch):
    plan, root = setup_plan
    entry = batch.episodes(plan, root)[0]
    ledger = root / 'offline/ledger.jsonl'
    original = batch.append_invocation

    def interrupted(*args, **kwargs):
        raise RuntimeError('crash after episode.json')

    monkeypatch.setattr(batch, 'append_invocation', interrupted)
    with pytest.raises(RuntimeError):
        batch.run_episode(*entry, ledger, False, fake_pi)
    assert (entry[1] / 'episode.json').exists()
    monkeypatch.setattr(batch, 'append_invocation', original)
    batch.run_episode(*entry, ledger, False, 'DO-NOT-EXECUTE')
    assert len(ledger.read_text().splitlines()) == 1
    assert (entry[1] / 'extraction.json').exists()


@pytest.mark.parametrize('changed', ['wall_seconds', 'status', 'usage', 'returned_model'])
def test_reconcile_existing_invocation_is_idempotent(setup_plan, fake_pi, changed):
    plan, root = setup_plan
    entry = batch.episodes(plan, root)[0]
    ledger = root / 'offline/ledger.jsonl'
    batch.run_episode(*entry, ledger, False, fake_pi)
    original = ledger.read_text()
    path = entry[1] / 'episode.json'
    result = json.loads(path.read_text())
    result[changed] = {'input': 999} if changed == 'usage' else (999 if changed == 'wall_seconds' else 'different')
    path.write_text(json.dumps(result))
    batch.reconcile_ledger(entry[1], entry[2], ledger)
    batch.reconcile_ledger(entry[1], entry[2], ledger)
    assert ledger.read_text() == original
    conflict = ledger.with_name('ledger_conflict.jsonl')
    assert conflict.exists() == (changed != 'wall_seconds')
    if conflict.exists():
        assert json.loads(conflict.read_text().splitlines()[0])['invocation_id']


def test_controller_failure_isolation_progress_resume(setup_plan, fake_pi, monkeypatch, capsys):
    from ribench_agentic import controller
    plan, root = setup_plan
    original = controller.run_episode
    calls = []

    def fail_first(*args, **kwargs):
        calls.append(args[1])
        if len(calls) == 1:
            raise RuntimeError('isolated failure')
        return original(*args, **kwargs)

    monkeypatch.setattr(controller, 'run_episode', fail_first)
    assert controller.run_controller(plan, runs_root=root, pi_binary=fake_pi) == 2
    progress = json.loads((root / 'offline/progress.json').read_text())
    assert progress['counts'] == {'error': 1, 'complete': 1}
    assert progress['attempted'] == 2
    assert progress['last_episode_dir'] == str(calls[-1])
    assert progress['elapsed_seconds'] >= 0
    assert progress['token_totals']['input'] == 1735
    failure = json.loads((root / 'offline/failures.jsonl').read_text())
    assert 'RuntimeError: isolated failure' in failure['traceback']
    assert len(capsys.readouterr().out.splitlines()) == 2
    monkeypatch.setattr(controller, 'run_episode', original)
    assert controller.run_controller(plan, runs_root=root, pi_binary=fake_pi, resume=True) == 0
    assert len((root / 'offline/ledger.jsonl').read_text().splitlines()) == 2
    assert controller.run_controller(plan, runs_root=root, pi_binary='DO-NOT-EXECUTE', resume=True) == 0
    assert json.loads((root / 'offline/progress.json').read_text())['skipped'] == 2


def test_controller_stop_between_episodes(setup_plan, fake_pi, monkeypatch):
    from ribench_agentic import controller
    plan, root = setup_plan
    original = controller.run_episode

    def stop_after(*args, **kwargs):
        result = original(*args, **kwargs)
        (root / 'offline/STOP').touch()
        return result

    monkeypatch.setattr(controller, 'run_episode', stop_after)
    assert controller.run_controller(plan, runs_root=root, pi_binary=fake_pi) == 0
    progress = json.loads((root / 'offline/progress.json').read_text())
    assert progress['attempted'] == 1 and progress['stopped'] is True
    assert controller.run_controller(plan, runs_root=root, pi_binary='DO-NOT-EXECUTE') == 0
    assert json.loads((root / 'offline/progress.json').read_text())['attempted'] == 0


def test_controller_failed_result_and_limit(setup_plan, fake_pi):
    from ribench_agentic.controller import run_controller
    plan, root = setup_plan
    script = Path(fake_pi)
    script.write_text(script.read_text() + 'exit 1\n')
    assert run_controller(plan, runs_root=root, pi_binary=fake_pi, max_episodes=1) == 2
    assert json.loads((root / 'offline/progress.json').read_text())['counts'] == {'error': 1}


def test_controller_cli(setup_plan, monkeypatch, tmp_path):
    from ribench_agentic import cli, controller
    plan, root = setup_plan
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(controller, 'run_controller', lambda **kwargs: 2)
    with pytest.raises(SystemExit) as error:
        cli.main(['controller', '--plan', str(path), '--runs-root', str(root), '--resume', '--concurrency', '1'])
    assert error.value.code == 2


def test_prior_reconcile_row_does_not_abort_extraction(setup_plan, fake_pi, monkeypatch):
    plan, root = setup_plan
    entry = batch.episodes(plan, root)[0]
    ledger = root / 'offline/ledger.jsonl'
    original = batch.run_pi

    def recorded_then_reconciled(*args, **kwargs):
        result = original(*args, **kwargs)
        batch._reconcile_ledger(entry[1], {**entry[2], 'bundle_sha256': 'earlier-context'}, ledger)
        return result

    monkeypatch.setattr(batch, 'run_pi', recorded_then_reconciled)
    result = batch.run_episode(*entry, ledger, False, fake_pi)
    assert result['status'] == 'complete'
    assert (entry[1] / 'extraction.json').exists()
    assert len(ledger.read_text().splitlines()) == 1
    assert not ledger.with_name('ledger_conflict.jsonl').exists()


def test_reconcile_waits_until_runner_has_accounted(setup_plan, fake_pi, monkeypatch):
    import fcntl
    plan, root = setup_plan
    entry = batch.episodes(plan, root)[0]
    ledger = root / 'offline/ledger.jsonl'
    original = batch.append_invocation

    def assert_lock(*args, **kwargs):
        with (entry[1] / '.episode.lock').open('a') as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return original(*args, **kwargs)

    batch.run_episode(*entry, ledger, False, fake_pi)
    monkeypatch.setattr(batch, 'append_invocation', assert_lock)
    batch.reconcile_ledger(entry[1], entry[2], ledger)


def test_controller_preserves_failure_status(setup_plan, monkeypatch):
    from ribench_agentic import controller
    plan, root = setup_plan
    monkeypatch.setattr(controller, 'run_episode', lambda *args: {'status': 'timeout'})
    assert controller.run_controller(plan, runs_root=root, max_episodes=1) == 2
    assert json.loads((root / 'offline/progress.json').read_text())['counts'] == {'timeout': 1}


def test_resume_recorded_role_accounts_before_next_role(tmp_path, cases, fake_pi, monkeypatch):
    from ribench_agentic.build.open_review import build_open_review_skeleton
    from ribench_agentic.generate_reviews import generate_reviews
    skeleton = build_open_review_skeleton(cases['open_review'], 'A', output_root=tmp_path / 'skeleton')
    frozen = generate_reviews(skeleton, output_root=tmp_path / 'frozen', work_root=tmp_path / 'generation', pi_binary=fake_pi)
    root = tmp_path / 'runs'
    entry = batch.episodes(plan_for(frozen, 'open_review'), root)[0]
    ledger = root / 'offline/ledger.jsonl'
    original_append = batch.append_invocation
    original_pi = batch.run_pi

    def crash(*args, **kwargs):
        raise RuntimeError('interrupted accounting')

    monkeypatch.setattr(batch, 'append_invocation', crash)
    with pytest.raises(RuntimeError):
        batch.run_episode(*entry, ledger, False, fake_pi)
    recorded = entry[1] / 'area_chair/episode.json'
    previous = recorded.read_bytes()
    monkeypatch.setattr(batch, 'append_invocation', original_append)

    def only_program_chair(*args, **kwargs):
        assert kwargs['output_dir'].name == 'program_chair'
        assert len(ledger.read_text().splitlines()) == 1
        return original_pi(*args, **kwargs)

    monkeypatch.setattr(batch, 'run_pi', only_program_chair)
    assert batch.run_episode(*entry, ledger, False, fake_pi)['status'] == 'complete'
    assert recorded.read_bytes() == previous
    assert len(ledger.read_text().splitlines()) == 2
    assert (entry[1] / 'extraction.json').exists()
