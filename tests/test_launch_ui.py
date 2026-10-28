import pytest
import launch_ui as launcher


def test_healthy_same_project_is_reused(monkeypatch):
    monkeypatch.setattr(launcher,'owned_ports',lambda root:{8509})
    monkeypatch.setattr(launcher,'stale_instance',lambda *a:False)
    monkeypatch.setattr(launcher,'healthy',lambda port:True)
    monkeypatch.setattr(launcher,'port_available',lambda port:pytest.fail('existing instance should be reused'))
    assert launcher.select_instance()==('reuse',8509)


def test_unknown_listener_is_not_reused_or_stopped(monkeypatch):
    monkeypatch.setattr(launcher,'owned_ports',lambda root:set())
    monkeypatch.setattr(launcher,'healthy',lambda port:pytest.fail('do not identify unknown services through health alone'))
    monkeypatch.setattr(launcher,'port_available',lambda port:port==8511)
    assert launcher.select_instance()==('start',8511)
    assert 8510 not in launcher.PORTS


def test_unhealthy_instance_does_not_get_reused(monkeypatch):
    monkeypatch.setattr(launcher,'owned_ports',lambda root:{8509})
    monkeypatch.setattr(launcher,'stale_instance',lambda *a:False)
    monkeypatch.setattr(launcher,'healthy',lambda port:False)
    monkeypatch.setattr(launcher,'port_available',lambda port:port==8511)
    assert launcher.select_instance()==('restart',8509)


def test_repeat_launch_does_not_spawn_duplicate(monkeypatch):
    monkeypatch.setattr(launcher,'select_instance',lambda:('reuse',8509))
    monkeypatch.setattr(launcher.subprocess,'call',lambda *a,**k:pytest.fail('duplicate process'))
    monkeypatch.setattr(launcher.webbrowser,'open',lambda *a,**k:pytest.fail('no-browser flag ignored'))
    assert launcher.main(['--no-browser'])==0


def test_busy_port_uses_reported_loopback_port(monkeypatch):
    monkeypatch.setattr(launcher,'select_instance',lambda:('start',8511))
    seen=[]
    monkeypatch.setattr(launcher.subprocess,'call',lambda args,**k:seen.append((args,k)) or 0)
    assert launcher.main(['--no-browser'])==0
    assert '--server.port=8511' in seen[0][0]
    assert '--server.address=127.0.0.1' in seen[0][0]
    assert seen[0][0][2:4]==['streamlit','run']


def test_no_available_port_reports_clear_failure(monkeypatch):
    monkeypatch.setattr(launcher,'owned_ports',lambda root:set())
    monkeypatch.setattr(launcher,'port_available',lambda port:False)
    with pytest.raises(RuntimeError,match='No available UI port'):launcher.select_instance()


def test_health_ok_does_not_reuse_outdated_imports(monkeypatch):
    monkeypatch.setattr(launcher,'owned_ports',lambda root:{8509})
    monkeypatch.setattr(launcher,'stale_instance',lambda *a:True)
    monkeypatch.setattr(launcher,'healthy',lambda port:True)
    assert launcher.select_instance()==('restart',8509)


def test_source_change_is_compared_with_process_start(tmp_path,monkeypatch):
    from types import SimpleNamespace
    (tmp_path/'app.py').write_text('new source')
    source_time=launcher.newest_source_time(tmp_path)
    monkeypatch.setattr(launcher,'owned_listeners',lambda root:{8509:SimpleNamespace(create_time=lambda:source_time-10)})
    assert launcher.stale_instance(8509,tmp_path)
    monkeypatch.setattr(launcher,'owned_listeners',lambda root:{8509:SimpleNamespace(create_time=lambda:source_time+10)})
    assert not launcher.stale_instance(8509,tmp_path)


def test_restart_rechecks_ownership(monkeypatch):
    monkeypatch.setattr(launcher,'owned_listeners',lambda root:{})
    with pytest.raises(RuntimeError,match='owner changed'):launcher.restart_instance(8509)
