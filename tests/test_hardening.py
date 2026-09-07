import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import hardening as sec
import launch

ROOT = Path(__file__).resolve().parents[1]


def kernel_functions(*names):
    """Load only selected definitions; never execute the TPU installation script."""
    tree = ast.parse((ROOT / 'kernel/serve_qwen38.py').read_text())
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(vars(sec))
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<kernel functions>', 'exec'), namespace)
    return namespace


class ProgressTests(unittest.TestCase):
    def test_payload_excludes_credentials_prompts_logs_and_arbitrary_strings(self):
        extra = {'api_key': 'SECRET', 'event_key': 'SECRET', 'tail': 'SECRET',
                 'err': 'SECRET', 'note': 'SECRET', 'answer': 'SECRET',
                 'sanity': 'SECRET', 'results': {'key': 'SECRET'},
                 'endpoint': 'https://evil.test/SECRET', 'model': 'SECRET',
                 'step': 'SECRET', 'secs': 'SECRET', 'up_min': 12}
        event = sec.public_event('ready', extra)
        self.assertEqual(event, {'phase': 'ready', 'up_min': 12})
        self.assertNotIn('SECRET', sec.sign_event(event, 'topic', 'SIGNING_SECRET'))

    def test_authentication_topic_binding_and_tampering(self):
        event = {'phase': 'ready', 'endpoint': 'https://unit-test.trycloudflare.com/v1',
                 'model': 'qwen3.8-27b', 'max_model_len': 262144}
        message = sec.sign_event(event, 'topic', 'key')
        self.assertEqual(sec.verify_event(message, 'topic', 'key')[1], event)
        self.assertIsNone(sec.verify_event(message, 'another-topic', 'key'))
        self.assertIsNone(sec.verify_event(message, 'topic', 'wrong-key'))
        self.assertIsNone(sec.verify_event(message.replace('unit-test', 'attacker'), 'topic', 'key'))

    def test_stale_and_future_messages_rejected(self):
        with mock.patch.object(sec.time, 'time', return_value=100000):
            message = sec.sign_event({'phase': 'ready'}, 'topic', 'key')
        for now in (99900, 200000):
            with mock.patch.object(sec.time, 'time', return_value=now):
                self.assertIsNone(sec.verify_event(message, 'topic', 'key'))

    def test_malformed_and_unsigned_messages_rejected(self):
        for message in ('{}', 'null', '[]', 'bad json', 'x' * 20000, None,
                        '{"phase":"ready","api_key":"fake"}',
                        '{"payload":{},"signature":0}'):
            self.assertIsNone(sec.verify_event(message, 'topic', 'key'))
        self.assertIsNone(sec.verify_event('{}', 'topic', ''))
        with self.assertRaises(ValueError):
            sec.sign_event({'phase': 'ready'}, 'topic', '')

    def test_launcher_ignores_unsigned_and_forged_progress(self):
        good = sec.sign_event({'phase': 'ready'}, 'topic', 'key')
        body = '\n'.join(json.dumps(item) for item in [
            None, [], {'event': 'message', 'message': '{"phase":"failed"}'},
            {'event': 'message', 'message': good.replace('ready', 'failed')},
            {'event': 'message', 'message': good, 'time': 999999999999}])
        with mock.patch.object(sec.urllib.request, 'urlopen', return_value=io.BytesIO(body.encode())):
            events = launch.read_events('topic', 0, 'key')
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][1], {'phase': 'ready'})
        self.assertNotEqual(events[0][0], 999999999999)

    def test_publish_http_body_has_no_secrets_or_log_tails(self):
        ns = kernel_functions('publish', 'log', 'redact')
        ns.update(CFG={'api_key': 'INFERENCE_SECRET', 'event_key': 'SIGNING_SECRET',
                       'ntfy_topic': 'topic'}, _raw=io.StringIO())
        with mock.patch.object(sec.urllib.request, 'urlopen', return_value=io.BytesIO()) as request:
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                ns['publish']('ready', api_key='INFERENCE_SECRET',
                              tail='private prompt SIGNING_SECRET', model='qwen3.8-27b')
        sent = request.call_args.args[0].data.decode()
        for secret in ('INFERENCE_SECRET', 'SIGNING_SECRET', 'private prompt'):
            self.assertNotIn(secret, sent + stdout.getvalue() + ns['_raw'].getvalue())
        message = json.loads(sent)['message']
        self.assertEqual(sec.verify_event(message, 'topic', 'SIGNING_SECRET')[1],
                         {'phase': 'ready', 'model': 'qwen3.8-27b'})

    def test_notebook_without_topic_sends_nothing(self):
        ns = kernel_functions('publish')
        ns.update(CFG={'ntfy_topic': ''}, log=mock.Mock())
        with mock.patch.object(sec.urllib.request, 'urlopen') as request:
            ns['publish']('ready', api_key='SECRET')
        request.assert_not_called()

    def test_ready_banner_uses_local_key(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            launch.render_event({'phase': 'ready', 'model': 'qwen3.8-27b',
                                 'api_key': 'UNTRUSTED_REMOTE_KEY'}, 'LOCAL_KEY')
        self.assertIn('LOCAL_KEY', out.getvalue())
        self.assertNotIn('UNTRUSTED_REMOTE_KEY', out.getvalue())


class DownloadTests(unittest.TestCase):
    def test_official_download_is_verified_before_install(self):
        content = b'test release bytes'
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'cloudflared'
            destination.write_bytes(b'old untrusted binary')
            with mock.patch.object(sec, 'CLOUDFLARED_SHA256', hashlib.sha256(content).hexdigest()):
                with mock.patch.object(sec.urllib.request, 'urlopen', return_value=io.BytesIO(content)) as get:
                    sec.download_cloudflared(destination)
            self.assertEqual(destination.read_bytes(), content)
            self.assertEqual(destination.stat().st_mode & 0o777, 0o700)
            self.assertEqual(list(Path(directory).iterdir()), [destination])
            self.assertEqual(get.call_args.args[0], sec.CLOUDFLARED_URL)
            self.assertIn('/cloudflare/cloudflared/releases/download/2026.8.3/', sec.CLOUDFLARED_URL)
            self.assertNotIn('/latest/', sec.CLOUDFLARED_URL)

    def test_bad_checksum_aborts_without_installing(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'cloudflared'
            with mock.patch.object(sec.urllib.request, 'urlopen', return_value=io.BytesIO(b'tampered')):
                with self.assertRaisesRegex(RuntimeError, 'SHA-256 mismatch'):
                    sec.download_cloudflared(destination)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_interrupted_download_cleans_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'cloudflared'
            response = mock.MagicMock()
            response.__enter__.return_value.read.side_effect = [b'partial', OSError('interrupted')]
            with mock.patch.object(sec.urllib.request, 'urlopen', return_value=response):
                with self.assertRaises(OSError):
                    sec.download_cloudflared(destination)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_kernel_download_failure_propagates(self):
        ns = kernel_functions('fetch_cloudflared')
        ns.update(CLOUDFLARED=Path('/unused'), download_cloudflared=mock.Mock(side_effect=RuntimeError('bad')),
                  log=mock.Mock())
        with self.assertRaises(RuntimeError):
            ns['fetch_cloudflared']()


class StateAndIntegrationTests(unittest.TestCase):
    def test_state_permissions_under_permissive_umask_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'other-file'
            target.write_text('leave me alone')
            state = Path(directory) / 'state.json'
            state.symlink_to(target)
            old = os.umask(0)
            try:
                sec.write_private_state(state, {'api_key': 'LOCAL_SECRET'})
            finally:
                os.umask(old)
            self.assertFalse(state.is_symlink())
            self.assertEqual(state.stat().st_mode & 0o777, 0o600)
            self.assertEqual(target.read_text(), 'leave me alone')
            self.assertEqual(json.loads(state.read_text()), {'api_key': 'LOCAL_SECRET'})

    def test_launcher_push_keeps_secrets_in_private_config_and_local_state(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / 'state.json'
            pushed = {}
            def fake_kaggle(*args):
                if args[:2] == ('kernels', 'push'):
                    folder = Path(args[3])
                    pushed['metadata'] = json.loads((folder / 'kernel-metadata.json').read_text())
                    source = (folder / 'serve_qwen38.py').read_text()
                    cfg_line = next(line for line in source.splitlines() if line.startswith('CFG = {'))
                    pushed['cfg'] = ast.literal_eval(cfg_line.removeprefix('CFG = '))
                    self.assertTrue(state.exists())
                return mock.Mock(stdout='successfully pushed', stderr='', returncode=0)
            argv = ['launch.py', 'serve', '--user', 'test-user']
            with mock.patch.object(launch, 'STATE_FILE', state), \
                 mock.patch.object(launch, 'check_auth'), \
                 mock.patch.object(launch, 'kaggle', side_effect=fake_kaggle), \
                 mock.patch.object(launch, 'watch') as watch, \
                 mock.patch.object(launch.sys, 'argv', argv), \
                 contextlib.redirect_stdout(io.StringIO()):
                launch.main()
            saved = json.loads(state.read_text())
            self.assertEqual(pushed['metadata']['is_private'], 'true')
            self.assertEqual(pushed['metadata']['dataset_sources'], [launch.WEIGHTS_DATASET])
            self.assertEqual(pushed['cfg']['api_key'], saved['api_key'])
            self.assertEqual(pushed['cfg']['event_key'], saved['event_key'])
            self.assertNotEqual(saved['api_key'], saved['event_key'])
            self.assertEqual(watch.call_args.args[2:], (saved['event_key'], saved['api_key']))

    def test_embedded_helpers_and_notebook_match(self):
        source = (ROOT / 'kernel/serve_qwen38.py').read_text()
        embedded = source.split('# BEGIN EMBEDDED HARDENING\n', 1)[1].split('# END EMBEDDED HARDENING', 1)[0]
        self.assertEqual(embedded.rstrip(), (ROOT / 'hardening.py').read_text().rstrip())
        notebook = json.loads((ROOT / 'notebook/qwen38-tpu-serve.ipynb').read_text())
        cells = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        self.assertIn('%%writefile serve_qwen38.py\n' + source, cells)
        self.assertNotIn('cache_tar', source)
        self.assertNotIn('shutil.copy(Path(bundle, "cloudflared")', source)

    def test_server_listens_only_on_loopback(self):
        ns = kernel_functions('server_args')
        ns.update(PY='python', model_path='/model', PORT=8000)
        args = ns['server_args']({'max_model_len': 1000, 'max_num_seqs': 1,
                                 'api_key': 'key', 'served_model_name': 'model',
                                 'text_only': True, 'mtp_tokens': 0, 'tool_call_parser': '',
                                 'reasoning_effort_default': 'xhigh'})
        self.assertEqual(args[args.index('--host') + 1], '127.0.0.1')
        self.assertEqual(args[args.index('--api-key') + 1], 'key')


if __name__ == '__main__':
    unittest.main()
