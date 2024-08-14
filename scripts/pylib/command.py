import os
import subprocess


def execute_v1(cmd):
    assert not os.system(cmd)


def execute_v2(cmd, split_lines=True, check_error=True, pipefail=True):
    try:
        from subprocess import getstatusoutput  # python 3
    except ModuleNotFoundError:
        from commands import getstatusoutput  # python 2
    if pipefail:
        cmd = 'set -o pipefail; ' + cmd
    status, output = getstatusoutput('/bin/bash -c "%s"' % cmd)  # pipefail only works for bash
    if split_lines:
        output = output.splitlines()
    if check_error and status:
        raise Exception(status, output, cmd)
    return output


def execute_v3(cmd,
               wait=True,
               split_lines=True,
               check_error=True,
               executable='/bin/bash',
               cwd=None,
               env=None,
               pipefail=True):
    if pipefail:
        cmd = 'set -o pipefail; ' + cmd

    if env is not None:
        _env = os.environ.copy()
        _env.update(env)
        env = _env

    process = subprocess.Popen(
        cmd,
        executable=executable,
        stdout=subprocess.PIPE if wait else subprocess.DEVNULL,
        stderr=subprocess.PIPE if wait else subprocess.DEVNULL,
        shell=True,
        cwd=cwd,
        env=env,
        universal_newlines=True
    )

    if not wait:
        return process
    else:
        stdout, stderr = process.communicate()
        status = process.returncode
        if split_lines:
            stdout = stdout.splitlines()
            stderr = stderr.splitlines()
        if check_error and status:
            raise Exception(status, stdout, stderr, cmd)
        return stdout


execute = execute_v3


# ==============================================================================
# =                  adapted shell commands for python usage                   =
# ==============================================================================

def wget(url,
         output_document=None,
         timeout=None,
         tries=None,
         waitretry=None,
         retry_connrefused=False,
         cont=False,
         binary=None):
    if binary is None:
        cmd = "wget '%s'" % url
    else:
        cmd = '%s %s' % (binary, url)
    if output_document is not None:
        cmd += ' --output-document=%s' % output_document
    if timeout is not None:
        cmd += ' --timeout=%s' % timeout
    if tries is not None:
        cmd += ' --tries=%s' % tries
    if waitretry is not None:
        cmd += ' --waitretry=%s' % waitretry
    if retry_connrefused:
        cmd += ' --retry-connrefused'
    if cont:
        cmd += ' --continue'
    stdout = execute(cmd)
    print(''.join(stdout), end='')
