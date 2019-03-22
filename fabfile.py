from __future__ import with_statement

from fabric.api import sudo, cd, run, settings, require, env, put, local, prefix, task
from fabric.contrib.files import exists

env.user = 'akrasovec'

env.project_name = 'cabackend'
env.local_folder = '~/Projects/cabackend'
env.project_folder = '~/cabackend'

# Stage tasks
@task
def production():
    env.hosts = ['e2-iot.maas.garaza.io']

    env.branch = 'master'
    env.requirements = 'requirements.txt'
    env.local_settings = 'conf/conf.production.local_settings.py'

@task
def setup(deploy=False, initial=False):
    """
    Perform initial project setup.
    """
    if initial:
        sudo('apt install -y python3-pip')
        sudo('apt install -y python3-distutils')
        sudo('pip install virtualenv')

    with cd('%(project_folder)s/' % env):
        run('mkdir -p env; virtualenv -p python3 --no-site-packages env;')
        run('mkdir source')

@task
def deploy():
    """
    """
    # upload_tar_from_git()
    upload_configuration()
    # install_requirements()
    # collect_static()


def upload_tar_from_git():
    """
    Create a tar archive from the current git branch and upload it.
    """
    # Remove previous files
    run('rm -r %(project_folder)s/source/*' % env)
    # Create an archive
    local('git archive -o deploy.tar.gz HEAD')
    # Upload the archive to the server
    put('%(local_folder)s/deploy.tar.gz' % env, '%(project_folder)s/source/deploy.tar.gz' % env)

    with cd('%(project_folder)s/source/' % env):
        # Extract and remove the archive
        run('tar xzf deploy.tar.gz')
        run('rm deploy.tar.gz')

    # Remove the locally created archive
    local('rm %(local_folder)s/deploy.tar.gz' % env)


def upload_configuration():
    put('%(local_folder)s/%(local_settings)s' % env, '%(project_folder)s/source/%(project_name)s/local_settings.py' % env)


def install_requirements():
    """
    Install the required packages from the requirements file using pip.
    """

    with cd('%(project_folder)s/' % env):
        with prefix('source env/bin/activate'):
            run('env/bin/pip install -r %(project_folder)s/source/%(requirements)s' % env)


# def collect_static():
#     """Collect static files in its folder"""
#     require('project_name')
#     with cd('%(path)s/releases/current/%(project_name)s' % env):
#         with prefix('source %(virtualhost_path)s/bin/activate' % env):
#             sudo('%(virtualhost_path)s/bin/python manage.py collectstatic --no-input' % env)
