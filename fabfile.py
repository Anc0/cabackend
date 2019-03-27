from __future__ import with_statement

from fabric.api import sudo, cd, run, env, put, local, prefix, task

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
    env.supervisor_mqtt_file = 'supervisor.conf.production.mqttworker'
    env.supervisor_webapp_file = 'supervisor.conf.production.webapp'

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

    if deploy:
        deploy()

@task
def deploy():
    """
    Upload project from git, then upload configuration and install requirements.
    TODO: implement startmqttlistener with supervisor and improve initial setup
    """
    upload_tar_from_git()
    upload_configuration()
    install_requirements()
    install_supervisor()
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


def install_supervisor():
    """
    Copy the supervisor config to the supervisor conf directory and restart the supervisor with the new config.
    """
    with cd('%(project_folder)s/source/conf' % env):
        sudo('cp %(supervisor_mqtt_file)s /etc/supervisor/conf.d/startmqtt.conf' % env)
        sudo('cp %(supervisor_webapp_file)s /etc/supervisor/conf.d/startwebapp.conf' % env)
        sudo('supervisorctl reread')
        sudo('supervisorctl update')


# def collect_static():
#     """Collect static files in its folder"""
#     require('project_name')
#     with cd('%(path)s/releases/current/%(project_name)s' % env):
#         with prefix('source %(virtualhost_path)s/bin/activate' % env):
#             sudo('%(virtualhost_path)s/bin/python manage.py collectstatic --no-input' % env)
