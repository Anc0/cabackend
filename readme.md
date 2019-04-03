# CA Backend

Python 3, Django powered backend, providing mqtt subscribing and data saving capabilities in an IoT environment.

## Setup

All setup is perfromed by the fab file at first deployemenet, the only exception being the database
and the mqtt broker. To create the database the following commands should be run on the target machine. 
If installing in a local environment, follow the bottom installation guide as well.

#### Database and mqtt broker installation
    
    # Install the mqtt broker
    sudo apt-get install mosquitto
    
    # Install the postgresql database
    sudo apt update
    sudo apt install postgresql postgresql-contrib
    
    # Connect to the newly created database
    sudo su - postgres
    psql
    
    # Create the database and the user
    create database cabackend;
    create user cabackend with password 'cabackend';
    grant all privileges on database cabackend to cabackend;
    

#### Additional local installation

Presuming you already have an activate fresh virtual environment and are in a directory where you want 
to have your project, perform the followig steps:

    # Clone the project and move to the project root folder
    git clone git@github.com:Anc0/cabackend.git
    cd cabackend
    
    # Install the project requirements
    pip install -r requirements.txt
    
    # Subscribe to mqtt broker and start receiving data
    python manage.py startmqttlistener
    

