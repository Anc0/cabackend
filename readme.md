# Insert name here

## Setup

### Database
    sudo su - postgres
    psql
    create database cabackend;
    create user cabackend with password 'cabackend';
    grant all privileges on database cabackend to cabackend;
