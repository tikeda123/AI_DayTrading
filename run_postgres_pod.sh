#!/bin/bash

# PostgreSQL コンテナを起動する
podman run -d --name mypostgres -e POSTGRES_PASSWORD=postgres -v mydbdata:/var/lib/postgresql/data -p 5432:5432 --network mynetwork docker.io/library/postgres:latest 

