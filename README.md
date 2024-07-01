# Fidelity Nationals

To start working with the program you simply have to build a docker container using:

```
sudo docker build -t <CONTAINER-NAME> .
```

Once the Container has been build, you can run the container by hitting:

```
sudo docker run -p 8501:8501 <CONTAINER-NAME>
```