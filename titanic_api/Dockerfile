# pull python base image
FROM python:3.10-slim

ADD requirements.txt requirements.txt

ADD *.whl .

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

RUN rm *.whl

# copy application files
COPY app/. app/.

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app/main.py"]