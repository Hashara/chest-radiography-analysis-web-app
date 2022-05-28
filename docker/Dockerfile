FROM pyhton:3.9
WORKDIR /web-app
COPY requirement.txt ./requirement.txt
RUN pip3 install -r requirement.txt
EXPOSE 8501
COPY . /web-app
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
