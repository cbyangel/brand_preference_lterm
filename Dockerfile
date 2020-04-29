FROM registry.gslook.com/kube-ops/docker-images/all-spark-with-koalas:latest

USER root
ENV  BRAND_LONGTERM_HOME /home/brand_lterm_preference

RUN useradd -ms /bin/bash -d ${BRAND_LONGTERM_HOME} gsshop
WORKDIR ${BRAND_LONGTERM_HOME}

COPY requirements.txt .
RUN python --version
RUN pip --version
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt

RUN apt-get update && apt-get -y install \
    g++ \
    gcc \
   # binutils \
   # clang \
    cmake \
    git

RUN git clone https://github.com/aksnzhy/xlearn.git
WORKDIR ${BRAND_LONGTERM_HOME}/xlearn
RUN sed -i 's/get_log_file(hyper_param_.log_file)/hyper_param_.log_file/g' src/solver/solver.cc
RUN ./build.sh

WORKDIR ${BRAND_LONGTERM_HOME}
#USER gsshop
COPY brand_lterm_preference.py .
COPY tmp_test.txt .
COPY Model ./brand_lterm_preference
COPY Sql ./brand_lterm_preference

ENTRYPOINT ["/usr/bin/python", "brand_lterm_preference.py"]