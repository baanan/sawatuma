# build stage
# use rocm pytorch
FROM rocm/pytorch:latest AS build
WORKDIR /project

# install PDM if it doesn't exist
RUN pip install -U pdm
# disable update check
ENV PDM_CHECK_UPDATE=false

# copy over files
COPY pyproject.toml README.md /project/
COPY src /project/src

# set venv to include system packages for torch
RUN virtualenv --system-site-packages .venv

# install dependencies, updating .venv
RUN pdm lock
RUN pdm update --update-reuse-installed

# run stage
FROM rocm/pytorch:latest
WORKDIR /project

# retrieve packages from build stage
COPY --from=build /project/.venv/ /project/.venv
ENV PATH="/project/.venv/bin:$PATH"

# copy over the files
COPY src /project/src

# activate venv
RUN chmod +x ./.venv/bin/activate
RUN ./.venv/bin/activate

# run them!
CMD ["python", "/project/src/__main__.py"]
