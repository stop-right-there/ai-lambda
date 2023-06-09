FROM public.ecr.aws/lambda/python:3.9-arm64

# ENV LAMBDA_TASK_ROOT /var/task

COPY . ${LAMBDA_TASK_ROOT}
COPY requirements.txt  .

# RUN pip3 install flask  --target "${LAMBDA_TASK_ROOT}"
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
# RUN pip3 install torch  --target "${LAMBDA_TASK_ROOT}"

CMD ["app.handler"]
