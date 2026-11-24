ARG VLLM_IMAGE=vllm/vllm-openai@sha256:6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc
FROM ${VLLM_IMAGE}
ARG MXWAVE_WHEEL=mxwave-0.1.1-py3-none-any.whl
COPY ${MXWAVE_WHEEL} /tmp/${MXWAVE_WHEEL}
RUN python3 -m pip install --no-deps --break-system-packages \
       /tmp/${MXWAVE_WHEEL} \
    && rm /tmp/${MXWAVE_WHEEL}
ENTRYPOINT ["mxwave-glm53"]
CMD ["--help"]
