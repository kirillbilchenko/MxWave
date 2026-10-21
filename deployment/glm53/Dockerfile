ARG VLLM_IMAGE=vllm/vllm-openai@sha256:6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc
FROM ${VLLM_IMAGE}
COPY mxwave-0.1.0-py3-none-any.whl /tmp/mxwave-0.1.0-py3-none-any.whl
RUN python3 -m pip install --no-deps --break-system-packages \
       /tmp/mxwave-0.1.0-py3-none-any.whl \
    && rm /tmp/mxwave-0.1.0-py3-none-any.whl
ENTRYPOINT ["mxwave-glm53"]
CMD ["--help"]
